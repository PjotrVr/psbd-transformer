"""Probe positions for PSBD: WHERE a perturbation is injected inside a network.

The companion module psbd.operators owns WHAT a probe does. This module owns
where it attaches, and knows nothing about the operator it plugs in beyond its
nn.Dropout-shaped interface. That separation is the central variable of the
study: the same operator at 2 different positions is 2 different experiments.

Every position is realized the same way, by attaching a fresh, independent
module at a named submodule boundary through a forward pre-hook or forward hook,
never by toggling a dropout the model already contains.

Why never reuse the model's own dropout. A trained dropout's inverted-scaling
factor was calibrated during training against the next layer's weights, so
switching it back on at inference conflates 2 different things: the model's own
regularization and PSBD's injected noise. Leaving every existing dropout (ViT's
embedding dropout, both architectures' MLP-internal dropouts, Swin's
stochastic_depth) at its natural eval identity and inserting separate modules
keeps the perturbation a clean, single-purpose probe. The one function that
deliberately breaks this, activate_model_dropout, exists to study the
conflation itself.

Why forward hooks rather than re-implementing a block's forward. A hook injects
at a module boundary without copying or knowing the surrounding control flow, so
adding a position never hand-duplicates a block class, and removing one is just
handle.remove() with nothing to restore, because the original model was never
mutated.

Two positions cannot be expressed as a hook, because the tensor they need never
crosses a module boundary. Both are realized by a removable per-instance forward
wrapper instead, which still mutates no weights:

  after_attention_residual  the residual stream right after the attention add is
                            a local variable consumed twice, once by the
                            MLP-branch norm and once by the second add, with no
                            module boundary in between. See attach_residual_wrapper.
  attention_heads           the per-head outputs live inside
                            F.multi_head_attention_forward, which reads
                            out_proj.weight directly and never calls out_proj as
                            a module. See attach_attention_wrapper.

Adding a position is 1 line in the architecture's registry below, plus a comment
saying what makes that boundary interesting, as long as a pre-hook or post-hook
can reach it.
"""

import functools
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn
from torch.utils.hooks import RemovableHandle
from torchvision.models.vision_transformer import EncoderBlock
from torchvision.models.swin_transformer import SwinTransformerBlock

from .models import network_core
from .operators import masked_attention_forward


@dataclass(frozen=True)
class PositionSpec:
    """One probe target inside a block, or once at model level.

    submodule_name is the dotted path to the module whose input (pre) or output
    (post) is perturbed. An empty submodule_name means the unit itself: the block
    for a block-scope position, which realizes after_mlp_residual by perturbing
    the block's own return value.

    hook_type is one of:
        "pre"        perturb the module's first positional input
        "post"       perturb the module's tensor output
        "residual"   swap the block's forward, the only way to reach the stream
                     immediately after the attention add
        "attention"  swap the attention forward, the only way to reach the head axis

    scope says where submodule_name resolves from:
        "block"  relative to each transformer block, so the position attaches
                 once per block (12 for ViT-B/16, 24 for Swin-S)
        "model"  from the network root, once, for a position that has no
                 per-block meaning such as the embedding dropout
        "root"   from the Sequential(Resize, network) wrapper, once, for a
                 position that has to sit outside the Resize
    """

    submodule_name: str
    hook_type: str
    scope: str = "block"


class ForwardRestore:
    """A remove()-able handle for a swapped forward, duck-typing RemovableHandle.

    Binding the replacement as an instance attribute shadows the class method
    without touching the class, so remove() only has to delete the attribute for
    the module to be exactly what it was. Shares unplug_dropout with the real
    hook handles, so callers never branch on which mechanism a position used.
    """

    def __init__(self, module: nn.Module):
        self.module = module

    def remove(self) -> None:
        # Deleting the instance attribute uncovers the class method again. Guarded
        # because unplug_dropout is safe to call twice on the same handle list.
        self.module.__dict__.pop("forward", None)


ProbeHandle = RemovableHandle | ForwardRestore


# ViT-B/16 EncoderBlock children: ln_1, self_attention, dropout, ln_2, mlp, one
# set per each of the 12 blocks. Model-scope entries resolve from the network
# root instead, which is where ViT's single embedding dropout sits.
VIT_POSITIONS: dict[str, PositionSpec] = {
    "after_embedding": PositionSpec("encoder.dropout", "pre", scope="model"),
    "before_attention_norm": PositionSpec("ln_1", "pre"),
    "before_attention": PositionSpec("self_attention", "pre"),
    # The per-head outputs never cross a module boundary, so a hook on out_proj
    # never fires (verified: the model output was bit-identical with one
    # attached). Reaching the head axis means recomputing attention.
    "attention_heads": PositionSpec("self_attention", "attention"),
    "before_attention_residual": PositionSpec("dropout", "post"),
    "before_mlp_norm": PositionSpec("ln_2", "pre"),
    # The PSBD paper's own ConvNet placement, dropout after the residual add. It
    # perturbs the stream right after x = x + input, so both the ln_2 branch and
    # the skip into x + y see the perturbation. Contrast before_mlp_norm, a
    # pre-hook on ln_2, which perturbs only the MLP branch's input and leaves
    # the skip untouched.
    "after_attention_residual": PositionSpec("", "residual"),
    "before_mlp": PositionSpec("mlp", "pre"),
    # mlp.3 receives the 3072-dim post-GELU hidden layer, so one channel there is
    # exactly one hidden neuron, which is what a structured operator masks.
    "mlp_neurons": PositionSpec("mlp.3", "pre"),
    "before_mlp_residual": PositionSpec("mlp", "post"),
    "after_mlp_residual": PositionSpec("", "post"),
    # LayerNorm OUTPUTS, which is where the IBD-PSC port acts. Scaling a
    # LayerNorm's gamma and beta together is exactly scaling its output, so
    # amplifying the affine parameters needs no weight mutation, only a
    # post-hook. Distinct from before_attention_norm / before_mlp_norm, which
    # are PRE-hooks on the same modules and perturb their inputs instead.
    "attention_norm_out": PositionSpec("ln_1", "post"),
    "mlp_norm_out": PositionSpec("ln_2", "post"),
    "final_norm_out": PositionSpec("encoder.ln", "post", scope="model"),
    # The raw normalized image, before the Resize wrapper, which is where the
    # SCALE-UP port acts. Root scope rather than model scope: model scope
    # resolves against network_core, which is inside the Resize, and SCALE-UP has
    # to clip at the original resolution.
    "input_pixels": PositionSpec("", "pre", scope="root"),
}

# Swin-S SwinTransformerBlock (V1) children: norm1, attn, stochastic_depth,
# norm2, mlp, one set per each of the 24 blocks. after_embedding targets the
# model-level patch-embed Sequential features.0, since no existing dropout sits
# there the way ViT's embedding dropout does.
SWIN_POSITIONS: dict[str, PositionSpec] = {
    "after_embedding": PositionSpec("features.0", "post", scope="model"),
    "before_attention_norm": PositionSpec("norm1", "pre"),
    "before_attention": PositionSpec("attn", "pre"),
    # Hooks attn and mlp directly, NOT stochastic_depth: self.stochastic_depth is
    # one instance called twice per block (x + stochastic_depth(attn(...)), then
    # x + stochastic_depth(mlp(...))), so a hook on it cannot tell which branch
    # invoked it. Hooking attn and mlp is unambiguous, at the cost of the probe
    # landing just before stochastic_depth sees the branch output rather than
    # just after it. That is the only ViT/Swin asymmetry, and stochastic_depth
    # itself stays untouched.
    "before_attention_residual": PositionSpec("attn", "post"),
    "before_mlp_norm": PositionSpec("norm2", "pre"),
    "after_attention_residual": PositionSpec("", "residual"),
    "before_mlp": PositionSpec("mlp", "pre"),
    "before_mlp_residual": PositionSpec("mlp", "post"),
    "after_mlp_residual": PositionSpec("", "post"),
    # Swin equivalents of the ViT ported positions. norm1/norm2 are Swin's
    # per-block LayerNorms (ViT calls them ln_1/ln_2). The final norm is
    # SwinTransformer.norm rather than ViT's encoder.ln.
    "attention_norm_out": PositionSpec("norm1", "post"),
    "mlp_norm_out": PositionSpec("norm2", "post"),
    "final_norm_out": PositionSpec("norm", "post", scope="model"),
    "input_pixels": PositionSpec("", "pre", scope="root"),
}

POSITION_REGISTRY: dict[str, dict[str, PositionSpec]] = {
    "vit": VIT_POSITIONS,
    "swin": SWIN_POSITIONS,
}

BLOCK_TYPES: dict[str, tuple[type, ...]] = {
    "vit": (EncoderBlock,),
    "swin": (SwinTransformerBlock,),
}

# The 9 atomic positions swept in isolation, in forward order through a block.
SINGLE_POSITION_NAMES: tuple[str, ...] = (
    "after_embedding",
    "before_attention_norm",
    "before_attention",
    "before_attention_residual",
    "after_attention_residual",
    "before_mlp_norm",
    "before_mlp",
    "before_mlp_residual",
    "after_mlp_residual",
)

# Positions where the channel axis indexes a transformer unit rather than an
# arbitrary feature, so a structured operator masks whole heads or whole neurons.
# Kept out of SINGLE_POSITION_NAMES because that tuple defines the 9-position
# placement study, and adding to it would silently change what every existing
# sweep over "all single positions" covers.
STRUCTURED_POSITION_NAMES: tuple[str, ...] = (
    "attention_heads",
    "mlp_neurons",
)

# Positions that exist to host the ported detectors rather than the placement
# study. Kept separate for the same reason.
PORTED_POSITION_NAMES: tuple[str, ...] = (
    "attention_norm_out",
    "mlp_norm_out",
    "final_norm_out",
    "input_pixels",
)

# Named multi-position combos. Every single position is also usable directly as a
# 1-element position_names tuple, so it needs no entry here.
#
# pre_residual perturbs each branch's contribution just before it is added, so the
# residual stream itself is never touched. post_residual perturbs the stream
# immediately after each of the 2 adds, the ConvNet placement of the PSBD paper.
# These 2 are the primary comparison this study exists to make.
DROPOUT_CONFIGS: dict[str, tuple[str, ...]] = {
    "pre_residual": ("before_attention_residual", "before_mlp_residual"),
    "post_residual": ("after_attention_residual", "after_mlp_residual"),
    "both_sublayer_inputs": ("before_attention_norm", "before_mlp_norm"),
}


def plug_dropout(
    model: nn.Module,
    architecture: str,
    position_names: tuple[str, ...],
    dropout_factory: dict[str, Callable[[float], nn.Module]],
    rate: float,
    block_range: tuple[int, int] | None = None,
) -> list[ProbeHandle]:
    """Attach a fresh probe at every named position, in every block.

    dropout_factory maps a position name to a constructor taking the rate,
    defaulting any unlisted position to nn.Dropout, so the perturbation kind is
    overridable per position without touching the plug mechanics. Returns one
    handle per attachment (12 or 24 per block-scope position, 1 per model-scope
    position), all removed together by unplug_dropout.

    block_range restricts block-scope positions to a contiguous 1-indexed
    inclusive span, so a placement can be aimed at the depth where a given
    attack's backdoor direction actually lives rather than applied uniformly.
    """
    positions = POSITION_REGISTRY[architecture]
    block_types = BLOCK_TYPES[architecture]
    core = network_core(model)

    handles: list[ProbeHandle] = []
    try:
        for name in position_names:
            spec = positions[name]
            factory = dropout_factory.get(name, nn.Dropout)
            for target in resolve_targets(model, core, spec, block_types, block_range):
                handles.append(
                    _attach_probe(target, spec.hook_type, architecture, factory, rate)
                )
    except Exception:
        # Handles registered before the failure are otherwise unreachable, and an
        # orphaned probe silently compounds with the next rate's.
        unplug_dropout(handles)
        raise

    return handles


def unplug_dropout(handles: list[ProbeHandle]) -> None:
    """Undo every attachment plug_dropout made, restoring the loaded model exactly."""
    for handle in handles:
        handle.remove()


def activate_model_dropout(
    model: nn.Module, rate: float
) -> list[tuple[nn.Module, float]]:
    """Switch the model's OWN dropout modules on, returning what to restore.

    Everything else in this file deliberately avoids touching the model's own
    dropouts, because a trained dropout's scaling was calibrated against the next
    layer's weights and reusing it conflates the model's regularization with
    PSBD's probe. This function exists to study exactly that conflation: what
    PSBD measures on a model whose own dropout is live, with the probe stacked on
    top.

    Skips *.encoder.dropout for the same reason the position registry treats it
    separately: it is the embedding dropout applied once before the block stack,
    not a per-block placement, so including it would change the depth profile of
    the disturbance rather than its magnitude.

    Removal compounds. A probe at p_probe on top of a model dropout at p_model
    leaves (1 - p_model)(1 - p_probe) alive, so the nominal probe rate no longer
    describes the disturbance and only measured shift ratio does.
    """
    restore: list[tuple[nn.Module, float]] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Dropout) or name.endswith("encoder.dropout"):
            continue
        restore.append((module, module.p))
        module.p = rate
        module.train()

    if not restore:
        raise ValueError(
            "no nn.Dropout modules found outside the embedding dropout, so model "
            "dropout cannot be activated for this architecture"
        )

    return restore


def restore_model_dropout(restore: list[tuple[nn.Module, float]]) -> None:
    """Put every rate back and return the modules to eval, exactly as loaded."""
    for module, rate in restore:
        module.p = rate
        module.eval()


def resolve_targets(
    model: nn.Module,
    core: nn.Module,
    spec: PositionSpec,
    block_types: tuple[type, ...],
    block_range: tuple[int, int] | None = None,
) -> list[nn.Module]:
    """Every module a position attaches to: one per block, or one at model level.

    block_range restricts a block-scope position to a contiguous span of blocks,
    1-indexed and inclusive, matching the layer numbering the latent analysis uses
    (block 1 produces layer-1 features). None means every block, the default.

    The restriction exists because where a trigger's backdoor direction reaches the
    CLS token is attack-dependent: measured on CIFAR-10 ViT, blend arrives by layer
    5 and a static patch trigger not until layer 9. Perturbing all 12 blocks cannot
    distinguish "this position matters" from "this depth matters", and those are
    different claims.
    """
    if spec.scope == "root":
        if block_range is not None:
            raise ValueError(
                "input_pixels is model-wide, so a block range is meaningless"
            )
        return [model]

    if spec.scope == "model":
        if block_range is not None:
            raise ValueError(
                f"position {spec.submodule_name!r} is model-scope, so it attaches "
                "once and a block range is meaningless for it"
            )
        model_level_target = core.get_submodule(spec.submodule_name)
        return [model_level_target]

    blocks = [module for module in model.modules() if isinstance(module, block_types)]
    if not blocks:
        # Silence here would be the worst failure mode available: no attachment
        # means no perturbation, so PSU is identically 0 and the position reads
        # as "had no effect" rather than as a model that never matched.
        raise ValueError(
            f"no {block_types} blocks found in the model, so position "
            f"{spec.submodule_name!r} attached nothing"
        )

    if block_range is not None:
        first, last = block_range
        if not 1 <= first <= last <= len(blocks):
            raise ValueError(
                f"block range {block_range} is outside 1..{len(blocks)} for this "
                "architecture"
            )
        blocks = blocks[first - 1 : last]

    targets = [
        block if spec.submodule_name == "" else block.get_submodule(spec.submodule_name)
        for block in blocks
    ]
    return targets


def _attach_probe(
    target: nn.Module,
    hook_type: str,
    architecture: str,
    dropout_factory: Callable[[float], nn.Module],
    rate: float,
) -> ProbeHandle:
    """Build a fresh perturbation module and attach it at target, returning its handle."""
    if hook_type == "residual":
        residual_handle = attach_residual_wrapper(
            target, architecture, dropout_factory, rate
        )
        return residual_handle
    if hook_type == "attention":
        attention_handle = attach_attention_wrapper(target, dropout_factory, rate)
        return attention_handle

    probe = dropout_factory(rate)
    # Not part of the model tree, so model.eval() never reaches it. Train mode is
    # set explicitly so the mask is actually sampled during the stochastic passes.
    probe.train()

    if hook_type == "pre":
        pre_handle = target.register_forward_pre_hook(_make_pre_hook(probe))
        return pre_handle
    if hook_type == "post":
        post_handle = target.register_forward_hook(_make_post_hook(probe))
        return post_handle
    raise ValueError(f"Unknown hook type: {hook_type}")


def _make_pre_hook(probe: nn.Module) -> Callable:
    """Perturb a module's positional input before it runs.

    When several positional args are the same tensor object (ViT's
    self_attention receives x as q, k, and v), one mask is drawn and shared, so
    q, k, and v stay identical after perturbation.
    """

    def pre_hook(module, args):
        if not args or not isinstance(args[0], torch.Tensor):
            raise TypeError(
                f"{type(module).__name__} was called with no positional tensor, "
                "so this position cannot perturb its input"
            )
        perturbed = probe(args[0])
        perturbed_args = tuple(perturbed if arg is args[0] else arg for arg in args)
        return perturbed_args

    return pre_hook


def _make_post_hook(probe: nn.Module) -> Callable:
    """Perturb a module's tensor output after it runs."""

    def post_hook(module, args, output):
        if not isinstance(output, torch.Tensor):
            # ViT's self_attention returns (output, weights). A post position
            # on any such module is a registry mistake, not a runtime condition.
            raise TypeError(
                f"{type(module).__name__} returned {type(output).__name__}, not a "
                "Tensor, so it cannot carry a post-hook dropout position"
            )
        perturbed = probe(output)
        return perturbed

    return post_hook


def attach_residual_wrapper(
    block: nn.Module,
    architecture: str,
    dropout_factory: Callable[[float], nn.Module],
    rate: float,
) -> ForwardRestore:
    """Swap in the block forward that perturbs the stream after the attention add.

    The first of the 2 positions no hook can express. Mutates no weights, so
    remove() restores the loaded model exactly.
    """
    probe = dropout_factory(rate)
    probe.train()
    block.forward = functools.partial(RESIDUAL_FORWARDS[architecture], block, probe)
    handle = ForwardRestore(block)
    return handle


def attach_attention_wrapper(
    attention: nn.Module,
    dropout_factory: Callable[[float], nn.Module],
    rate: float,
) -> ForwardRestore:
    """Swap in the attention forward that exposes the head axis.

    The second position a hook cannot express, for the same reason as the
    residual one: the per-head outputs are a local inside
    F.multi_head_attention_forward and never cross a module boundary. Mutates no
    weights, so remove() restores the loaded model exactly.
    """
    probe = dropout_factory(rate)
    probe.train()
    attention.forward = functools.partial(masked_attention_forward, attention, probe)
    handle = ForwardRestore(attention)
    return handle


def _vit_post_attention_residual_forward(
    block: nn.Module, probe: nn.Module, input_tensor
):
    """EncoderBlock.forward with the probe on the stream after the attention add.

    Mirrors torchvision's EncoderBlock.forward exactly except for the one probe
    call. The MLP-branch add is left alone: after_mlp_residual is a plain
    post-hook on the block, so plugging both positions composes into the full
    post-residual placement without either mechanism knowing about the other.
    """
    x = block.ln_1(input_tensor)
    x, _ = block.self_attention(x, x, x, need_weights=False)
    x = block.dropout(x)
    x = probe(x + input_tensor)
    y = block.ln_2(x)
    y = block.mlp(y)

    out = x + y
    return out


def _swin_post_attention_residual_forward(
    block: nn.Module, probe: nn.Module, input_tensor
):
    """SwinTransformerBlock.forward with the probe after the attention add."""
    x = probe(
        input_tensor + block.stochastic_depth(block.attn(block.norm1(input_tensor)))
    )
    x = x + block.stochastic_depth(block.mlp(block.norm2(x)))
    return x


RESIDUAL_FORWARDS: dict[str, Callable] = {
    "vit": _vit_post_attention_residual_forward,
    "swin": _swin_post_attention_residual_forward,
}
