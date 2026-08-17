"""Dropout-position registry for PSBD, inserted by forward hooks.

The central variable of study is where inside a transformer block PSBD's
perturbation is injected. Every position is realized the same way: a fresh,
independent dropout module attached at a named submodule boundary through a
forward pre-hook or forward hook, never by toggling a dropout the model already
contains.

Why never reuse the model's own dropout. A trained dropout's inverted-scaling
factor was calibrated during training against the next layer's weights, so
switching it back on at inference conflates two different things: the model's
own regularization and PSBD's injected noise. Leaving every existing dropout
(ViT's embedding dropout, both architectures' MLP-internal dropouts, Swin's
stochastic_depth) at its natural eval identity and inserting separate modules
keeps the perturbation a clean, single-purpose probe.

Why forward hooks rather than re-implementing a block's forward. A hook injects
at a module boundary without copying or knowing the surrounding control flow, so
adding a position never hand-duplicates a block class, and removing one is just
handle.remove() with nothing to restore, because the original model was never
mutated.

The one position a hook cannot express is dropout on the residual stream
immediately after the attention add. In both architectures that value is a local
variable consumed twice (once by the MLP-branch norm, once by the second add),
and no module boundary sits between the add and those two uses. That position is
therefore realized by a removable per-instance forward wrapper instead; see
_attach_residual_wrapper. It still mutates no weights, so unplugging restores the
loaded model exactly.

Both architecture registries are defined here. This pass runs ViT only.
"""

import functools
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn
from torch.utils.hooks import RemovableHandle
from torchvision.models.vision_transformer import EncoderBlock
from torchvision.models.swin_transformer import SwinTransformerBlock

from defences.perturbations import masked_attention_forward
from models import network_core


@dataclass(frozen=True)
class PositionSpec:
    """One hook target inside a block, or once at model level.

    submodule_name is the dotted path to the module whose input (pre) or output
    (post) is perturbed, resolved either relative to each transformer block
    (scope "block") or from the network root once (scope "model"). An empty
    submodule_name means the unit itself: the block for a block-scope position,
    which realizes after_mlp_residual by perturbing the block's own return value.

    scope is not in the plan's two-field sketch, but the one model-level position
    (after_embedding) cannot be expressed relative to a block, so it carries the
    scope that tells plug_dropout to resolve it from the network root instead.
    """

    submodule_name: str
    hook_type: str  # "pre", "post", "residual", or "attention"
    scope: str = "block"


# ViT-B/16 EncoderBlock children: ln_1, self_attention, dropout, ln_2, mlp, one
# per each of 12 blocks. after_embedding targets the model-level Encoder.dropout.
#
# after_attention_residual is the one "residual" position: it perturbs the stream
# right after x = x + input, so both the ln_2 branch and the skip into x + y see
# the perturbation. That is what the PSBD paper's ConvNet placement does (dropout
# after the residual add, before the activation) and it is not reachable with a
# hook, since x is a local consumed twice. Contrast before_mlp_norm, a pre-hook on
# ln_2, which perturbs only the MLP branch's input and leaves the skip untouched.
#
# attention_heads and mlp_neurons exist for the structured operators in
# defences.perturbations, and are the only two positions where a transformer's
# own units are separable along the channel axis:
#
#   attention_heads is the second position no hook can express. The per-head
#   outputs exist only inside F.multi_head_attention_forward, which reads
#   out_proj.weight directly and never calls out_proj as a module, so a hook on
#   out_proj never fires (verified: the model output was bit-identical). It
#   therefore uses a forward wrapper that recomputes attention and exposes the
#   head axis; see _attach_attention_wrapper.
#
#   mlp_neurons is an ordinary pre-hook. mlp.3 receives the 3072-dim post-GELU
#   hidden layer, so one channel there is exactly one hidden neuron.
VIT_POSITIONS: dict[str, PositionSpec] = {
    "after_embedding": PositionSpec("encoder.dropout", "pre", scope="model"),
    "before_attention_norm": PositionSpec("ln_1", "pre"),
    "before_attention": PositionSpec("self_attention", "pre"),
    "attention_heads": PositionSpec("self_attention", "attention"),
    "before_attention_residual": PositionSpec("dropout", "post"),
    "before_mlp_norm": PositionSpec("ln_2", "pre"),
    "after_attention_residual": PositionSpec("", "residual"),
    "before_mlp": PositionSpec("mlp", "pre"),
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
    # SCALE-UP port acts. Model-root scope rather than "model": the position
    # registry resolves model scope against network_core, which is inside the
    # Resize, and SCALE-UP has to clip at the original resolution.
    "input_pixels": PositionSpec("", "pre", scope="root"),
}

# Swin-S SwinTransformerBlock (V1) children: norm1, attn, stochastic_depth,
# norm2, mlp, one per each of 24 blocks. after_embedding targets the model-level
# patch-embed Sequential features.0 (no existing dropout sits there, unlike ViT).
#
# before_attention_residual and before_mlp_residual hook attn and mlp directly
# (post), not stochastic_depth: self.stochastic_depth is one instance called
# twice per block (x + stochastic_depth(attn(...)), then x + stochastic_depth(
# mlp(...))), so a hook on it cannot tell which branch invoked it. Hooking attn
# and mlp is unambiguous, at the cost of our dropout landing just before
# stochastic_depth sees the branch output rather than just after it. That is the
# only ViT/Swin asymmetry, and stochastic_depth itself stays untouched.
SWIN_POSITIONS: dict[str, PositionSpec] = {
    "after_embedding": PositionSpec("features.0", "post", scope="model"),
    "before_attention_norm": PositionSpec("norm1", "pre"),
    "before_attention": PositionSpec("attn", "pre"),
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
# study. Kept separate so a sweep over "all single positions" keeps meaning what
# it meant before these were added.
PORTED_POSITION_NAMES: tuple[str, ...] = (
    "attention_norm_out",
    "mlp_norm_out",
    "final_norm_out",
    "input_pixels",
)

# Named multi-position combos. Every single position is also usable directly as a
# one-element position_names tuple, so it needs no entry here.
#
# pre_residual perturbs each branch's contribution just before it is added, so the
# residual stream itself is never touched. post_residual perturbs the stream
# immediately after each of the two adds, the ConvNet placement of the PSBD paper.
# These two are the primary comparison this study exists to make.
DROPOUT_CONFIGS: dict[str, tuple[str, ...]] = {
    "pre_residual": ("before_attention_residual", "before_mlp_residual"),
    "post_residual": ("after_attention_residual", "after_mlp_residual"),
    "both_sublayer_inputs": ("before_attention_norm", "before_mlp_norm"),
}


def _vit_post_attention_residual_forward(
    block: nn.Module, dropout: nn.Module, input_tensor
):
    """EncoderBlock.forward with dropout on the stream after the attention add.

    Mirrors torchvision's EncoderBlock.forward exactly except for the one
    dropout call. The MLP-branch add is left alone: after_mlp_residual is a
    plain post-hook on the block, so plugging both positions composes into the
    full post-residual placement without either mechanism knowing about the other.
    """
    x = block.ln_1(input_tensor)
    x, _ = block.self_attention(x, x, x, need_weights=False)
    x = block.dropout(x)
    x = dropout(x + input_tensor)
    y = block.ln_2(x)
    y = block.mlp(y)
    return x + y


def _swin_post_attention_residual_forward(
    block: nn.Module, dropout: nn.Module, input_tensor
):
    """SwinTransformerBlock.forward with dropout after the attention add."""
    x = dropout(
        input_tensor + block.stochastic_depth(block.attn(block.norm1(input_tensor)))
    )
    x = x + block.stochastic_depth(block.mlp(block.norm2(x)))
    return x


RESIDUAL_FORWARDS: dict[str, Callable] = {
    "vit": _vit_post_attention_residual_forward,
    "swin": _swin_post_attention_residual_forward,
}


class _ForwardRestore:
    """A remove()-able handle for a swapped forward, duck-typing RemovableHandle.

    Binding the replacement as an instance attribute shadows the class method
    without touching the class, so remove() only has to delete the attribute for
    the block to be exactly what it was. Shares unplug_dropout with the real hook
    handles, so callers never branch on which mechanism a position used.
    """

    def __init__(self, block: nn.Module):
        self.block = block

    def remove(self) -> None:
        # Deleting the instance attribute uncovers the class method again. Guarded
        # because unplug_dropout is safe to call twice on the same handle list.
        self.block.__dict__.pop("forward", None)


def _attach_residual_wrapper(
    block: nn.Module,
    architecture: str,
    dropout_factory: Callable[[float], nn.Module],
    rate: float,
) -> _ForwardRestore:
    dropout = dropout_factory(rate)
    dropout.train()
    block.forward = functools.partial(RESIDUAL_FORWARDS[architecture], block, dropout)
    return _ForwardRestore(block)


def _attach_attention_wrapper(
    attention: nn.Module,
    dropout_factory: Callable[[float], nn.Module],
    rate: float,
) -> _ForwardRestore:
    """Swap in the attention forward that exposes the head axis.

    The second position a hook cannot express, for the same reason as the
    residual one: the per-head outputs are a local inside
    F.multi_head_attention_forward and never cross a module boundary. Mutates no
    weights, so remove() restores the loaded model exactly.
    """
    mask = dropout_factory(rate)
    mask.train()
    attention.forward = functools.partial(masked_attention_forward, attention, mask)
    return _ForwardRestore(attention)


def _make_pre_hook(dropout: nn.Module) -> Callable:
    """Perturb a module's positional input before it runs.

    When several positional args are the same tensor object (ViT's
    self_attention receives x as q, k, and v), one dropout mask is drawn and
    shared, so q, k, and v stay identical after perturbation.
    """

    def pre_hook(module, args):
        if not args or not isinstance(args[0], torch.Tensor):
            raise TypeError(
                f"{type(module).__name__} was called with no positional tensor, "
                "so this position cannot perturb its input"
            )
        perturbed = dropout(args[0])
        return tuple(perturbed if arg is args[0] else arg for arg in args)

    return pre_hook


def _make_post_hook(dropout: nn.Module) -> Callable:
    """Perturb a module's tensor output after it runs."""

    def post_hook(module, args, output):
        if not isinstance(output, torch.Tensor):
            # ViT's self_attention returns (output, weights); a post position on
            # any such module is a registry mistake, not a runtime condition.
            raise TypeError(
                f"{type(module).__name__} returned {type(output).__name__}, not a "
                "Tensor, so it cannot carry a post-hook dropout position"
            )
        return dropout(output)

    return post_hook


def _resolve_targets(
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
            raise ValueError("input_pixels is model-wide; a block range is meaningless")
        return [model]
    if spec.scope == "model":
        if block_range is not None:
            raise ValueError(
                f"position {spec.submodule_name!r} is model-scope, so it attaches "
                "once and a block range is meaningless for it"
            )
        return [core.get_submodule(spec.submodule_name)]

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

    return [
        block if spec.submodule_name == "" else block.get_submodule(spec.submodule_name)
        for block in blocks
    ]


def _attach_hook(
    target: nn.Module,
    hook_type: str,
    architecture: str,
    dropout_factory: Callable[[float], nn.Module],
    rate: float,
) -> RemovableHandle | _ForwardRestore:
    """Build a fresh dropout and attach it at target, returning its handle."""
    if hook_type == "residual":
        return _attach_residual_wrapper(target, architecture, dropout_factory, rate)
    if hook_type == "attention":
        return _attach_attention_wrapper(target, dropout_factory, rate)

    dropout = dropout_factory(rate)
    # Not part of the model tree, so model.eval() never reaches it. Train mode is
    # set explicitly so the mask is actually sampled during the stochastic passes.
    dropout.train()
    if hook_type == "pre":
        return target.register_forward_pre_hook(_make_pre_hook(dropout))
    if hook_type == "post":
        return target.register_forward_hook(_make_post_hook(dropout))
    raise ValueError(f"Unknown hook type: {hook_type}")


def plug_dropout(
    model: nn.Module,
    architecture: str,
    position_names: tuple[str, ...],
    dropout_factory: dict[str, Callable[[float], nn.Module]],
    rate: float,
    block_range: tuple[int, int] | None = None,
) -> list[RemovableHandle | _ForwardRestore]:
    """Attach a fresh dropout at every named position, in every block.

    dropout_factory maps a position name to a constructor taking the rate,
    defaulting any unlisted position to nn.Dropout, so the perturbation kind is
    overridable per position without touching the plug mechanics. Returns one
    handle per attachment (12 or 24 per block-scope position, 1 per model-scope
    position), all removed together by unplug_dropout.

    block_range restricts block-scope positions to a contiguous 1-indexed inclusive
    span, so a placement can be aimed at the depth where a given attack's backdoor
    direction actually lives rather than applied uniformly.
    """
    positions = POSITION_REGISTRY[architecture]
    block_types = BLOCK_TYPES[architecture]
    core = network_core(model)

    handles: list[RemovableHandle | _ForwardRestore] = []
    try:
        for name in position_names:
            spec = positions[name]
            factory = dropout_factory.get(name, nn.Dropout)
            for target in _resolve_targets(model, core, spec, block_types, block_range):
                handles.append(
                    _attach_hook(target, spec.hook_type, architecture, factory, rate)
                )
    except Exception:
        # Handles registered before the failure are otherwise unreachable, and an
        # orphaned dropout silently compounds with the next rate's.
        unplug_dropout(handles)
        raise
    return handles


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


def unplug_dropout(handles: list[RemovableHandle | _ForwardRestore]) -> None:
    """Undo every attachment plug_dropout made, restoring the loaded model exactly."""
    for handle in handles:
        handle.remove()
