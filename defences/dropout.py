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

import torch.nn as nn
from torch.utils.hooks import RemovableHandle
from torchvision.models.vision_transformer import EncoderBlock
from torchvision.models.swin_transformer import SwinTransformerBlock

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
    hook_type: str  # "pre", "post", or "residual"
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
VIT_POSITIONS: dict[str, PositionSpec] = {
    "after_embedding": PositionSpec("encoder.dropout", "pre", scope="model"),
    "before_attention_norm": PositionSpec("ln_1", "pre"),
    "before_attention": PositionSpec("self_attention", "pre"),
    "before_attention_residual": PositionSpec("dropout", "post"),
    "before_mlp_norm": PositionSpec("ln_2", "pre"),
    "after_attention_residual": PositionSpec("", "residual"),
    "before_mlp": PositionSpec("mlp", "pre"),
    "before_mlp_residual": PositionSpec("mlp", "post"),
    "after_mlp_residual": PositionSpec("", "post"),
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


def _make_pre_hook(dropout: nn.Module) -> Callable:
    """Perturb a module's positional input before it runs.

    When several positional args are the same tensor object (ViT's
    self_attention receives x as q, k, and v), one dropout mask is drawn and
    shared, so q, k, and v stay identical after perturbation.
    """

    def pre_hook(module, args):
        perturbed = dropout(args[0])
        return tuple(perturbed if arg is args[0] else arg for arg in args)

    return pre_hook


def _make_post_hook(dropout: nn.Module) -> Callable:
    """Perturb a module's tensor output after it runs."""

    def post_hook(module, args, output):
        return dropout(output)

    return post_hook


def _resolve_targets(
    model: nn.Module,
    core: nn.Module,
    spec: PositionSpec,
    block_types: tuple[type, ...],
) -> list[nn.Module]:
    """Every module a position attaches to: one per block, or one at model level."""
    if spec.scope == "model":
        return [core.get_submodule(spec.submodule_name)]
    targets = []
    for module in model.modules():
        if isinstance(module, block_types):
            target = (
                module
                if spec.submodule_name == ""
                else module.get_submodule(spec.submodule_name)
            )
            targets.append(target)
    return targets


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
) -> list[RemovableHandle | _ForwardRestore]:
    """Attach a fresh dropout at every named position, in every block.

    dropout_factory maps a position name to a constructor taking the rate,
    defaulting any unlisted position to nn.Dropout, so the perturbation kind is
    overridable per position without touching the plug mechanics. Returns one
    handle per attachment (12 or 24 per block-scope position, 1 per model-scope
    position), all removed together by unplug_dropout.
    """
    positions = POSITION_REGISTRY[architecture]
    block_types = BLOCK_TYPES[architecture]
    core = network_core(model)

    handles: list[RemovableHandle | _ForwardRestore] = []
    for name in position_names:
        spec = positions[name]
        factory = dropout_factory.get(name, nn.Dropout)
        for target in _resolve_targets(model, core, spec, block_types):
            handles.append(
                _attach_hook(target, spec.hook_type, architecture, factory, rate)
            )
    return handles


def unplug_dropout(handles: list[RemovableHandle | _ForwardRestore]) -> None:
    """Undo every attachment plug_dropout made, restoring the loaded model exactly."""
    for handle in handles:
        handle.remove()
