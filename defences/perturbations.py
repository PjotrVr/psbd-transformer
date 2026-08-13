"""Perturbation operators for PSBD, beyond element-wise dropout.

PSBD inherits Bernoulli dropout from a ConvNet setting, where a channel is the
natural unit. A transformer has several natural units and dropout is not aligned
with any of them: it samples an independent mask per (token, channel), so it
never removes an attention head, a hidden neuron, or a token as a whole. It
removes a little of everything instead.

That distinction is the point. A backdoor shortcut is carried by specific
structures, and a perturbation that dissolves uniformly across all of them
measures something different from one that removes a structure at a time. The
operators here are the structured alternatives, all sharing dropout's interface
so any of them can be plugged at any position in the registry.

Every operator follows nn.Dropout's two conventions, because PSU compares a
perturbed forward pass against an unperturbed one and any deviation from them
would show up as signal:

  1. Identity when self.training is False, or when rate is 0.
  2. Inverted scaling. Survivors are divided by the keep probability so the
     expected activation is unchanged, which keeps the unperturbed and perturbed
     passes on the same scale.

Shape convention throughout is (batch, tokens, channels), which is what ViT
carries with batch_first=True and what every registry position sees.
"""

import torch
import torch.nn as nn


def _keep_scale(rate: float) -> float:
    """Inverted-dropout scale. Guarded because rate 1.0 would divide by zero."""
    keep = 1.0 - rate
    if keep <= 0.0:
        raise ValueError(f"rate {rate} removes everything, leaving nothing to scale")
    return 1.0 / keep


class GroupChannelMask(nn.Module):
    """Zero whole groups of channels, shared across every token of a sample.

    group_size 1 masks individual embedding channels, which at the mlp_neurons
    position means whole hidden neurons of the MLP.

    Sharing the mask across tokens is the substantive difference from nn.Dropout.
    A neuron either contributes to the sample or it does not; dropping it
    independently per token leaves it partly alive everywhere, which is what
    dropout already does and is not what removing a unit means.
    """

    def __init__(self, rate: float, group_size: int = 1):
        super().__init__()
        self.rate = float(rate)
        self.group_size = int(group_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.rate == 0.0:
            return x
        if x.dim() != 3:
            raise ValueError(
                f"expected (batch, tokens, channels), got {tuple(x.shape)}; this "
                "operator is channel-structured and cannot infer groups otherwise"
            )
        batch, _, channels = x.shape
        if channels % self.group_size:
            raise ValueError(
                f"{channels} channels do not divide into groups of {self.group_size}"
            )
        groups = channels // self.group_size
        keep = torch.empty(batch, 1, groups, device=x.device, dtype=x.dtype).bernoulli_(
            1.0 - self.rate
        )
        mask = keep.repeat_interleave(self.group_size, dim=2)
        return x * mask * _keep_scale(self.rate)


class TokenMask(nn.Module):
    """Zero whole tokens, keeping every channel of the surviving ones.

    The transformer-native counterpart to dropping a spatial location. A patch
    trigger occupies specific tokens, so removing tokens wholesale probes a
    different failure mode than removing channels.

    Token 0 is the CLS token and is never masked: it is the classifier's only
    read point, so dropping it destroys the prediction outright rather than
    perturbing it, which would make PSU measure a broken forward pass.
    """

    def __init__(self, rate: float, protect_cls: bool = True):
        super().__init__()
        self.rate = float(rate)
        self.protect_cls = protect_cls

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.rate == 0.0:
            return x
        if x.dim() != 3:
            raise ValueError(
                f"expected (batch, tokens, channels), got {tuple(x.shape)}"
            )
        batch, tokens, _ = x.shape
        keep = torch.empty(batch, tokens, 1, device=x.device, dtype=x.dtype).bernoulli_(
            1.0 - self.rate
        )
        if self.protect_cls:
            keep[:, 0, :] = 1.0
        return x * keep * _keep_scale(self.rate)


class DropPath(nn.Module):
    """Zero a whole branch output for a whole sample, the stochastic-depth noise.

    Attached at a branch-output position (before_attention_residual or
    before_mlp_residual) this drops the entire attention or MLP contribution for
    that sample while the residual stream carries the input through unchanged.
    It is the residual-native perturbation: the block becomes the identity for
    that sample instead of computing a corrupted version of its function.
    """

    def __init__(self, rate: float):
        super().__init__()
        self.rate = float(rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.rate == 0.0:
            return x
        shape = (x.shape[0],) + (1,) * (x.dim() - 1)
        keep = torch.empty(shape, device=x.device, dtype=x.dtype).bernoulli_(
            1.0 - self.rate
        )
        return x * keep * _keep_scale(self.rate)


class GaussianNoise(nn.Module):
    """Additive noise scaled to the activation's own magnitude.

    A continuous perturbation rather than a removal, included as the control for
    whether PSBD needs structure removed at all or merely needs the activation
    disturbed. rate is read as a relative standard deviation, so the noise
    magnitude tracks the layer's own scale and one rate means the same relative
    disturbance everywhere.

    No inverted scaling, because additive zero-mean noise already leaves the
    expected activation unchanged.
    """

    def __init__(self, rate: float):
        super().__init__()
        self.rate = float(rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.rate == 0.0:
            return x
        scale = x.detach().std()
        return x + torch.randn_like(x) * (self.rate * scale)


class HeadMask(nn.Module):
    """Zero whole attention heads, on the per-head tensor (batch, heads, tokens, dim).

    Separate from GroupChannelMask because a head is only addressable before
    out_proj mixes the heads together, and that tensor is never exposed at a
    module boundary: nn.MultiheadAttention runs F.multi_head_attention_forward,
    which reads out_proj.weight directly and never calls out_proj as a module, so
    no forward hook on it ever fires. The head axis is reachable only by
    recomputing attention, which is what masked_attention_forward does.
    """

    def __init__(self, rate: float):
        super().__init__()
        self.rate = float(rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.rate == 0.0:
            return x
        if x.dim() != 4:
            raise ValueError(
                f"expected (batch, heads, tokens, dim), got {tuple(x.shape)}"
            )
        batch, heads = x.shape[0], x.shape[1]
        keep = torch.empty(
            batch, heads, 1, 1, device=x.device, dtype=x.dtype
        ).bernoulli_(1.0 - self.rate)
        return x * keep * _keep_scale(self.rate)


def masked_attention_forward(attention, mask, query, key, value, **kwargs):
    """nn.MultiheadAttention.forward, reimplemented so the head axis is maskable.

    Matches torchvision's call site, self_attention(x, x, x, need_weights=False),
    which is self-attention with no mask and batch_first=True. Weights are read
    off the loaded module rather than copied, so nothing is mutated and removing
    the wrapper restores the original behaviour exactly.

    Deliberately does NOT support cross-attention or key padding: this project
    only ever calls it as self-attention, and silently accepting a key that
    differs from the query would compute something plausible and wrong.
    """
    if query is not key or query is not value:
        raise ValueError("head masking supports self-attention only")

    batch, tokens, channels = query.shape
    heads = attention.num_heads
    dim = channels // heads

    projected = nn.functional.linear(
        query, attention.in_proj_weight, attention.in_proj_bias
    )
    q, k, v = projected.chunk(3, dim=-1)

    def split_heads(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.view(batch, tokens, heads, dim).transpose(1, 2)

    attended = nn.functional.scaled_dot_product_attention(
        split_heads(q), split_heads(k), split_heads(v)
    )

    attended = mask(attended)

    merged = attended.transpose(1, 2).reshape(batch, tokens, channels)
    # need_weights=False at the call site, so the second element is never read.
    return attention.out_proj(merged), None


class FixedHeadMask(nn.Module):
    """Zero one named attention head, deterministically, for every sample.

    The ablation counterpart to HeadMask's random sampling. Because the mask is
    fixed rather than drawn, a single forward pass measures the effect exactly and
    no Monte Carlo averaging is needed, which is what makes a full 144-head sweep
    affordable.

    No inverted scaling is applied. The point of a leave-one-out ablation is to
    measure what the model loses when a specific head is gone, and rescaling the
    survivors would compensate for exactly that loss.
    """

    def __init__(self, head_index: int):
        super().__init__()
        self.head_index = int(head_index)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(
                f"expected (batch, heads, tokens, dim), got {tuple(x.shape)}"
            )
        if not 0 <= self.head_index < x.shape[1]:
            raise IndexError(
                f"head {self.head_index} outside 0..{x.shape[1] - 1} for this block"
            )
        # Cloned because the caller's tensor is still live inside the attention
        # forward; masking in place would corrupt it for any later reader.
        out = x.clone()
        out[:, self.head_index] = 0.0
        return out


def head_mask(rate: float) -> HeadMask:
    """Whole attention heads. Requires the attention forward wrapper."""
    return HeadMask(rate)


def fixed_head_mask(head_index: int):
    """Factory matching plug_dropout's (rate) -> Module contract.

    plug_dropout calls factory(rate); here the rate slot is unused because the
    head to remove is fixed in advance, so the returned closure ignores it.
    """

    def build(_rate: float) -> FixedHeadMask:
        return FixedHeadMask(head_index)

    return build


def channel_mask(rate: float) -> GroupChannelMask:
    """Whole embedding channels, shared across tokens."""
    return GroupChannelMask(rate, group_size=1)


# Every entry takes a rate and returns a module perturbing (batch, tokens,
# channels), so any of these is a drop-in for nn.Dropout in plug_dropout's
# factory argument.
PERTURBATIONS: dict[str, type[nn.Module]] = {
    "dropout": nn.Dropout,
    "head_mask": head_mask,
    "channel_mask": channel_mask,
    "token_mask": TokenMask,
    "droppath": DropPath,
    "gaussian": GaussianNoise,
}

# Which perturbations are meaningful at which positions. head_mask is the
# constrained one: it is only a head mask on the concatenated per-head outputs,
# and anywhere else its 64-wide groups are arbitrary channel blocks.
PERTURBATION_POSITIONS: dict[str, tuple[str, ...]] = {
    "head_mask": ("attention_heads",),
    "droppath": ("before_attention_residual", "before_mlp_residual"),
}


def build_perturbation(name: str):
    """Look up an operator by name, failing loudly on a typo.

    A silent fallback to nn.Dropout here would produce a complete, plausible
    sweep that answers a different question than the one asked.
    """
    if name not in PERTURBATIONS:
        raise KeyError(f"unknown perturbation {name!r}; known: {sorted(PERTURBATIONS)}")
    return PERTURBATIONS[name]
