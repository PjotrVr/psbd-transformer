"""Perturbation operators: what a probe does once it is attached.

models.positions owns where a probe attaches. Every operator here shares
nn.Dropout's interface, a Module built from a rate that maps a tensor to a tensor
of the same shape, so any operator plugs into any compatible position without
either side knowing about the other.

Dropout samples an independent mask per (token, channel), so it never removes an
attention head, a hidden neuron or a token whole. It removes a little of
everything. A backdoor shortcut is carried by specific structures, and removing a
structure at a time measures something different from dissolving all of them
uniformly, which is what the structured operators here exist to test.

Every operator keeps nn.Dropout's 2 conventions, because PSU compares a perturbed
pass against an unperturbed pass and any deviation would read as signal. The
first is identity when self.training is False or rate is 0. The second is
inverted scaling: survivors are divided by the keep probability so the expected
activation is unchanged.

Shapes are (batch, tokens, channels), which ViT carries with batch_first=True.
Swin carries (batch, height, width, channels), so the token-axis operators
flatten the 2 spatial axes and restore them afterwards.

Adding an operator is 3 steps: write the Module after whichever existing operator
masks the same axis, add it to PERTURBATIONS and list its position restriction in
PERTURBATION_POSITIONS if it only makes sense somewhere specific.
"""

import torch
import torch.nn as nn


def _keep_scale(rate: float) -> float:
    """Inverted-dropout scale, the 1 / (1 - p) factor applied to survivors.

        original form
            y = (m * x) / (1 - p),   m ~ Bernoulli(1 - p)

        descriptive form
            survivor = kept_value / keep_probability

    so the expected output equals the input and the perturbed pass stays on the
    same scale as the unperturbed pass. Guarded because rate 1.0 would divide by 0.
    """
    keep_probability = 1.0 - rate
    if keep_probability <= 0.0:
        raise ValueError(f"rate {rate} removes everything, leaving nothing to scale")
    scale = 1.0 / keep_probability
    return scale


def _to_token_layout(x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
    """Any supported layout viewed as (batch, tokens, channels), plus its original shape.

    ViT already carries the token layout and passes straight through. Swin
    features are (batch, height, width, channels), so the 2 spatial axes are
    flattened into a single token axis. The returned shape lets the caller restore
    the layout after masking.
    """
    if x.dim() == 3:
        return x, tuple(x.shape)
    if x.dim() == 4:
        batch, height, width, channels = x.shape
        tokens = x.reshape(batch, height * width, channels)  # (batch, h * w, channels)
        return tokens, tuple(x.shape)
    raise ValueError(
        f"expected (batch, tokens, channels) or (batch, H, W, channels), "
        f"got {tuple(x.shape)}"
    )


class GroupChannelMask(nn.Module):
    """Zero whole groups of channels, shared across every token of a sample.

    group_size 1 masks individual embedding channels, which at the mlp_neurons
    position means whole hidden neurons of the MLP.

    Sharing the mask across tokens is the substantive difference from nn.Dropout.
    A neuron either contributes to the sample or it does not. Dropping it
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

        tokens_view, original_shape = _to_token_layout(x)
        masked = self._mask_channels(tokens_view)
        out = masked.reshape(original_shape)
        return out

    def _mask_channels(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, channels = x.shape  # (batch, tokens, channels)
        if channels % self.group_size:
            raise ValueError(
                f"{channels} channels do not divide into groups of {self.group_size}"
            )

        groups = channels // self.group_size
        keep = torch.empty(batch, 1, groups, device=x.device, dtype=x.dtype).bernoulli_(
            1.0 - self.rate
        )
        mask = keep.repeat_interleave(self.group_size, dim=2)  # (batch, 1, channels)

        masked = x * mask * _keep_scale(self.rate)
        return masked


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

        # A Swin feature map is all spatial positions and carries no CLS token,
        # so row 0 of its flattened token axis is an ordinary patch.
        protect_cls = self.protect_cls and x.dim() == 3

        tokens_view, original_shape = _to_token_layout(x)
        masked = self._mask_tokens(tokens_view, protect_cls)
        out = masked.reshape(original_shape)
        return out

    def _mask_tokens(self, x: torch.Tensor, protect_cls: bool) -> torch.Tensor:
        batch, tokens, _ = x.shape  # (batch, tokens, channels)
        keep = torch.empty(batch, tokens, 1, device=x.device, dtype=x.dtype).bernoulli_(
            1.0 - self.rate
        )
        keep *= _keep_scale(self.rate)
        if protect_cls:
            keep[:, 0, :] = 1.0

        masked = x * keep
        return masked


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

        # A single Bernoulli draw per sample, broadcast over every other axis, so
        # the layout does not matter and both ViT and Swin are handled unchanged.
        sample_shape = (x.shape[0],) + (1,) * (x.dim() - 1)
        keep = torch.empty(sample_shape, device=x.device, dtype=x.dtype).bernoulli_(
            1.0 - self.rate
        )

        masked = x * keep * _keep_scale(self.rate)
        return masked


class GaussianNoise(nn.Module):
    """Additive noise scaled to the activation's own magnitude.

    A continuous perturbation rather than a removal, included as the control for
    whether PSBD needs structure removed at all or only needs the activation
    disturbed. rate is a relative standard deviation, so the noise magnitude
    tracks the layer's own scale and a rate means the same relative disturbance
    everywhere.

    The standard deviation is computed per sample, over every axis except the
    batch. A batch-wide reduction would make a sample's noise level depend on which
    other samples shared its batch. The validation, clean and backdoor splits hold
    different image populations, so the 3 would sit at 3 noise levels while the
    comparison between them assumes a single level. This is the only operator
    where that coupling was possible.

    No inverted scaling, because additive zero-mean noise already leaves the
    expected activation unchanged.
    """

    def __init__(self, rate: float):
        super().__init__()
        self.rate = float(rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.rate == 0.0:
            return x

        non_batch_axes = tuple(range(1, x.dim()))
        scale = x.detach().std(dim=non_batch_axes, keepdim=True)  # (batch, 1, ..., 1)

        noised = x + torch.randn_like(x) * (self.rate * scale)
        return noised


class RademacherNoise(nn.Module):
    """Additive symmetric Bernoulli noise, the minimum variance trace probe.

    Identical to GaussianNoise except that each entry is drawn from {-1, +1}
    rather than from a normal, at the same per sample scale.

    The reason it exists is a prediction rather than a hunch. The second order
    expansion in docs/theory-perturbation-consistency.md gives

        psu(x) is approximately -0.5 * trace(H * Sigma)

    and for an isotropic Sigma = variance * I that is -0.5 * variance *
    trace(H). Estimating a trace as the expectation of z^T H z over probes z with
    zero mean and identity covariance is Hutchinson's estimator, so PSBD run with
    isotropic noise IS a Hutchinson trace estimator of the Hessian of the
    predicted class probability.

    Hutchinson's variance is known for both probe distributions:

        gaussian     2 * frobenius_norm(H)^2
        rademacher   2 * (frobenius_norm(H)^2 - sum of squared diagonal entries)

    The Rademacher form is smaller by exactly the diagonal energy, and it is the
    minimum variance choice among all probe distributions with identity
    covariance. Both estimate the same expectation, so at a matched scale the 2
    operators should agree in the limit of many passes and Rademacher should
    separate better at the small pass counts actually used.

    That is a falsifiable prediction, and it is the point of including this. If
    Rademacher does not beat Gaussian at matched shift ratio, the trace estimator
    reading of the method is wrong.

    No inverted scaling, for the same reason as GaussianNoise: the noise is zero
    mean, so the expected activation is already unchanged.
    """

    def __init__(self, rate: float):
        super().__init__()
        self.rate = float(rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.rate == 0.0:
            return x

        non_batch_axes = tuple(range(1, x.dim()))
        scale = x.detach().std(dim=non_batch_axes, keepdim=True)  # (batch, 1, ..., 1)

        # Drawn as 0 or 1 then mapped to -1 or +1, which is exact in every dtype
        # and avoids the sign of a value that could be 0.
        signs = torch.randint(0, 2, x.shape, device=x.device, dtype=x.dtype) * 2 - 1

        noised = x + signs * (self.rate * scale)
        return noised


class HeadMask(nn.Module):
    """Zero whole attention heads, on the per-head (batch, heads, tokens, dim) tensor.

    Separate from GroupChannelMask because a head is only addressable before
    out_proj mixes the heads together, and that tensor is never exposed at a
    module boundary: nn.MultiheadAttention runs F.multi_head_attention_forward,
    which reads out_proj.weight directly and never calls out_proj as a module, so
    no forward hook on it ever fires. The head axis is reachable only by
    recomputing attention, which is what masked_attention_forward does, so this
    operator only works at the attention_heads position.
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

        masked = x * keep * _keep_scale(self.rate)
        return masked


class FixedHeadMask(nn.Module):
    """Zero a single named attention head, deterministically, for every sample.

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
        # forward, and masking in place would corrupt it for any later reader.
        out = x.clone()
        out[:, self.head_index] = 0.0
        return out


class GainScale(nn.Module):
    """Amplify a normalization layer's output, the IBD-PSC perturbation.

    IBD-PSC (Hou et al., ICML 2024) detects backdoors by scaling BatchNorm's
    affine parameters and measuring whether the prediction survives. ViT has no
    BatchNorm, so a direct port is impossible. The LayerNorm analogue is exact
    rather than approximate:

        original form
            y = gamma * x_hat + beta,  scale both by omega

        descriptive form
            omega*gamma * x_hat + omega*beta = omega * (gamma * x_hat + beta) = omega * y

    so amplifying both affine parameters is identical to multiplying the layer's
    output, which a post-hook can do without touching a single weight.

    rate is read as omega - 1, so rate 0 is the identity like every other
    operator here and the rate grids stay comparable in shape.

    This is a parameter-space perturbation expressed in activation space, and the
    only operator in the study that amplifies rather than removes.
    """

    def __init__(self, rate: float):
        super().__init__()
        self.rate = float(rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.rate == 0.0:
            return x

        amplified = x * (1.0 + self.rate)
        return amplified


class ScaleUp(nn.Module):
    """Amplify pixel values, the SCALE-UP perturbation, in normalized space.

    SCALE-UP (Guo et al., ICLR 2023) multiplies pixel values by an integer factor
    and asks whether the prediction stays the same. A backdoored input keeps its
    target prediction because the trigger survives amplification.

        original form
            x' = clip(n * x),  n in {2, 3, ...},  x in [0, 1]

        descriptive form
            pixels = clip(denormalize(x))
            amplified = clip(factor * pixels)
            x' = normalize(amplified)

    The loaders deliver normalized tensors, so the operator has to undo the
    normalization, amplify, clip to the valid pixel range and renormalize. Scaling
    the normalized tensor directly would amplify the dataset mean as though it were
    signal. The clip, which is where SCALE-UP's nonlinearity lives, would also land
    in the wrong place.

    rate is read as factor - 1, so rate 0 is the identity, matching every other
    operator.
    """

    def __init__(self, rate: float, mean, std):
        super().__init__()
        self.rate = float(rate)
        self.register_buffer("mean", torch.tensor(mean).view(1, -1, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, -1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.rate == 0.0:
            return x

        mean = self.mean.to(x.device, x.dtype)  # (1, channels, 1, 1)
        std = self.std.to(x.device, x.dtype)  # (1, channels, 1, 1)

        pixels = (x * std + mean).clamp(0.0, 1.0)
        amplified = (pixels * (1.0 + self.rate)).clamp(0.0, 1.0)

        renormalized = (amplified - mean) / std
        return renormalized


def masked_attention_forward(attention, mask, query, key, value, **kwargs):
    """nn.MultiheadAttention.forward, reimplemented so the head axis is maskable.

    Matches torchvision's call site, self_attention(x, x, x, need_weights=False),
    which is self-attention with no mask and batch_first=True. Weights are read
    off the loaded module rather than copied, so nothing is mutated and removing
    the wrapper restores the original behaviour exactly.

    Does not support cross-attention or key padding. This project only ever calls
    it as self-attention, and silently accepting a key that differs from the query
    would compute something plausible and wrong.
    """
    if query is not key or query is not value:
        raise ValueError("head masking supports self-attention only")

    batch, tokens, channels = query.shape
    heads = attention.num_heads
    dim = channels // heads

    projected = nn.functional.linear(
        query, attention.in_proj_weight, attention.in_proj_bias
    )
    q, k, v = projected.chunk(3, dim=-1)  # each (batch, tokens, channels)

    def split_heads(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.view(batch, tokens, heads, dim).transpose(1, 2)

    # (batch, heads, tokens, dim), the one layout where a whole head is a slice.
    attended = nn.functional.scaled_dot_product_attention(
        split_heads(q), split_heads(k), split_heads(v)
    )

    attended = mask(attended)

    merged = attended.transpose(1, 2).reshape(batch, tokens, channels)
    # need_weights=False at the call site, so the second element is never read.
    return attention.out_proj(merged), None


def head_mask(rate: float) -> HeadMask:
    """Whole attention heads. Requires the attention forward wrapper."""
    return HeadMask(rate)


def channel_mask(rate: float) -> GroupChannelMask:
    """Whole embedding channels, shared across tokens."""
    return GroupChannelMask(rate, group_size=1)


def fixed_head_mask(head_index: int):
    """Factory matching plug_dropout's factory contract, a callable taking a rate.

    plug_dropout calls factory(rate). Here the rate slot is unused because the
    head to remove is fixed in advance, so the returned closure ignores it.
    """

    def build(_rate: float) -> FixedHeadMask:
        return FixedHeadMask(head_index)

    return build


def scale_up(mean, std):
    """Factory matching plug_dropout's factory contract, a callable taking a rate.

    The normalization constants are dataset-specific and are not available at
    the registry level, so they are bound here by the caller.
    """

    def build(rate: float) -> ScaleUp:
        return ScaleUp(rate, mean, std)

    return build


# Every entry takes a rate and returns a module perturbing (batch, tokens,
# channels), so any of these is a drop-in for nn.Dropout in plug_dropout's
# factory argument.
OPERATORS: dict[str, type[nn.Module]] = {
    "dropout": nn.Dropout,
    "head_mask": head_mask,
    "channel_mask": channel_mask,
    "token_mask": TokenMask,
    "droppath": DropPath,
    "gaussian": GaussianNoise,
    # Same isotropic covariance as gaussian, lower estimator variance. See the
    # class docstring: this is a prediction of the trace estimator reading, not a
    # variation for its own sake.
    "rademacher": RademacherNoise,
    # Ports of 2 published perturbation-consistency detectors, so all 3
    # perturbation families (input, activation, parameter) sit in a single registry.
    # scale_up is absent because it needs the dataset's normalization constants
    # and so cannot be built from a rate alone. See scale_up().
    "gain_scale": GainScale,
}

# Operators whose output is a deterministic function of their input, so every
# Monte Carlo pass returns the same value. PSU is an expectation over k passes, so
# for these it is exact at k = 1 and a k > 1 sweep writes k identical rows. Their
# shift ratio is also a per-sample flip indicator rather than a fraction of
# passes, which is worth stating wherever sigma is compared across operators.
DETERMINISTIC_OPERATORS: frozenset[str] = frozenset({"gain_scale", "scale_up"})

# Which perturbations are meaningful at which positions. head_mask is the
# constrained case: it is only a head mask on the concatenated per-head outputs.
# Anywhere else its 64-wide groups are arbitrary channel blocks.
OPERATOR_POSITIONS: dict[str, tuple[str, ...]] = {
    "head_mask": ("attention_heads",),
    "droppath": ("before_attention_residual", "before_mlp_residual"),
}

# Positions no structured operator may attach to, with the reason. Both entries
# are cases where the operator runs happily and produces a complete, plausible,
# wrong answer, which is worse than a crash.
STRUCTURED_OPERATORS: frozenset[str] = frozenset(
    {"token_mask", "channel_mask", "head_mask", "droppath"}
)

FORBIDDEN_OPERATOR_POSITIONS: dict[tuple[str, str], str] = {
    (operator, "input_pixels"): (
        "input_pixels carries a (batch, channels, height, width) image, and the "
        "token operators read a rank-4 tensor as Swin's (batch, height, width, "
        "channels). token_mask would mask image rows independently per colour "
        "channel, channel_mask would mask image columns, and neither is the axis "
        "the operator names. Use an elementwise operator here."
    )
    for operator in STRUCTURED_OPERATORS
}

# ViT's head reads token 0 alone and TokenMask never masks token 0, so a token
# mask applied after the final norm changes nothing the classifier ever sees.
# PSU would be identically 0 and AUROC exactly 0.5, reading as "this position has
# no effect" rather than as "this probe never fired".
FORBIDDEN_OPERATOR_POSITIONS[("token_mask", "final_norm_out")] = (
    "final_norm_out sits after the final LayerNorm, and ViT's head reads only "
    "token 0, which token_mask protects by design. The probe would be inert and "
    "its 0 result would be indistinguishable from a position that does nothing."
)


def check_operator_position(operator: str, position: str) -> None:
    """Refuse operator and position pairs that would produce a plausible wrong answer.

    Raises rather than warning. Every combination rejected here runs without
    error and writes a complete sweep, so a warning would be read past and the
    result would enter a table looking like every other cell.
    """
    forbidden = FORBIDDEN_OPERATOR_POSITIONS.get((operator, position))
    if forbidden is not None:
        raise ValueError(f"{operator} at {position}: {forbidden}")

    allowed = OPERATOR_POSITIONS.get(operator)
    if allowed is not None and position not in allowed:
        raise ValueError(
            f"{operator} is only meaningful at {allowed}, not at {position!r}"
        )


def effective_forward_passes(operator: str, forward_passes: int) -> int:
    """Passes actually needed: 1 for a deterministic operator, k otherwise."""
    needed = 1 if operator in DETERMINISTIC_OPERATORS else forward_passes
    return needed


def build_operator(name: str):
    """The operator registered under name, failing loudly on a typo.

    A silent fallback to nn.Dropout here would produce a complete, plausible sweep
    that answers a different question than the one asked.
    """
    if name not in OPERATORS:
        raise KeyError(f"unknown perturbation {name!r}, known: {sorted(OPERATORS)}")
    return OPERATORS[name]
