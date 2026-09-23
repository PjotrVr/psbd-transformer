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

import math

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
        token_shape = tuple(x.shape)
        return x, token_shape
    if x.dim() == 4:
        batch, height, width, channels = x.shape
        tokens = x.reshape(batch, height * width, channels)  # (batch, tokens, channels)
        spatial_shape = tuple(x.shape)
        return tokens, spatial_shape
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
        """x with channel groups zeroed, returned in the layout it arrived in.

        x is (batch, tokens, channels) from ViT or (batch, height, width,
        channels) from Swin.
        """
        if not self.training or self.rate == 0.0:
            return x

        tokens_view, original_shape = _to_token_layout(x)  # (batch, tokens, channels)
        masked = self._mask_channels(tokens_view)  # (batch, tokens, channels)
        out = masked.reshape(original_shape)
        return out

    def _mask_channels(self, x: torch.Tensor) -> torch.Tensor:
        """x with whole channel groups zeroed, (batch, tokens, channels) in and out.

        Raises when the channels do not divide into groups of group_size.
        """
        batch, _, channels = x.shape  # (batch, tokens, channels)
        if channels % self.group_size:
            raise ValueError(
                f"{channels} channels do not divide into groups of {self.group_size}"
            )

        groups = channels // self.group_size
        keep = torch.empty(batch, 1, groups, device=x.device, dtype=x.dtype).bernoulli_(
            1.0 - self.rate
        )  # (batch, 1, groups)
        mask = keep.repeat_interleave(self.group_size, dim=2)  # (batch, 1, channels)

        masked = x * mask * _keep_scale(self.rate)  # (batch, tokens, channels)
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
        """x with whole tokens zeroed, returned in the layout it arrived in.

        x is (batch, tokens, channels) from ViT or (batch, height, width,
        channels) from Swin.
        """
        if not self.training or self.rate == 0.0:
            return x

        # A Swin feature map is all spatial positions and carries no CLS token,
        # so row 0 of its flattened token axis is an ordinary patch.
        protect_cls = self.protect_cls and x.dim() == 3

        tokens_view, original_shape = _to_token_layout(x)  # (batch, tokens, channels)
        masked = self._mask_tokens(
            tokens_view, protect_cls
        )  # (batch, tokens, channels)
        out = masked.reshape(original_shape)
        return out

    def _mask_tokens(self, x: torch.Tensor, protect_cls: bool) -> torch.Tensor:
        """x with whole tokens zeroed, (batch, tokens, channels) in and out."""
        batch, tokens, _ = x.shape  # (batch, tokens, channels)
        keep = torch.empty(batch, tokens, 1, device=x.device, dtype=x.dtype).bernoulli_(
            1.0 - self.rate
        )  # (batch, tokens, 1)
        keep *= _keep_scale(self.rate)
        if protect_cls:
            keep[:, 0, :] = 1.0

        masked = x * keep  # (batch, tokens, channels)
        return masked


class TokenSubstitute(nn.Module):
    """Replace whole tokens with another token of the same sample.

    TokenMask zeroes a token, and a zero token is off the data manifold: no
    training image ever produced one, so part of the prediction shift it causes is
    the network reacting to an input it has no calibration for rather than to the
    loss of that token's content. Substitution removes the content and leaves the
    input in distribution, which isolates the effect being measured.

    The replacement is another token of the SAME sample, taken by rolling the
    token axis, so every substituted token is a real activation this layer
    produced for this image.

    Borrowing from another sample in the batch would be the more obvious
    construction and it is wrong here. PSU is a per-sample score, and a score
    that depends on which other images share the batch is not a property of the
    input. This project has already retired 1 arm for that fault, the
    batch-coupled Gaussian whose tensors sit in archive/gaussian_batchstd and
    which cli/compare.py still classifies as batch_coupled_superseded. Rolling
    within the sample keeps the substitution on the manifold and keeps the score
    a function of the input alone.

    No inverted scaling is applied, unlike every masking operator here. Scaling
    exists to hold the expectation equal to the input when part of it is zeroed.
    A substituted token is already a sample from the right distribution, so
    dividing by the keep probability would inflate the activations instead of
    correcting them.

    Token 0 is the CLS token and is never substituted, for the reason TokenMask
    never masks it: it is the classifier's only read point.
    """

    def __init__(self, rate: float, protect_cls: bool = True):
        super().__init__()
        self.rate = float(rate)
        self.protect_cls = protect_cls

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x with whole tokens swapped for another sample's, same layout out.

        x is (batch, tokens, channels) from ViT or (batch, height, width,
        channels) from Swin.
        """
        if not self.training or self.rate == 0.0:
            return x

        protect_cls = self.protect_cls and x.dim() == 3
        tokens_view, original_shape = _to_token_layout(x)  # (batch, tokens, channels)
        if tokens_view.shape[1] - int(protect_cls) < 2:
            # Fewer than 2 patch tokens leaves nothing to substitute from, so the
            # pass is left unperturbed rather than degenerating into a no-op mask.
            return x

        swapped = self._substitute(tokens_view, protect_cls)
        out = swapped.reshape(original_shape)
        return out

    def _substitute(self, x: torch.Tensor, protect_cls: bool) -> torch.Tensor:
        """x with chosen tokens taken from elsewhere in the same sample.

        (batch, tokens, channels) in and out.
        """
        batch, tokens, _ = x.shape  # (batch, tokens, channels)
        offset = 1 if protect_cls else 0
        patches = x[:, offset:, :]  # (batch, patches, channels)

        # A roll by a random non-zero amount, per sample, so a token is replaced
        # by a different token of the same image rather than by itself.
        shifts = torch.randint(
            1, patches.shape[1], (batch,), device=x.device
        )  # (batch,)
        index = (
            torch.arange(patches.shape[1], device=x.device).view(1, -1)
            - shifts.view(-1, 1)
        ) % patches.shape[1]  # (batch, patches)
        donor = torch.gather(
            patches, 1, index.unsqueeze(-1).expand_as(patches)
        )  # (batch, patches, channels)

        replace = torch.empty(
            batch, patches.shape[1], 1, device=x.device, dtype=x.dtype
        ).bernoulli_(self.rate)  # (batch, patches, 1)

        substituted = x.clone()
        substituted[:, offset:, :] = patches * (1.0 - replace) + donor * replace
        return substituted


class TokenBlockMask(nn.Module):
    """Zero a contiguous patch-aligned rectangle of tokens rather than a random subset.

    TokenMask drops tokens independently, so it removes a share of a local
    trigger's tokens in proportion to the rate and degrades the shortcut
    gradually. A contiguous mask either covers the trigger or misses it, which
    turns that gradual degradation into a high-variance one, and the statistic
    already averages over k passes.

    The rectangle's area is the requested rate of the token grid, its aspect is
    square where the grid allows, and its position is uniform over the grid. For
    a global trigger this should do no better than dropping the same number of
    tokens at random, which is the prediction that makes it a useful control.

    Token 0 is the CLS token and sits outside the grid, so it is never covered.
    """

    def __init__(self, rate: float, protect_cls: bool = True):
        super().__init__()
        self.rate = float(rate)
        self.protect_cls = protect_cls

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x with a contiguous block of tokens zeroed, same layout out.

        x is (batch, tokens, channels) from ViT or (batch, height, width,
        channels) from Swin.
        """
        if not self.training or self.rate == 0.0:
            return x

        protect_cls = self.protect_cls and x.dim() == 3
        tokens_view, original_shape = _to_token_layout(x)  # (batch, tokens, channels)
        grid = self._grid_side(tokens_view.shape[1], protect_cls)
        if grid is None:
            # A token count that is not a square grid has no rectangle to speak
            # of, so fall back to independent dropping rather than inventing a
            # geometry the layout does not have.
            return TokenMask(self.rate, self.protect_cls).train(self.training)(x)

        masked = self._mask_block(tokens_view, grid, protect_cls)
        out = masked.reshape(original_shape)
        return out

    def _grid_side(self, tokens: int, protect_cls: bool) -> int | None:
        """The side of the square patch grid, or None when there is not one."""
        patches = tokens - 1 if protect_cls else tokens
        side = int(math.isqrt(patches))
        if side * side != patches:
            return None
        return side

    def _mask_block(
        self, x: torch.Tensor, grid: int, protect_cls: bool
    ) -> torch.Tensor:
        """x with 1 rectangle per sample zeroed, (batch, tokens, channels) in and out."""
        batch, tokens, _ = x.shape  # (batch, tokens, channels)
        offset = 1 if protect_cls else 0

        # A square block whose area is the requested share of the grid, clamped so
        # it always removes at least 1 token and never the whole grid.
        side = max(1, min(grid, round(grid * math.sqrt(self.rate))))
        keep = torch.ones(batch, tokens, 1, device=x.device, dtype=x.dtype)
        rows = torch.randint(0, grid - side + 1, (batch,), device=x.device)
        cols = torch.randint(0, grid - side + 1, (batch,), device=x.device)
        for sample in range(batch):
            row, col = int(rows[sample]), int(cols[sample])
            for step in range(side):
                start = offset + (row + step) * grid + col
                keep[sample, start : start + side, :] = 0.0

        # The surviving tokens carry the scale of the tokens that were removed, so
        # the perturbed pass stays comparable to the unperturbed one, exactly as
        # TokenMask does.
        removed = side * side
        kept_share = 1.0 - removed / float(grid * grid)
        if kept_share > 0.0:
            keep = keep / kept_share
            if protect_cls:
                keep[:, 0, :] = 1.0

        masked = x * keep  # (batch, tokens, channels)
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
        """x with whole samples zeroed, any layout with batch first, same shape out."""
        if not self.training or self.rate == 0.0:
            return x

        # A single Bernoulli draw per sample, broadcast over every other axis, so
        # the layout does not matter and both ViT and Swin are handled unchanged.
        sample_shape = (x.shape[0],) + (1,) * (x.dim() - 1)
        keep = torch.empty(sample_shape, device=x.device, dtype=x.dtype).bernoulli_(
            1.0 - self.rate
        )  # (batch, 1, ..., 1)

        masked = x * keep * _keep_scale(self.rate)  # same shape as x
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

    No inverted scaling, because additive noise with mean 0 already leaves the
    expected activation unchanged.
    """

    def __init__(self, rate: float):
        super().__init__()
        self.rate = float(rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x plus noise at rate times its own spread, any layout, same shape out."""
        if not self.training or self.rate == 0.0:
            return x

        non_batch_axes = tuple(range(1, x.dim()))
        scale = x.detach().std(dim=non_batch_axes, keepdim=True)  # (batch, 1, ..., 1)

        noised = x + torch.randn_like(x) * (self.rate * scale)  # same shape as x
        return noised


class RademacherNoise(nn.Module):
    """Additive symmetric Bernoulli noise, the minimum variance trace probe.

    Identical to GaussianNoise except that each entry is drawn from {-1, +1}
    rather than from a normal, at the same per sample scale.

    The reason it exists is a prediction rather than a hunch. The second order
    expansion in docs/theory-perturbation-consistency.md, with Hutchinson's
    variance for each probe distribution, gives

        original form
            phi(x) ~ -(1/2) tr(H Sigma)
            tr(H Sigma) = sigma^2 tr(H)                  for Sigma = sigma^2 I
            Var_gaussian   = 2 ||H||_F^2
            Var_rademacher = 2 (||H||_F^2 - sum_j H_jj^2)

        symbols
            phi(x)     PSU of sample x
            H          Hessian of the predicted class probability with respect
                       to the perturbed activation
            Sigma      covariance of the perturbation
            sigma^2    variance of an isotropic perturbation
            z          a probe vector with mean 0 and identity covariance
            ||H||_F    Frobenius norm of H
            H_jj       the j-th diagonal entry of H

        descriptive form
            psu(x) is approximately -0.5 * trace(H * Sigma)
            gaussian     2 * frobenius_norm(H)^2
            rademacher   2 * (frobenius_norm(H)^2 - sum of squared diagonal entries)

    Estimating a trace as the expectation of z^T H z over such probes z is
    Hutchinson's estimator, so PSBD run with isotropic noise IS a Hutchinson
    trace estimator of the Hessian of the predicted class probability. The
    Rademacher form is smaller by exactly the diagonal energy, and it is the
    minimum variance choice among all probe distributions with identity
    covariance. Both estimate the same expectation, so at a matched scale the 2
    operators should agree in the limit of many passes and Rademacher should
    separate better at the small pass counts actually used.

    That is a falsifiable prediction, and it is the point of including this. If
    Rademacher does not beat Gaussian at matched shift ratio, the trace estimator
    reading of the method is wrong.

    No inverted scaling, for the same reason as GaussianNoise: the noise has mean
    0, so the expected activation is already unchanged.
    """

    def __init__(self, rate: float):
        super().__init__()
        self.rate = float(rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x plus sign noise at rate times its own spread, any layout, same shape out."""
        if not self.training or self.rate == 0.0:
            return x

        non_batch_axes = tuple(range(1, x.dim()))
        scale = x.detach().std(dim=non_batch_axes, keepdim=True)  # (batch, 1, ..., 1)

        # Drawn as 0 or 1 then mapped to -1 or +1, which is exact in every dtype
        # and avoids the sign of a value that could be 0.
        signs = (
            torch.randint(0, 2, x.shape, device=x.device, dtype=x.dtype) * 2 - 1
        )  # same shape as x

        noised = x + signs * (self.rate * scale)  # same shape as x
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
        """x with whole heads zeroed, (batch, heads, tokens, dim) in and out.

        Raises on any other rank, since only the attention wrapper produces the
        per-head layout.
        """
        if not self.training or self.rate == 0.0:
            return x
        if x.dim() != 4:
            raise ValueError(
                f"expected (batch, heads, tokens, dim), got {tuple(x.shape)}"
            )

        batch, heads = x.shape[0], x.shape[1]
        keep = torch.empty(
            batch, heads, 1, 1, device=x.device, dtype=x.dtype
        ).bernoulli_(1.0 - self.rate)  # (batch, heads, 1, 1)

        masked = x * keep * _keep_scale(self.rate)  # (batch, heads, tokens, dim)
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
        """x with 1 named head zeroed, (batch, heads, tokens, dim) in and out.

        Raises on any other rank or on a head index outside the block's heads.
        """
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
        out = x.clone()  # (batch, heads, tokens, dim)
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
        """x times 1 + rate, any layout, same shape out."""
        if not self.training or self.rate == 0.0:
            return x

        amplified = x * (1.0 + self.rate)  # same shape as x
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
        self.register_buffer(
            "mean", torch.tensor(mean).view(1, -1, 1, 1)
        )  # (1, channels, 1, 1)
        self.register_buffer(
            "std", torch.tensor(std).view(1, -1, 1, 1)
        )  # (1, channels, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """The amplified image, (batch, channels, height, width) in and out."""
        if not self.training or self.rate == 0.0:
            return x

        mean = self.mean.to(x.device, x.dtype)  # (1, channels, 1, 1)
        std = self.std.to(x.device, x.dtype)  # (1, channels, 1, 1)

        pixels = (x * std + mean).clamp(0.0, 1.0)  # (batch, channels, height, width)
        amplified = (pixels * (1.0 + self.rate)).clamp(0.0, 1.0)  # same shape

        renormalized = (amplified - mean) / std  # (batch, channels, height, width)
        return renormalized


def masked_attention_forward(attention, mask, query, key, value, **kwargs):
    """nn.MultiheadAttention.forward, reimplemented so the head axis is maskable.

    query, key and value are the same (batch, tokens, channels) tensor. The
    return is (output, None) with output (batch, tokens, channels), the pair the
    module's own forward returns under need_weights=False.

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
    )  # (batch, tokens, 3 * channels)
    q, k, v = projected.chunk(3, dim=-1)  # each (batch, tokens, channels)

    def split_heads(tensor: torch.Tensor) -> torch.Tensor:
        per_head = tensor.view(batch, tokens, heads, dim).transpose(
            1, 2
        )  # (batch, heads, tokens, dim)
        return per_head

    # The only layout where a whole head is a slice.
    attended = nn.functional.scaled_dot_product_attention(
        split_heads(q), split_heads(k), split_heads(v)
    )  # (batch, heads, tokens, dim)

    attended = mask(attended)  # (batch, heads, tokens, dim)

    merged = attended.transpose(1, 2).reshape(
        batch, tokens, channels
    )  # (batch, tokens, channels)
    output = attention.out_proj(merged)  # (batch, tokens, channels)

    # need_weights=False at the call site, so the second element is never read.
    return output, None


def head_mask(rate: float) -> HeadMask:
    """Whole attention heads. Requires the attention forward wrapper."""
    operator = HeadMask(rate)
    return operator


def channel_mask(rate: float) -> GroupChannelMask:
    """Whole embedding channels, shared across tokens."""
    operator = GroupChannelMask(rate, group_size=1)
    return operator


def fixed_head_mask(head_index: int):
    """Factory matching plug_dropout's factory contract, a callable taking a rate.

    plug_dropout calls factory(rate). Here the rate slot is unused because the
    head to remove is fixed in advance, so the returned closure ignores it.
    """

    def build(_rate: float) -> FixedHeadMask:
        operator = FixedHeadMask(head_index)
        return operator

    return build


def scale_up(mean, std):
    """Factory matching plug_dropout's factory contract, a callable taking a rate.

    The normalization constants are dataset-specific and are not available at
    the registry level, so they are bound here by the caller.
    """

    def build(rate: float) -> ScaleUp:
        operator = ScaleUp(rate, mean, std)
        return operator

    return build


# Every entry takes a rate and returns a module perturbing (batch, tokens,
# channels), so any of these is a drop-in for nn.Dropout in plug_dropout's
# factory argument.
OPERATORS: dict[str, type[nn.Module]] = {
    "dropout": nn.Dropout,
    "head_mask": head_mask,
    "channel_mask": channel_mask,
    "token_mask": TokenMask,
    # 2 probes the basis did not contain, both motivated in
    # docs/attack-design/improving-the-defense.md. token_substitute is
    # token_mask without the off-manifold component, token_block_mask is
    # token_mask with the geometry of a local trigger.
    "token_substitute": TokenSubstitute,
    "token_block_mask": TokenBlockMask,
    "droppath": DropPath,
    "gaussian": GaussianNoise,
    # Same isotropic covariance as gaussian, lower estimator variance. See the
    # class docstring: it exists to test a prediction of the trace estimator
    # reading.
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

# The operators that remove whole structures (tokens, channels, heads or a
# branch) rather than disturbing every entry, which the input_pixels rule below
# applies to as a set.
STRUCTURED_OPERATORS: frozenset[str] = frozenset(
    {"token_mask", "channel_mask", "head_mask", "droppath"}
)

# Operator and position pairs that may not attach, with the reason. Both rules
# cover cases where the operator runs happily and produces a complete, plausible,
# wrong answer, which is worse than a crash.
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
    operator = OPERATORS[name]
    return operator
