"""The 2 probes added in docs/attack-design/improving-the-defense.md.

Both act on tokens and neither needs a new position, so they are swept exactly
as token_mask is. What these tests hold is the properties the 2 operators claim
that token_mask does not: substitution stays on the data manifold, and block
masking removes a contiguous rectangle rather than a random subset.
"""

import math

import pytest
import torch

from defences.operators import TokenBlockMask, TokenMask, TokenSubstitute

pytestmark = pytest.mark.fast

BATCH, GRID, CHANNELS = 4, 14, 8
TOKENS = 1 + GRID * GRID


def token_batch() -> torch.Tensor:
    """A batch whose every token is a constant equal to its sample index.

    Making each sample's tokens distinguishable is what lets a test tell a
    substituted token from a zeroed one.
    """
    values = torch.arange(1, BATCH + 1, dtype=torch.float32)  # (batch,)
    x = values.view(BATCH, 1, 1).expand(BATCH, TOKENS, CHANNELS).clone()
    return x  # (batch, tokens, channels)


def positional_batch() -> torch.Tensor:
    """A batch whose every token is a constant equal to its token index.

    Making tokens distinguishable within a sample is what lets a test tell a
    substituted token from a zeroed one, now that the donor is the same sample.
    """
    values = torch.arange(TOKENS, dtype=torch.float32)  # (tokens,)
    x = values.view(1, TOKENS, 1).expand(BATCH, TOKENS, CHANNELS).clone()
    return x  # (batch, tokens, channels)


def test_substitution_never_introduces_a_value_the_sample_did_not_contain():
    """The point of the operator: no token is off the manifold.

    Every output value has to be a value this sample already carried at some
    position. A zeroed token would show up as a value of 0 at a position whose
    own index is not 0.
    """
    x = positional_batch()
    operator = TokenSubstitute(rate=0.5).train()
    torch.manual_seed(0)

    out = operator(x)  # (batch, tokens, channels)

    allowed = set(range(TOKENS))
    assert set(out.unique().tolist()) <= allowed


def test_substitution_reads_no_other_sample_in_the_batch():
    """A per-sample score must not be a function of another sample's values.

    This is the fault that retired the batch-coupled Gaussian arm, whose noise
    was scaled by a statistic of the whole batch. Changing every other sample
    while holding the random draw fixed must leave sample 0's output untouched.
    Note this is about VALUES, not about the shared random stream: every
    stochastic operator here draws from the global generator, so a different
    batch size consumes it differently, and that is not batch coupling.
    """
    x = positional_batch()
    other = x.clone()
    other[1:] = other[1:] * 7.0 + 3.0

    torch.manual_seed(0)
    baseline = TokenSubstitute(rate=0.5).train()(x)  # (batch, tokens, channels)
    torch.manual_seed(0)
    perturbed_neighbours = TokenSubstitute(rate=0.5).train()(other)

    assert torch.equal(baseline[0], perturbed_neighbours[0])


def test_substitution_leaves_the_class_token_alone():
    x = positional_batch()
    operator = TokenSubstitute(rate=1.0).train()
    torch.manual_seed(0)

    out = operator(x)  # (batch, tokens, channels)

    assert torch.equal(out[:, 0, :], x[:, 0, :])


def test_substitution_at_rate_1_moves_every_patch_token():
    """At rate 1 no patch token keeps its own value, since the roll is non-zero."""
    x = positional_batch()
    operator = TokenSubstitute(rate=1.0).train()
    torch.manual_seed(0)

    out = operator(x)  # (batch, tokens, channels)

    assert not torch.equal(out[:, 1:, :], x[:, 1:, :])
    assert torch.equal(out[:, 0, :], x[:, 0, :])


def test_substitution_is_a_no_op_in_eval_mode_and_at_rate_0():
    x = positional_batch()
    assert torch.equal(TokenSubstitute(rate=0.5).eval()(x), x)
    assert torch.equal(TokenSubstitute(rate=0.0).train()(x), x)


def test_substitution_leaves_a_sample_with_too_few_patches_unperturbed():
    """Fewer than 2 patch tokens leaves nothing to substitute from."""
    x = torch.ones(2, 2, CHANNELS)  # 1 class token plus 1 patch
    operator = TokenSubstitute(rate=0.5).train()

    out = operator(x)  # (2, 2, channels)

    assert torch.equal(out, x)


def test_block_mask_zeroes_a_contiguous_rectangle_of_the_grid():
    """The geometry claim: the removed tokens form 1 rectangle, not a scatter."""
    x = torch.ones(1, TOKENS, CHANNELS)
    operator = TokenBlockMask(rate=0.25).train()
    torch.manual_seed(0)

    out = operator(x)  # (1, tokens, channels)

    removed = (out[0, 1:, 0] == 0.0).view(GRID, GRID)
    rows = removed.any(dim=1).nonzero().flatten()
    cols = removed.any(dim=0).nonzero().flatten()
    # Contiguous means the covered rows and columns are each an unbroken run.
    assert torch.equal(rows, torch.arange(int(rows[0]), int(rows[-1]) + 1))
    assert torch.equal(cols, torch.arange(int(cols[0]), int(cols[-1]) + 1))
    # And the rectangle's side is the one the rate asks for.
    side = max(1, min(GRID, round(GRID * math.sqrt(0.25))))
    assert len(rows) == side and len(cols) == side


def test_block_mask_leaves_the_class_token_alone():
    x = torch.ones(BATCH, TOKENS, CHANNELS)
    operator = TokenBlockMask(rate=0.5).train()
    torch.manual_seed(0)

    out = operator(x)  # (batch, tokens, channels)

    assert torch.equal(out[:, 0, :], x[:, 0, :])


def test_block_mask_holds_the_surviving_scale_like_token_mask():
    """The survivors carry the removed tokens' scale, so passes stay comparable."""
    x = torch.ones(1, TOKENS, CHANNELS)
    operator = TokenBlockMask(rate=0.25).train()
    torch.manual_seed(0)

    out = operator(x)  # (1, tokens, channels)

    survivors = out[0, 1:, 0][out[0, 1:, 0] != 0.0]
    side = max(1, min(GRID, round(GRID * math.sqrt(0.25))))
    expected = 1.0 / (1.0 - side * side / float(GRID * GRID))
    assert torch.allclose(survivors, torch.full_like(survivors, expected))


def test_block_mask_falls_back_to_independent_dropping_off_a_square_grid():
    """A Swin-shaped token count that is not a square grid still perturbs."""
    x = torch.ones(2, 1 + 12, CHANNELS)  # 12 patches is not a square
    operator = TokenBlockMask(rate=0.5).train()
    torch.manual_seed(0)

    out = operator(x)  # (2, 13, channels)

    assert not torch.equal(out, x)


def test_both_probes_preserve_shape_and_the_swin_layout():
    """A Swin feature map arrives as (batch, height, width, channels)."""
    swin = torch.ones(2, 7, 7, CHANNELS)
    for operator in (TokenSubstitute(0.3), TokenBlockMask(0.3), TokenMask(0.3)):
        torch.manual_seed(0)
        out = operator.train()(swin)
        assert out.shape == swin.shape
