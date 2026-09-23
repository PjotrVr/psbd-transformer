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


def test_substitution_never_introduces_a_value_the_batch_did_not_contain():
    """The point of the operator: no token is off the manifold.

    Every output value has to be a value some sample already carried. A zeroed
    token would show up as a 0, which no sample carries.
    """
    x = token_batch()
    operator = TokenSubstitute(rate=0.5).train()
    torch.manual_seed(0)

    out = operator(x)  # (batch, tokens, channels)

    allowed = set(range(1, BATCH + 1))
    assert set(out.unique().tolist()) <= allowed
    assert 0.0 not in out.unique().tolist()


def test_substitution_leaves_the_class_token_alone():
    x = token_batch()
    operator = TokenSubstitute(rate=1.0).train()
    torch.manual_seed(0)

    out = operator(x)  # (batch, tokens, channels)

    assert torch.equal(out[:, 0, :], x[:, 0, :])


def test_substitution_at_rate_1_replaces_every_patch_token():
    """At rate 1 every patch token comes from the donor, which is the roll by 1."""
    x = token_batch()
    operator = TokenSubstitute(rate=1.0).train()
    torch.manual_seed(0)

    out = operator(x)  # (batch, tokens, channels)

    donor = torch.roll(x, shifts=1, dims=0)  # (batch, tokens, channels)
    assert torch.equal(out[:, 1:, :], donor[:, 1:, :])


def test_substitution_is_a_no_op_in_eval_mode_and_at_rate_0():
    x = token_batch()
    assert torch.equal(TokenSubstitute(rate=0.5).eval()(x), x)
    assert torch.equal(TokenSubstitute(rate=0.0).train()(x), x)


def test_substitution_leaves_a_single_sample_batch_unperturbed():
    """With 1 sample there is no donor, so the pass must not be silently masked."""
    x = token_batch()[:1]  # (1, tokens, channels)
    operator = TokenSubstitute(rate=0.5).train()

    out = operator(x)  # (1, tokens, channels)

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
