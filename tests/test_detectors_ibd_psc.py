"""IBD-PSC's calibration search, checked without a model.

Algorithm 1 is exercised on real checkpoints by the smoke and the sign gate.
What a unit test can pin is the search over omega: it stops at the first factor
whose trace crosses xi, returns the paper's factor untouched when that one
already crosses, and falls back to the last factor with k = L when none does.
"""

import torch.nn as nn

from detectors import ibd_psc


def stub_layers(count: int) -> list[nn.LayerNorm]:
    layers = [nn.LayerNorm(4) for _ in range(count)]
    return layers


def fake_selection(crossing_factor: float | None, total_layers: int):
    """A select_start_layer_count stand-in crossing xi only from crossing_factor up."""

    def select(model, loader, device, layers, factor, threshold, use_bfloat16):
        if crossing_factor is not None and factor >= crossing_factor:
            return 2, [0.1, threshold + 0.1]
        return total_layers, [0.1] * total_layers

    return select


def test_the_search_stops_at_the_first_crossing_factor(monkeypatch):
    monkeypatch.setattr(ibd_psc, "select_start_layer_count", fake_selection(3.0, 5))
    factor, start, trace = ibd_psc.calibrate_scaling_factor(
        None, None, None, stub_layers(5), (1.5, 2.0, 3.0, 5.0), 0.6, True
    )
    assert factor == 3.0
    assert start == 2
    assert trace[-1] > 0.6


def test_the_papers_factor_is_kept_when_it_already_crosses(monkeypatch):
    monkeypatch.setattr(ibd_psc, "select_start_layer_count", fake_selection(1.5, 5))
    factor, start, _ = ibd_psc.calibrate_scaling_factor(
        None, None, None, stub_layers(5), (1.5, 2.0, 3.0), 0.6, True
    )
    assert factor == 1.5
    assert start == 2


def test_no_crossing_returns_the_last_factor_at_every_layer(monkeypatch):
    monkeypatch.setattr(ibd_psc, "select_start_layer_count", fake_selection(None, 5))
    factor, start, trace = ibd_psc.calibrate_scaling_factor(
        None, None, None, stub_layers(5), (1.5, 2.0), 0.6, True
    )
    assert factor == 2.0
    assert start == 5
    assert max(trace) < 0.6
