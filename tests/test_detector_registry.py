"""The registry's tables must agree with each other and with the modules they describe.

A detector missing from a table surfaces as a KeyError deep inside a run, which
is the wrong place. These tests make the registry refuse an incomplete entry at
import time of the suite instead, and pin the 1 constant that used to be copied
into 4 files.
"""

import pytest

from detectors import (
    CROSS_FITTED,
    DATA_REQUIREMENT,
    DETECTOR_BUILDERS,
    DETECTOR_HYPERPARAMETERS,
    DETECTOR_NAMES,
    EXPERIMENTAL_DETECTOR_NAMES,
    FORWARD_PASSES_PER_INPUT,
    NEEDS_FITTING,
    PRECISION_POLICY,
    STRIP_OVERLAYS,
    DetectorContext,
    effective_precision,
)
from detectors import strip as strip_module

EVERY_NAME = DETECTOR_NAMES + EXPERIMENTAL_DETECTOR_NAMES

TABLES = {
    "FORWARD_PASSES_PER_INPUT": FORWARD_PASSES_PER_INPUT,
    "DATA_REQUIREMENT": DATA_REQUIREMENT,
    "PRECISION_POLICY": PRECISION_POLICY,
    "DETECTOR_HYPERPARAMETERS": DETECTOR_HYPERPARAMETERS,
    "DETECTOR_BUILDERS": DETECTOR_BUILDERS,
}


@pytest.mark.parametrize("table_name", sorted(TABLES))
def test_every_registered_name_has_a_row_in_every_table(table_name):
    assert set(TABLES[table_name]) == set(EVERY_NAME), table_name


def test_the_set_memberships_name_registered_detectors_only():
    assert NEEDS_FITTING <= set(EVERY_NAME)
    assert CROSS_FITTED <= set(EVERY_NAME)
    assert not set(DETECTOR_NAMES) & set(EXPERIMENTAL_DETECTOR_NAMES)


def test_precision_policy_takes_the_2_known_values():
    assert set(PRECISION_POLICY.values()) <= {"autocast", "float32"}


def test_strip_overlays_has_1_source_and_is_still_8():
    assert STRIP_OVERLAYS is strip_module.DEFAULT_NUM_OVERLAYS
    assert STRIP_OVERLAYS == 8
    assert DETECTOR_HYPERPARAMETERS["strip"]["overlays"] == STRIP_OVERLAYS
    assert FORWARD_PASSES_PER_INPUT["strip"] == STRIP_OVERLAYS


def test_effective_precision_follows_the_context_for_autocast_methods():
    import torch

    cpu = DetectorContext(
        model=torch.nn.Identity(), device=torch.device("cpu"), mean=(0.0,), std=(1.0,)
    )
    assert effective_precision("strip", cpu) == "float32"
    assert effective_precision("confidence", cpu) == "float32"
