"""The detector job generator's invariants, checked without a cluster.

The groups must partition the registry, a job name must survive qstat's 10
character truncation, every emitted flag must be one cli.baselines accepts, and
a dry run must write nothing.
"""

import importlib.util
import json
import os
import subprocess
import sys

import pytest

from detectors import DETECTOR_NAMES

GENERATOR = os.path.join(os.getcwd(), "pbs", "generate_detector_jobs.py")


@pytest.fixture(scope="module")
def generator():
    spec = importlib.util.spec_from_file_location("generate_detector_jobs", GENERATOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_groups_partition_the_registry(generator):
    grouped = [name for names in generator.DETECTOR_GROUPS.values() for name in names]
    assert sorted(grouped) == sorted(DETECTOR_NAMES)
    assert len(grouped) == len(set(grouped))


def test_job_names_survive_qstat_truncation(generator):
    for group, letter in generator.GROUP_LETTER.items():
        assert len(f"det_{letter}_001") <= 10, group


def test_emitted_flags_are_accepted_by_cli_baselines(generator):
    args = generator.build_arg_parser().parse_args([])
    cell = {"folder": "vit_gtsrb_benign", "dataset": "gtsrb", "attack": "benign"}
    script = generator.render_job(
        "cheap", 1, [(cell, list(generator.DETECTOR_GROUPS["cheap"]))], args
    )

    generator.verify_flags(script)
    assert "--probe-attack badnet_a2o" in script
    assert "--results-dir " + args.results_dir in script
    assert "#PBS -N det_c_001" in script


def test_a_wrong_flag_is_refused(generator):
    with pytest.raises(SystemExit, match="rejects"):
        generator.verify_flags("python -m cli.baselines --no-such-flag 1")


def test_walltime_has_a_floor_and_a_margin(generator):
    assert generator.walltime_text(10.0) == "06:00:00"
    assert generator.walltime_text(300.0) == "10:00:00"


def test_dry_run_writes_nothing(generator, tmp_path):
    coverage = {
        "cells": [
            {
                "folder_name": "vit_gtsrb_badnet_a2o_0_05",
                "dataset": "gtsrb",
                "attack": "badnet_a2o",
                "asr_class": "clears",
            },
            {
                "folder_name": "vit_gtsrb_sig_0_05",
                "dataset": "gtsrb",
                "attack": "sig",
                "asr_class": "below_bar",
            },
        ]
    }
    declaration = {"benign_reference": {"gtsrb": "vit_gtsrb_benign", "_comment": "x"}}
    results = tmp_path / "results"
    (results / "coverage").mkdir(parents=True)
    (results / "coverage" / "coverage.json").write_text(json.dumps(coverage))
    (tmp_path / "basis.json").write_text(json.dumps(declaration))

    completed = subprocess.run(
        [
            sys.executable,
            GENERATOR,
            "--dry-run",
            "--results-dir",
            str(results),
            "--declaration",
            str(tmp_path / "basis.json"),
            "--group",
            "cd_l",
        ],
        capture_output=True,
        text=True,
        check=True,
        cwd=os.getcwd(),
    )
    assert "2 checkpoints with work" in completed.stdout
    assert "nothing written" in completed.stdout
    assert not (tmp_path / "pbs").exists()
