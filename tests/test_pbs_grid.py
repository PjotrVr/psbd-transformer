"""Smoke and correctness tests for the general PBS grid flattener.

Pure string and file generation, no cluster and no GPU, so the whole suite runs
in milliseconds. Covers the flatten-vs-in-job partition, command substitution,
the continue-on-error wrapper, filenames, and the input-validation guards.
"""

import pytest

from pbs.grid import (
    PbsHeader,
    build_pbs_grid,
    cartesian_product,
    write_pbs_grid,
)


def test_cartesian_product_counts_and_shape():
    combos = cartesian_product({"a": [1, 2], "b": ["x", "y", "z"]})
    assert len(combos) == 6
    assert {"a": 1, "b": "x"} in combos
    assert cartesian_product({}) == [{}]


def test_flatten_everything_except_one_matches_the_psbd_shape():
    # The motivating case: flatten every axis except the rate, so one job sweeps
    # all rates and there is one job per (checkpoint, position).
    grid = {
        "checkpoint": ["vit_cifar100_wanet_0_01", "vit_cifar100_wanet_0_1"],
        "position": ["before_attention", "before_mlp_residual", "pre_residual"],
        "rate": [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    }
    jobs = build_pbs_grid(
        "python psbd_dropout_sweep.py --checkpoint {checkpoint} --position {position} --rate {rate}",
        grid,
        keep_in_job=("rate",),
    )
    # 2 checkpoints x 3 positions flattened = 6 files.
    assert len(jobs) == 6
    filenames = {name for name, _ in jobs}
    assert "grid_vit_cifar100_wanet_0_01_before_attention.pbs" in filenames

    # Each file sweeps all 10 rates in-job: 10 command lines, each substituted.
    _, content = next((n, c) for n, c in jobs if "0_01_before_attention" in n)
    assert content.count("psbd_dropout_sweep.py") == 10
    assert "--rate 0.05" in content
    assert "--rate 0.9" in content
    assert "--checkpoint vit_cifar100_wanet_0_01" in content
    assert "--position before_attention" in content


def test_error_handling_wrapper_is_present():
    jobs = build_pbs_grid("echo {x}", {"x": [1, 2]}, keep_in_job=())
    _, content = jobs[0]
    # A failing command must be logged and skipped, not abort the job.
    assert "run_command()" in content
    assert "[FAILED rc=$?]" in content
    assert "eval" in content


def test_no_flatten_produces_one_job_with_full_product():
    # Keep both axes in-job: a single file that runs the whole 2x2 grid.
    jobs = build_pbs_grid(
        "run {a} {b}", {"a": [1, 2], "b": [3, 4]}, keep_in_job=("a", "b")
    )
    assert len(jobs) == 1
    _, content = jobs[0]
    assert content.count("run ") == 4


def test_all_flatten_produces_one_job_per_combination():
    jobs = build_pbs_grid("run {a} {b}", {"a": [1, 2], "b": [3, 4]}, keep_in_job=())
    assert len(jobs) == 4
    for _, content in jobs:
        # One flattened point per file means exactly one command.
        assert content.count("run ") == 1


def test_value_sanitizing_in_filenames():
    jobs = build_pbs_grid("run {lr}", {"lr": [0.05, 0.5]}, keep_in_job=())
    filenames = sorted(name for name, _ in jobs)
    assert filenames == ["grid_0_05.pbs", "grid_0_5.pbs"]


def test_header_fields_reach_the_script():
    header = PbsHeader(
        queue="gpu",
        walltime="02:30:00",
        job_name_prefix="psbd",
        log_dir="logs/psbd_sweep",
    )
    jobs = build_pbs_grid("run {x}", {"x": [1]}, keep_in_job=(), header=header)
    _, content = jobs[0]
    assert "#PBS -q gpu" in content
    assert "#PBS -l walltime=02:30:00" in content
    assert "#PBS -N psbd_1" in content
    assert "logs/psbd_sweep" in content
    assert 'export http_proxy="http://10.150.1.1:3128"' in content


def test_write_creates_files_and_log_dir(tmp_path):
    out = tmp_path / "jobs"
    header = PbsHeader(log_dir=str(tmp_path / "logs" / "grid"))
    paths = write_pbs_grid(
        "run {a}", {"a": [1, 2, 3]}, str(out), keep_in_job=(), header=header
    )
    assert len(paths) == 3
    for path in paths:
        assert path.endswith(".pbs")
        with open(path) as handle:
            assert "#!/bin/bash" in handle.read()
    assert (tmp_path / "logs" / "grid").is_dir()


def test_empty_grid_rejected():
    with pytest.raises(ValueError, match="grid is empty"):
        build_pbs_grid("run", {}, keep_in_job=())


def test_empty_value_list_rejected():
    with pytest.raises(ValueError, match="non-empty list"):
        build_pbs_grid("run {a}", {"a": []}, keep_in_job=())


def test_unknown_keep_in_job_rejected():
    with pytest.raises(ValueError, match="not in grid"):
        build_pbs_grid("run {a}", {"a": [1]}, keep_in_job=("missing",))
