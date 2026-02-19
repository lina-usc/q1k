"""Tests for q1k.slurm — find_unprocessed utility."""

import os

from q1k.slurm import find_unprocessed


def test_find_unprocessed_basic(tmp_path):
    """Subjects in input but not output are returned."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()

    # Create input files
    (input_dir / "sub-0042P_ses-01_task-RS_run-1_eeg.edf").touch()
    (input_dir / "sub-0043P_ses-01_task-RS_run-1_eeg.edf").touch()
    (input_dir / "sub-0100F1_ses-01_task-RS_run-1_eeg.edf").touch()

    # Create output for only one subject
    (output_dir / "sub-0042P_ses-01_task-RS_run-1_eeg.edf").touch()

    unprocessed = find_unprocessed(
        str(input_dir / "*.edf"),
        str(output_dir / "*.edf"),
    )

    unprocessed_basenames = {os.path.basename(f) for f in unprocessed}
    assert "sub-0043P_ses-01_task-RS_run-1_eeg.edf" in unprocessed_basenames
    assert "sub-0100F1_ses-01_task-RS_run-1_eeg.edf" in unprocessed_basenames
    assert "sub-0042P_ses-01_task-RS_run-1_eeg.edf" not in unprocessed_basenames


def test_find_unprocessed_all_processed(tmp_path):
    """When all inputs have outputs, returns empty list."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()

    (input_dir / "sub-0042P_ses-01_task-RS_run-1_eeg.edf").touch()
    (output_dir / "sub-0042P_ses-01_task-RS_run-1_eeg.edf").touch()

    unprocessed = find_unprocessed(
        str(input_dir / "*.edf"),
        str(output_dir / "*.edf"),
    )
    assert len(unprocessed) == 0


def test_find_unprocessed_no_outputs(tmp_path):
    """When no outputs exist, all inputs are returned."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()

    (input_dir / "sub-0042P_ses-01_task-RS_run-1_eeg.edf").touch()
    (input_dir / "sub-0043P_ses-01_task-RS_run-1_eeg.edf").touch()

    unprocessed = find_unprocessed(
        str(input_dir / "*.edf"),
        str(output_dir / "*.edf"),
    )
    assert len(unprocessed) == 2
