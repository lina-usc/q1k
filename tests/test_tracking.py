"""Tests for q1k.tracking.tools — tracking helper functions."""

import pandas as pd
import pytest

from q1k.config import PIPELINE_STAGES
from q1k.tracking.tools import (
    _extract_subject_from_path,
    _find_csv,
    _infer_group,
    _infer_site,
    compute_last_stage,
    has_skipped_steps,
)


class TestInferSite:
    def test_hsj(self):
        assert _infer_site("Q1K_HSJ_10043_P") == "hsj"

    def test_nim(self):
        assert _infer_site("Q1K_NIM_3530-3062_F1") == "hsj"

    def test_mhc(self):
        assert _infer_site("Q1K_MHC_20042_P") == "mni"

    def test_unknown(self):
        assert _infer_site("Q1K_XYZ_99999_P") == "unknown"

    def test_nan(self):
        assert _infer_site(float("nan")) is None


class TestInferGroup:
    def test_proband(self):
        assert _infer_group("Q1K_HSJ_10043_P") == "proband"

    def test_father(self):
        assert _infer_group("Q1K_HSJ_10043_F1") == "father"

    def test_mother(self):
        assert _infer_group("Q1K_HSJ_10043_M1") == "mother"

    def test_sibling(self):
        assert _infer_group("Q1K_HSJ_10043_S1") == "sibling"

    def test_unknown(self):
        assert _infer_group("Q1K_HSJ_10043") == "unknown"

    def test_nan(self):
        assert _infer_group(float("nan")) is None


class TestHasSkippedSteps:
    def test_contiguous(self):
        row = pd.Series({
            "EEG Raw Files": True,
            "BIDS": True,
            "Pylossless": True,
            "ET_sync_loss": False,
            "Segmentation": False,
            "Autoreject": False,
        })
        assert has_skipped_steps(row) is False

    def test_skipped(self):
        row = pd.Series({
            "EEG Raw Files": True,
            "BIDS": True,
            "Pylossless": False,
            "ET_sync_loss": False,
            "Segmentation": True,
            "Autoreject": False,
        })
        assert has_skipped_steps(row) is True

    def test_single_stage(self):
        row = pd.Series({
            "EEG Raw Files": True,
            "BIDS": False,
            "Pylossless": False,
            "ET_sync_loss": False,
            "Segmentation": False,
            "Autoreject": False,
        })
        assert has_skipped_steps(row) is False

    def test_no_stages(self):
        row = pd.Series({s: False for s in PIPELINE_STAGES})
        assert has_skipped_steps(row) is False


class TestComputeLastStage:
    def test_completed(self):
        row = pd.Series({s: True for s in PIPELINE_STAGES})
        assert compute_last_stage(row) == "Completed"

    def test_partial(self):
        row = pd.Series({
            "EEG Raw Files": True,
            "BIDS": True,
            "Pylossless": True,
            "ET_sync_loss": False,
            "Segmentation": False,
            "Autoreject": False,
        })
        assert compute_last_stage(row) == "Pylossless"

    def test_none(self):
        row = pd.Series({s: False for s in PIPELINE_STAGES})
        assert compute_last_stage(row) == "None"


class TestExtractSubjectFromPath:
    def test_bids_path(self):
        assert _extract_subject_from_path(
            "/data/sub-0042P/ses-01/eeg/file.edf"
        ) == "0042P"

    def test_no_match(self):
        assert _extract_subject_from_path("/data/file.edf") is None


class TestFindCsv:
    def test_found(self, tmp_path):
        (tmp_path / "Q1K_TaskCompletion_2024.csv").touch()
        result = _find_csv(tmp_path, "TaskCompletion")
        assert result.name == "Q1K_TaskCompletion_2024.csv"

    def test_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="No CSV"):
            _find_csv(tmp_path, "NonExistent")
