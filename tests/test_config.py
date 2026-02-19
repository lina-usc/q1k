"""Tests for q1k.config constants."""

from q1k.config import (
    FREQ_BANDS,
    FRONTAL_LEFT_ROI,
    FRONTAL_ROI,
    PARIETAL_LEFT_ROI,
    PIPELINE_STAGES,
    TEMPORAL_LEFT_ROI,
    VALID_TASKS,
)


def test_valid_tasks_are_strings():
    assert all(isinstance(t, str) for t in VALID_TASKS)
    assert len(VALID_TASKS) > 0


def test_freq_bands_ranges():
    for band, (fmin, fmax) in FREQ_BANDS.items():
        assert isinstance(band, str)
        assert fmin < fmax, f"{band}: fmin={fmin} >= fmax={fmax}"


def test_pipeline_stages_ordered():
    assert isinstance(PIPELINE_STAGES, list)
    assert len(PIPELINE_STAGES) >= 3
    assert PIPELINE_STAGES[0] == "EEG Raw Files"
    assert PIPELINE_STAGES[-1] == "Autoreject"


def test_roi_channels_nonempty():
    assert len(FRONTAL_ROI) > 0
    assert len(FRONTAL_LEFT_ROI) > 0
    assert len(PARIETAL_LEFT_ROI) > 0
    assert len(TEMPORAL_LEFT_ROI) > 0


def test_roi_channels_are_e_prefixed():
    for ch in FRONTAL_ROI + FRONTAL_LEFT_ROI + PARIETAL_LEFT_ROI + TEMPORAL_LEFT_ROI:
        assert ch.startswith("E"), f"Channel {ch} doesn't start with 'E'"
