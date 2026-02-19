"""Tests for q1k.segment.tasks — epoch segmentation functions."""

import numpy as np
import pytest

import mne

from q1k.segment.tasks import (
    SEGMENT_FUNCTIONS,
    TASK_PARAMS,
    _detect_rsrio,
    segment_task,
    segment_vep,
    segment_aep,
    segment_go,
    segment_plr,
)


def _make_raw_with_annotations(sfreq=250.0, duration=60.0, annotations=None):
    """Helper: create a raw object with specified annotations."""
    n_channels = 10
    ch_names = [f"E{i}" for i in range(1, n_channels + 1)]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    data = np.random.RandomState(0).randn(n_channels, int(sfreq * duration)) * 1e-6
    raw = mne.io.RawArray(data, info)

    if annotations is not None:
        onsets, durations, descriptions = zip(*annotations)
        raw.set_annotations(mne.Annotations(
            onset=list(onsets),
            duration=list(durations),
            description=list(descriptions),
        ))

    return raw


class TestDetectRsrio:
    def test_with_eyeo(self):
        assert _detect_rsrio({"eyeo": 1, "DIN1": 2}) is True

    def test_with_comm(self):
        assert _detect_rsrio({"comm": 1, "DIN1": 2}) is True

    def test_without_rsrio_markers(self):
        assert _detect_rsrio({"vs01": 1, "vs02": 2, "dbrk": 3}) is False

    def test_empty_dict(self):
        assert _detect_rsrio({}) is False


class TestSegmentVep:
    def test_returns_epochs(self):
        annotations = [
            (1.0, 0.0, "sv06_d"),
            (3.0, 0.0, "sv06_d"),
            (5.0, 0.0, "sv15_d"),
            (7.0, 0.0, "sv15_d"),
        ]
        raw = _make_raw_with_annotations(annotations=annotations)
        events, event_dict = mne.events_from_annotations(raw)

        epochs, epoch_event_dict, conditions = segment_vep(
            raw, events, event_dict
        )
        assert isinstance(epochs, mne.Epochs)
        assert "sv06_d" in epoch_event_dict
        assert "sv15_d" in epoch_event_dict
        assert len(conditions) == 2


class TestSegmentAep:
    def test_returns_epochs(self):
        annotations = [
            (1.0, 0.0, "ae06_d"),
            (5.0, 0.0, "ae40_d"),
        ]
        raw = _make_raw_with_annotations(annotations=annotations)
        events, event_dict = mne.events_from_annotations(raw)

        epochs, epoch_event_dict, conditions = segment_aep(
            raw, events, event_dict
        )
        assert isinstance(epochs, mne.Epochs)
        assert "ae06_d" in epoch_event_dict


class TestSegmentGo:
    def test_returns_epochs(self):
        annotations = [
            (1.0, 0.0, "dtbc_d"),
            (3.0, 0.0, "dtoc_d"),
            (5.0, 0.0, "dtgc_d"),
        ]
        raw = _make_raw_with_annotations(annotations=annotations)
        events, event_dict = mne.events_from_annotations(raw)

        epochs, epoch_event_dict, conditions = segment_go(
            raw, events, event_dict
        )
        assert isinstance(epochs, mne.Epochs)
        assert any("_d" in c for c in conditions)


class TestSegmentPlr:
    def test_returns_epochs(self):
        annotations = [
            (2.0, 0.0, "plro_d"),
            (8.0, 0.0, "plro_d"),
        ]
        raw = _make_raw_with_annotations(duration=20.0, annotations=annotations)
        events, event_dict = mne.events_from_annotations(raw)

        epochs, epoch_event_dict, conditions = segment_plr(
            raw, events, event_dict
        )
        assert isinstance(epochs, mne.Epochs)
        assert "plro_d" in epoch_event_dict
        assert conditions == ["plro_d"]


class TestSegmentTask:
    def test_dispatcher_vep(self):
        annotations = [(1.0, 0.0, "sv06_d"), (5.0, 0.0, "sv15_d")]
        raw = _make_raw_with_annotations(annotations=annotations)
        events, event_dict = mne.events_from_annotations(raw)

        epochs, _, _ = segment_task("VEP", raw, events, event_dict)
        assert isinstance(epochs, mne.Epochs)

    def test_dispatcher_unknown_task(self):
        raw = _make_raw_with_annotations()
        with pytest.raises(ValueError, match="Unknown task"):
            segment_task("INVALID", raw)

    def test_all_tasks_have_params(self):
        for task in SEGMENT_FUNCTIONS:
            assert task in TASK_PARAMS, f"Missing params for {task}"
