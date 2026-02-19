"""Shared test fixtures for the Q1K test suite."""

import numpy as np
import pytest

import mne


@pytest.fixture
def synthetic_raw():
    """Create a synthetic Raw object with GSN-HydroCel-128 channel names."""
    n_channels = 129
    sfreq = 250.0
    duration = 10.0  # seconds

    ch_names = [f"E{i}" for i in range(1, n_channels + 1)]
    ch_types = ["eeg"] * n_channels
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    data = np.random.RandomState(42).randn(n_channels, int(sfreq * duration)) * 1e-6
    raw = mne.io.RawArray(data, info)
    return raw


@pytest.fixture
def synthetic_epochs(synthetic_raw):
    """Create synthetic epochs from raw data with 1-second events."""
    sfreq = synthetic_raw.info["sfreq"]
    n_events = 8
    events = np.array([
        [int(i * sfreq), 0, 1] for i in range(n_events)
    ])
    event_id = {"stim": 1}
    epochs = mne.Epochs(
        synthetic_raw, events, event_id=event_id,
        tmin=-0.2, tmax=0.5, baseline=None, preload=True,
    )
    return epochs


@pytest.fixture
def tmp_bids_dir(tmp_path):
    """Create a temporary BIDS-like directory structure."""
    # Create subject dirs with fake files
    for subj in ["0042P", "0043P", "0100F1"]:
        eeg_dir = tmp_path / f"sub-{subj}" / "ses-01" / "eeg"
        eeg_dir.mkdir(parents=True)
        (eeg_dir / f"sub-{subj}_ses-01_task-RS_run-1_eeg.edf").touch()

    # Create derivatives structure
    pyll = tmp_path / "derivatives" / "pylossless"
    for subj in ["0042P", "0043P"]:
        eeg_dir = pyll / f"sub-{subj}" / "ses-01" / "eeg"
        eeg_dir.mkdir(parents=True)
        (eeg_dir / f"sub-{subj}_ses-01_task-RS_run-1_eeg.edf").touch()

    sync = pyll / "derivatives" / "sync_loss"
    for subj in ["0042P"]:
        eeg_dir = sync / f"sub-{subj}" / "ses-01" / "eeg"
        eeg_dir.mkdir(parents=True)
        (eeg_dir / f"sub-{subj}_ses-01_task-RS_run-1_eeg.edf").touch()

    seg = sync / "derivatives" / "segment" / "epoch_fif_files" / "RS"
    seg.mkdir(parents=True)
    (seg / "sub-0042P_ses-01_task-RS_run-1_eeg_epo.fif").touch()

    ar = seg.parent.parent / "derivatives" / "autorej" / "epoch_fif_files" / "RS"
    ar.mkdir(parents=True)

    return tmp_path
