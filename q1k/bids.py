"""BIDS utilities for filename parsing and data writing."""

import re

import mne
import mne_bids
import numpy as np


# Pattern for BIDS EEG filenames
_BIDS_PATTERN = re.compile(
    r"sub-(?P<subject>.*?)_ses-(?P<session>.*?)_task-(?P<task>.*?)_run-(?P<run>.*?)_eeg\.(?P<ext>edf|fif)"
)


def extract_bids_info(filename):
    """Extract subject, session, task, and run from a BIDS filename.

    Parameters
    ----------
    filename : str
        A BIDS-formatted filename, e.g.
        ``sub-0042P_ses-01_task-RS_run-1_eeg.edf``.

    Returns
    -------
    subject_id : str
    session_id : str
    task_id : str
    run_id : str

    Raises
    ------
    ValueError
        If the filename does not match the expected BIDS pattern.
    """
    match = _BIDS_PATTERN.match(filename)
    if not match:
        raise ValueError(
            f"Filename does not match BIDS pattern: {filename}"
        )
    return (
        match.group("subject"),
        match.group("session"),
        match.group("task"),
        match.group("run"),
    )


def fillna(raw, fill_val=0):
    """Replace NaN values in raw data with a fill value.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw MNE object (must be preloaded or will be loaded).
    fill_val : float
        Value to replace NaNs with. Default is 0.

    Returns
    -------
    mne.io.RawArray
        New RawArray with NaNs replaced.
    """
    return mne.io.RawArray(
        np.nan_to_num(raw.get_data(), nan=fill_val), raw.info
    )


def write_bids_eeg(
    eeg_raw,
    eeg_events,
    eeg_event_dict,
    subject_id,
    session_id,
    task_id,
    root,
    run_id="1",
):
    """Write EEG data to BIDS format.

    Parameters
    ----------
    eeg_raw : mne.io.Raw
        Raw EEG data (NaNs will be filled with 0).
    eeg_events : np.ndarray
        MNE events array.
    eeg_event_dict : dict
        Event ID mapping.
    subject_id : str
        BIDS subject identifier.
    session_id : str
        BIDS session identifier.
    task_id : str
        BIDS task identifier.
    root : str or Path
        BIDS root directory.
    run_id : str
        BIDS run identifier. Default is ``"1"``.

    Returns
    -------
    mne_bids.BIDSPath
        The BIDSPath of the written file.
    """
    eeg_raw = fillna(eeg_raw, fill_val=0)

    bids_path = mne_bids.BIDSPath(
        subject=subject_id,
        session=session_id,
        task=task_id,
        run=run_id,
        datatype="eeg",
        root=root,
    )

    mne_bids.write_raw_bids(
        raw=eeg_raw,
        bids_path=bids_path,
        events=eeg_events,
        event_id=eeg_event_dict,
        format="EDF",
        overwrite=True,
        allow_preload=True,
    )

    return bids_path
