"""BIDS utilities for filename parsing, ID formatting, and data writing."""

import re

import mne
import mne_bids
import numpy as np


def format_id(raw_id):
    """Format a raw subject ID to a standardized form.

    Pads the numeric portion to 4 digits and preserves the letter suffix
    with an underscore separator.

    Parameters
    ----------
    raw_id : str
        Raw ID string, e.g. ``"42_P"`` or ``"42P"``.

    Returns
    -------
    str
        Formatted ID, e.g. ``"0042_P"``.

    Raises
    ------
    ValueError
        If the ID cannot be split into numeric and letter parts.
    """
    # Ensure underscore separator exists
    if "_" not in raw_id:
        # Find boundary between digits and letters
        for i, ch in enumerate(raw_id):
            if ch.isalpha():
                raw_id = raw_id[:i] + "_" + raw_id[i:]
                break

    parts = raw_id.split("_", 1)
    if len(parts) != 2:
        raise ValueError(f"Cannot parse ID: {raw_id}")

    numeric_part = parts[0]
    letter_part = parts[1]
    padded_numeric_part = numeric_part.zfill(4)
    return f"{padded_numeric_part}_{letter_part}"


def eb_id_transform(file):
    """Transform an eye-tracking subject ID to standardized format.

    Handles IDs with or without ``Q`` prefix and various formatting
    inconsistencies from the eye-tracking system.

    Parameters
    ----------
    file : str
        Raw eye-tracking participant folder name, e.g. ``"Q248_P"``
        or ``"281_M1"``.

    Returns
    -------
    str
        Standardized ID, e.g. ``"0248_P"`` or ``"0281_M1"``.
    """
    # Remove leading "Q" or "q"
    if file.startswith("Q") or file.startswith("q"):
        file = file[1:]

    # If no underscore, find first alphabetic character and insert one
    if "_" not in file:
        for i, ch in enumerate(file):
            if ch.isalpha():
                break
        file = file[:i] + "_" + file[i:]

    return format_id(file)


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
