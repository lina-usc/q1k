"""AutoReject pipeline for epoch cleaning.

Applies the AutoReject algorithm to automatically repair or reject
bad epochs.
"""

from pathlib import Path

import mne
from autoreject import AutoReject


def run_autoreject(file_path, out_path):
    """Apply AutoReject to epoch data.

    Parameters
    ----------
    file_path : str or Path
        Path to input epoch file (``.fif``).
    out_path : str or Path
        Output directory for cleaned epochs.
    """
    file_path = Path(file_path)
    out_path = Path(out_path)
    fname = file_path.name

    print(f"Current file: {fname}")

    # Read epochs
    epochs = mne.read_epochs(file_path)

    # Apply AutoReject
    ar = AutoReject()
    epochs.load_data()
    epochs_clean = ar.fit_transform(epochs)

    # Save cleaned epochs
    out_path.mkdir(parents=True, exist_ok=True)
    out_file = out_path / fname
    epochs_clean.save(out_file, overwrite=True)
    print(f"Saved cleaned epochs: {out_file}")
