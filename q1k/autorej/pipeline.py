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

    print(f"Processing file: {fname}")
    try:
        # Read epochs
        epochs = mne.read_epochs(file_path, verbose=False)
        # Only process EEG channels
        eeg_picks = mne.pick_types(epochs.info, meg=False, eeg=True, exclude='bads')

        if len(eeg_picks) == 0:
            raise ValueError(f"No EEG channels found in {fname}")

        # Apply AutoReject on EEG channels only
        ar = AutoReject(picks=eeg_picks, random_state=42, n_jobs=1, verbose=False)
        epochs.load_data()
        epochs_clean = ar.fit_transform(epochs)

        # Save cleaned epochs
        out_path.mkdir(parents=True, exist_ok=True)
        out_file = out_path / fname
        epochs_clean.save(out_file, overwrite=True)
        print(f"✓ Saved: {out_file}")
        print(f"  Dropped {len(epochs) - len(epochs_clean)}/{len(epochs)} epochs")
        return out_file
    except Exception as e:
        print(f"✗ Error processing {fname}: {e}")
        raise
