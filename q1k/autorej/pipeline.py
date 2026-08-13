"""AutoReject pipeline for epoch cleaning.

Applies the AutoReject algorithm to automatically repair or reject
bad epochs.
"""

import argparse
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
        # Reading epochs
        epochs = mne.read_epochs(file_path, verbose=False)

        # Apply AutoReject (OLD style - default parameters)
        ar = AutoReject()
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


def main():
    """Run AutoReject from the command line.

    This entry point is used by the Slurm wrapper script.
    """
    parser = argparse.ArgumentParser(
        prog="python -m q1k.autorej.pipeline",
        description="Apply AutoReject to one epoch FIF file.",
    )
    parser.add_argument("file_path", help="Input epoch FIF file.")
    parser.add_argument("out_path", help="Output directory for cleaned epochs.")
    args = parser.parse_args()
    run_autoreject(args.file_path, args.out_path)


if __name__ == "__main__":
    main()
