"""CLI for Stage 5: AutoReject epoch cleaning."""

import argparse
import glob
import os
from pathlib import Path

from q1k.config import DEFAULT_RUN_ID, DEFAULT_SESSION_ID, VALID_TASKS


def create_parser():
    parser = argparse.ArgumentParser(
        prog="q1k-autorej",
        description=(
            "Stage 5: Apply AutoReject to clean epochs. "
            "Can run locally or submit Slurm jobs."
        ),
    )
    parser.add_argument(
        "--project-path", required=True,
        help="Path to the project experimental directory.",
    )
    parser.add_argument(
        "--task", required=True, choices=["PLR", "GO", "VEP"],
        help="Task code to process.",
    )
    parser.add_argument(
        "--subject", default=True,
        help="Single subject ID to process(e.g., HSJ10046F1).",
    )
    parser.add_argument(
        "--session", default="01",
        help=f"Session ID (default:01).",
    )
    parser.add_argument(
        "--run", default="1",
        help=f"Run ID (default:1).",
    )

    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()

    if not args.subject:
        parser.error("--subject must be specified.")

    from q1k.autorej.pipeline import run_autoreject

    project_path = Path(args.project_path)

    input_file = (
        project_path
        / "derivatives"
        / "segment"
        / "epoch_fif_files"
        / args.task
        / f"sub-{args.subject}_ses-{args.session}_task-{args.task}_run-{args.run}_eeg_epo.fif"
    )

    out_path = (
        project_path
        / "derivatives"
        / "autoreject"
        / "epoch_fif_files"
        / args.task
    )

    print(f"Processing: {input_file}")
    print(f"Output to: {out_path}")

    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    run_autoreject(str(input_file), str(out_path))


if __name__ == "__main__":
    main()
