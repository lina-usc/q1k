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
        "--task", required=True, choices=VALID_TASKS,
        help="Task code to process.",
    )
    parser.add_argument(
        "--subject", default=None,
        help="Single subject ID to process.",
    )
    parser.add_argument(
        "--all", dest="process_all", action="store_true",
        help="Process all unprocessed subjects.",
    )
    parser.add_argument(
        "--session", default=DEFAULT_SESSION_ID,
        help=f"Session ID (default: {DEFAULT_SESSION_ID}).",
    )
    parser.add_argument(
        "--run", default=DEFAULT_RUN_ID,
        help=f"Run ID (default: {DEFAULT_RUN_ID}).",
    )
    parser.add_argument(
        "--slurm", action="store_true",
        help="Submit as Slurm job instead of running locally.",
    )
    parser.add_argument(
        "--derivative-base", default="sync_loss",
        choices=["sync_loss", "postproc"],
        help="Derivative chain to use. Default: sync_loss.",
    )
    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()

    if not args.subject and not args.process_all:
        parser.error("Either --subject or --all must be specified.")

    from q1k.io import get_autorej_path, get_segment_path

    pp = Path(args.project_path)
    seg_path = get_segment_path(pp.parent, args.derivative_base)
    ar_path = get_autorej_path(pp.parent, args.derivative_base)

    epoch_dir = seg_path / "epoch_fif_files" / args.task
    out_dir = ar_path / "epoch_fif_files" / args.task

    if args.slurm:
        from q1k.slurm import submit_slurm_job

        slurm_script = Path(__file__).parent.parent / "slurm" / "autorej_job.sh"
        pattern = str(epoch_dir / f"*task-{args.task}*_epo.fif")
        files = glob.glob(pattern)

        # Filter for specific subject if provided
        if args.subject:
            files = [f for f in files if f"sub-{args.subject}" in f]

        for f in files:
            fname = os.path.basename(f)
            job_name = fname.replace(".fif", "_autorej")
            submit_slurm_job(
                slurm_script, job_name, "slurm_output", args.task,
                f, str(out_dir) + "/",
            )
    else:
        from q1k.autorej.pipeline import run_autoreject

        pattern = str(epoch_dir / f"*task-{args.task}*_epo.fif")
        files = glob.glob(pattern)

        if args.subject:
            files = [f for f in files if f"sub-{args.subject}" in f]

        if not files:
            print(f"No epoch files found for task {args.task}")
            return

        error_subjects = []
        for f in files:
            fname = os.path.basename(f)
            print(f"Processing {fname}...")
            try:
                run_autoreject(f, out_dir)
            except Exception as e:
                error_subjects.append(fname)
                print(f"Error processing {fname}: {e}")

        if error_subjects:
            print(f"Files with errors: {error_subjects}")


if __name__ == "__main__":
    main()
