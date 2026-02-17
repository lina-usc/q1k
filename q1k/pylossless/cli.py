"""CLI for Stage 2: PyLossless artifact rejection."""

import argparse
import os
from pathlib import Path

from q1k.config import DEFAULT_RUN_ID, DEFAULT_SESSION_ID, VALID_TASKS


def create_parser():
    parser = argparse.ArgumentParser(
        prog="q1k-pylossless",
        description=(
            "Stage 2: Run PyLossless artifact rejection pipeline. "
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
    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()

    if not args.subject and not args.process_all:
        parser.error("Either --subject or --all must be specified.")

    project_path = args.project_path
    pyll_path = os.path.join(project_path, "derivatives", "pylossless")

    if args.slurm:
        from q1k.slurm import submit_slurm_job, find_unprocessed
        from q1k.bids import extract_bids_info

        slurm_script = Path(__file__).parent.parent / "slurm" / "pylossless_job.sh"

        if args.subject:
            job_name = f"pyll_{args.subject}_{args.task}"
            submit_slurm_job(
                slurm_script, job_name, "slurm_output", args.task,
                project_path, args.subject, args.session,
                args.task, args.run, pyll_path,
            )
        else:
            input_pattern = os.path.join(
                project_path, "**", "eeg", f"*task-{args.task}*_eeg.edf"
            )
            output_pattern = os.path.join(
                pyll_path, "**", "eeg", f"*task-{args.task}*_eeg.edf"
            )
            unprocessed = find_unprocessed(input_pattern, output_pattern)
            for f in unprocessed:
                basename = os.path.basename(f)
                subject_id, session_id, task_id, run_id = extract_bids_info(basename)
                job_name = basename.replace(".edf", "_pyll")
                submit_slurm_job(
                    slurm_script, job_name, "slurm_output", task_id,
                    project_path, subject_id, session_id,
                    task_id, run_id, pyll_path,
                )
    else:
        from q1k.pylossless.pipeline import run_pylossless

        if args.subject:
            run_pylossless(
                project_path, args.subject, args.session,
                args.task, args.run, pyll_path,
            )
        else:
            from q1k.slurm import find_unprocessed
            from q1k.bids import extract_bids_info

            input_pattern = os.path.join(
                project_path, "**", "eeg", f"*task-{args.task}*_eeg.edf"
            )
            output_pattern = os.path.join(
                pyll_path, "**", "eeg", f"*task-{args.task}*_eeg.edf"
            )
            unprocessed = find_unprocessed(input_pattern, output_pattern)
            for f in unprocessed:
                basename = os.path.basename(f)
                subject_id, session_id, task_id, run_id = extract_bids_info(basename)
                print(f"Processing {subject_id}...")
                try:
                    run_pylossless(
                        project_path, subject_id, session_id,
                        task_id, run_id, pyll_path,
                    )
                except Exception as e:
                    print(f"Error processing {subject_id}: {e}")


if __name__ == "__main__":
    main()
