"""CLI for Stage 5: AutoReject epoch cleaning."""

import argparse
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
        help="Single subject ID to process(e.g., HSJ10046F1).",
    )
    parser.add_argument(
        "--all", dest="process_all", action="store_true",
        help="Process all unprocessed epoch files for the task.",
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
        "--derivative-base", default="segment",
        choices=["segment"],
        help="Input derivative stage. Currently only 'segment' is supported.",
    )
    parser.add_argument(
        "--slurm", action="store_true",
        help="Submit Slurm jobs instead of running locally.",
    )

    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()

    if not args.subject and not args.process_all:
        parser.error("Either --subject or --all must be specified.")

    from q1k.autorej.pipeline import run_autoreject

    project_path = Path(args.project_path)
    out_path = (
        project_path
        / "derivatives"
        / "autoreject"
        / "epoch_fif_files"
        / args.task
    )

    def input_file_for_subject(subject):
        return (
            project_path
            / "derivatives"
            / "segment"
            / "epoch_fif_files"
            / args.task
            / f"sub-{subject}_ses-{args.session}_task-{args.task}_run-{args.run}_eeg_epo.fif"
        )

    def run_one(input_file):
        print(f"Processing: {input_file}")
        print(f"Output to: {out_path}")
        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")
        return run_autoreject(str(input_file), str(out_path))

    if args.slurm:
        from q1k.slurm import find_unprocessed, submit_slurm_job

        slurm_script = Path(__file__).parent.parent / "slurm" / "autorej_job.sh"
        if args.subject:
            input_files = [input_file_for_subject(args.subject)]
        else:
            input_pattern = str(
                project_path
                / "derivatives"
                / "segment"
                / "epoch_fif_files"
                / args.task
                / f"*task-{args.task}*_epo.fif"
            )
            output_pattern = str(out_path / f"*task-{args.task}*_epo.fif")
            input_files = [Path(f) for f in find_unprocessed(input_pattern, output_pattern)]

        for input_file in input_files:
            subject = input_file.name.split("_", 1)[0].removeprefix("sub-")
            job_name = f"ar_{subject}_{args.task}"
            submit_slurm_job(
                slurm_script,
                job_name,
                "slurm_output",
                args.task,
                input_file,
                out_path,
            )
        return

    if args.subject:
        run_one(input_file_for_subject(args.subject))
        return

    input_pattern = (
        project_path
        / "derivatives"
        / "segment"
        / "epoch_fif_files"
        / args.task
    ).glob(f"*task-{args.task}*_epo.fif")
    for input_file in sorted(input_pattern):
        output_file = out_path / input_file.name
        if output_file.exists():
            continue
        run_one(input_file)


if __name__ == "__main__":
    main()
