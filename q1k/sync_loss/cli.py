"""CLI for Stage 3: EEG/ET synchronization + PyLossless cleaning."""

import argparse
import os
import subprocess
from pathlib import Path

from q1k.config import DEFAULT_RUN_ID, DEFAULT_SESSION_ID, VALID_TASKS

# Tasks that require eye-tracking synchronization
ET_SYNC_TASKS = {"VEP", "GO", "NSP", "PLR", "VS"}


def create_parser():
    parser = argparse.ArgumentParser(
        prog="q1k-sync-loss",
        description=(
            "Stage 3: Synchronize EEG/ET data and apply PyLossless "
            "cleaning. Generates per-subject report notebooks."
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
    return parser


def run_sync_loss(project_path, task, subject_id, session_id, run_id):
    """Run sync_loss processing for a single subject.

    Generates a per-subject marimo notebook as a log.
    """
    from q1k.io import get_report_path

    report_dir = get_report_path(
        "sync_loss", task, root=Path(project_path).parent
    )
    report_dir.mkdir(parents=True, exist_ok=True)

    notebook_template = (
        Path(__file__).parent.parent / "notebooks" / "sync_loss_report.py"
    )
    out_notebook = report_dir / f"{subject_id}_{task}_sync_loss.py"

    et_sync = task in ET_SYNC_TASKS

    # Copy template and inject parameters
    template_content = notebook_template.read_text()
    param_block = (
        f'project_path = "{project_path}"\n'
        f'task_id = "{task}"\n'
        f'subject_id = "{subject_id}"\n'
        f'session_id = "{session_id}"\n'
        f'run_id = "{run_id}"\n'
        f"et_sync = {et_sync}\n"
    )
    output_content = template_content.replace(
        "# __Q1K_PARAMETERS__", param_block
    )
    out_notebook.write_text(output_content)

    # Export HTML report
    out_html = report_dir / f"{subject_id}_{task}_sync_loss.html"
    try:
        subprocess.run(
            ["marimo", "export", "html", str(out_notebook),
             "-o", str(out_html)],
            check=True,
        )
        print(f"Report saved: {out_html}")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Warning: Could not export HTML report: {e}")
        print(f"Marimo notebook saved: {out_notebook}")

    return out_notebook


def main():
    parser = create_parser()
    args = parser.parse_args()

    if not args.subject and not args.process_all:
        parser.error("Either --subject or --all must be specified.")

    if args.subject:
        run_sync_loss(
            args.project_path, args.task, args.subject,
            args.session, args.run,
        )
    else:
        from q1k.bids import extract_bids_info
        from q1k.slurm import find_unprocessed

        pyll_path = os.path.join(
            args.project_path, "derivatives", "pylossless"
        )
        sync_loss_path = os.path.join(pyll_path, "derivatives", "sync_loss")

        input_pattern = os.path.join(
            pyll_path, "**", "eeg", f"*task-{args.task}*_eeg.edf"
        )
        output_pattern = os.path.join(
            sync_loss_path, "**", "eeg", f"*task-{args.task}*_eeg.edf"
        )
        unprocessed = find_unprocessed(input_pattern, output_pattern)

        error_subjects = []
        for f in unprocessed:
            basename = os.path.basename(f)
            subject_id, session_id, task_id, run_id = extract_bids_info(
                basename
            )
            print(f"Processing {subject_id}...")
            try:
                run_sync_loss(
                    args.project_path, args.task, subject_id,
                    session_id, run_id,
                )
            except Exception as e:
                error_subjects.append(subject_id)
                print(f"Error processing {subject_id}: {e}")

        if error_subjects:
            print(f"Subjects with errors: {error_subjects}")


if __name__ == "__main__":
    main()
