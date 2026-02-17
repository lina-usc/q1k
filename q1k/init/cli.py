"""CLI for Stage 1: BIDS conversion and report generation."""

import argparse
import glob
import os
import re
import shutil
import subprocess
from pathlib import Path

from q1k.config import (
    DEFAULT_RUN_ID, DEFAULT_SESSION_ID, NO_DIN_OFFSET_TASKS, VALID_TASKS,
)


def create_parser():
    parser = argparse.ArgumentParser(
        prog="q1k-init",
        description=(
            "Stage 1: Convert raw EEG/ET data to BIDS format and "
            "generate per-subject report notebooks."
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
        help="Single subject ID to process. If omitted, use --all.",
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
        "--site", default="HSJ", choices=["HSJ", "MHC", "NIM"],
        help="Site code (default: HSJ).",
    )
    return parser


def run_init(project_path, task, subject_id, session_id, run_id, site):
    """Run the BIDS initialization for a single subject.

    This generates a per-subject marimo notebook with the processing
    results, and exports it as HTML for quick review.
    """
    from q1k.io import get_report_path

    report_dir = get_report_path("init", task, root=Path(project_path).parent)
    report_dir.mkdir(parents=True, exist_ok=True)

    notebook_template = Path(__file__).parent.parent / "notebooks" / "init_report.py"
    out_notebook = report_dir / f"{subject_id}_{task}_init.py"

    # For RS, append "_" to the search pattern to avoid matching
    # RSRio files. RSRio uses the full "RSRio" string as-is.
    if task == "RS":
        task_id_in_search = "RS_"
    else:
        task_id_in_search = task

    # Copy template and inject parameters
    template_content = notebook_template.read_text()
    param_block = (
        f'project_path = "{project_path}"\n'
        f'task_id_in = "{task_id_in_search}"\n'
        f'task_id_in_et = "{task_id_in_search}"\n'
        f'task_id_out = "{task}"\n'
        f'subject_id = "{subject_id}"\n'
        f'session_id = "{session_id}"\n'
        f'run_id = "{run_id}"\n'
        f'site_code = "{site}"\n'
    )
    # Replace the placeholder parameter block
    output_content = template_content.replace(
        "# __Q1K_PARAMETERS__", param_block
    )
    out_notebook.write_text(output_content)

    # Export HTML report
    out_html = report_dir / f"{subject_id}_{task}_init.html"
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
        run_init(
            args.project_path, args.task, args.subject,
            args.session, args.run, args.site,
        )
    else:
        # Find all unprocessed subjects
        sourcedata = os.path.join(args.project_path, "sourcedata", "eeg")
        pattern = os.path.join(sourcedata, "Q1K*", f"*{args.task}*.mff")
        files = glob.glob(pattern)

        if not files:
            print(f"No source files found for task {args.task}")
            return

        for f in files:
            # Extract subject ID from path
            match = re.search(r"Q1K_\w+_(\d+_\w+)", os.path.basename(f))
            if match:
                subject_id = match.group(1)
                print(f"Processing {subject_id}...")
                try:
                    run_init(
                        args.project_path, args.task, subject_id,
                        args.session, args.run, args.site,
                    )
                except Exception as e:
                    print(f"Error processing {subject_id}: {e}")


if __name__ == "__main__":
    main()
