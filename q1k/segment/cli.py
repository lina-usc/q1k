"""CLI for Stage 4: Task-specific epoch segmentation."""

import argparse
import os
import subprocess
from pathlib import Path

from q1k.config import DEFAULT_RUN_ID, DEFAULT_SESSION_ID, VALID_TASKS


def create_parser():
    parser = argparse.ArgumentParser(
        prog="q1k-segment",
        description=(
            "Stage 4: Segment preprocessed data into task-specific epochs. "
            "Generates per-subject report notebooks."
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
        "--derivative-base", default="sync_loss",
        choices=["sync_loss", "postproc"],
        help=(
            "Which derivative chain to use for input data. "
            "'sync_loss' (default) for the standard pipeline, "
            "'postproc' for the legacy postprocessing path."
        ),
    )
    return parser


def run_segment(project_path, task, subject_id, session_id, run_id,
                derivative_base="sync_loss"):
    """Run segmentation for a single subject.

    Generates a per-subject marimo notebook as a log.
    """
    from q1k.io import get_report_path

    report_dir = get_report_path(
        "segment", task, root=Path(project_path).parent
    )
    report_dir.mkdir(parents=True, exist_ok=True)

    # Pick the appropriate template based on task
    # RSRio uses the same RS template (with auto-detection inside)
    template_task = "RS" if task == "RSRio" else task
    notebook_name = f"segment_{template_task}.py"
    notebook_template = (
        Path(__file__).parent.parent / "notebooks" / notebook_name
    )

    # Fall back to a generic template if task-specific one doesn't exist
    if not notebook_template.exists():
        print(f"Warning: No template for {task}, using generic segmentation")
        return None

    out_notebook = report_dir / f"{subject_id}_{task}_segment.py"

    # Copy template and inject parameters
    template_content = notebook_template.read_text()
    param_block = (
        f'project_path = "{project_path}"\n'
        f'task_id = "{task}"\n'
        f'subject_id = "{subject_id}"\n'
        f'session_id = "{session_id}"\n'
        f'run_id = "{run_id}"\n'
        f'derivative_base = "{derivative_base}"\n'
    )
    output_content = template_content.replace(
        "# __Q1K_PARAMETERS__", param_block
    )
    out_notebook.write_text(output_content)

    # Export HTML report
    out_html = report_dir / f"{subject_id}_{task}_segment.html"
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
        run_segment(
            args.project_path, args.task, args.subject,
            args.session, args.run, args.derivative_base,
        )
    else:
        from q1k.bids import extract_bids_info
        from q1k.slurm import find_unprocessed

        pp = Path(args.project_path)
        if args.derivative_base == "sync_loss":
            input_base = (
                pp / "derivatives" / "pylossless" / "derivatives" / "sync_loss"
            )
        else:
            input_base = pp / "derivatives" / "pylossless" / "derivatives" / "postproc"

        seg_base = input_base / "derivatives" / "segment"

        # For RS, use "RS_" glob to avoid matching RSRio files
        task_glob = f"{args.task}_" if args.task == "RS" else args.task
        input_pattern = str(
            input_base / "**" / "eeg" / f"*task-{task_glob}*_eeg.edf"
        )
        output_pattern = str(
            seg_base / "epoch_fif_files" / args.task / f"*task-{args.task}*_epo.fif"
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
                run_segment(
                    args.project_path, args.task, subject_id,
                    session_id, run_id, args.derivative_base,
                )
            except Exception as e:
                error_subjects.append(subject_id)
                print(f"Error processing {subject_id}: {e}")

        if error_subjects:
            print(f"Subjects with errors: {error_subjects}")


if __name__ == "__main__":
    main()
