"""CLI for Stage 1: BIDS conversion and report generation."""

import argparse
import glob
import os
import re
import subprocess
from pathlib import Path

from q1k.config import (
    DEFAULT_RUN_ID,
    DEFAULT_SESSION_ID,
    VALID_TASKS,
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
    # RSRio files. RSRio uses the full "RSRMo" string as-is.
    if task == "RS":
        task_id_in_search = "RS_"
    else:
        task_id_in_search = task

    # Copy template and inject parameters
 # Copy template and inject parameters
    template_content = notebook_template.read_text()
    
    # Find the parameters function and replace its content
    lines = template_content.split('\n')
    in_params = False
    param_start = None
    param_end = None
    
    for i, line in enumerate(lines):
        if 'def parameters():' in line:
            in_params = True
            param_start = i + 1
        elif in_params and 'return' in line:
            param_end = i + 1
            break
    
    if param_start is not None and param_end is not None:
        # Get the indentation from the first parameter line
        # (should be 8 spaces for marimo)
        first_param_line = lines[param_start] if param_start < len(lines) else ""
        indent = first_param_line[:len(first_param_line) - len(first_param_line.lstrip())]
        if not indent:
            indent = "    "  # Default to 4 spaces if can't detect
        
        # Create parameter block with consistent indentation
        new_params = [
            f'{indent}# __Q1K_PARAMETERS__',
            f'{indent}# The above comment is replaced by the CLI with actual values.',
            f'{indent}# For interactive use, set your parameters here:',
            f'{indent}project_path = "{project_path}"',
            f'{indent}task_id_in = "{task_id_in_search}"',
            f'{indent}task_id_in_et = "{task_id_in_search}"',
            f'{indent}task_id_out = "{task}"',
            f'{indent}subject_id = "{subject_id}"',
            f'{indent}session_id = "{session_id}"',
            f'{indent}run_id = "{run_id}"',
            f'{indent}site_code = "{site}"',
            f'{indent}return (project_path, task_id_in, task_id_in_et, task_id_out,',
            f'{indent}        subject_id, session_id, run_id, site_code)',
        ]
        
        # Replace the lines
        lines[param_start:param_end] = new_params
        output_content = '\n'.join(lines)
    else:
        # Fallback
        output_content = template_content
    
    out_notebook.write_text(output_content)
#.....

    template_content = notebook_template.read_text()
    # Create properly indented parameter block (8 spaces for marimo cells)
    indent = "    "  # 4 spaces for the function body
    param_block = (
        f'{indent}project_path = "{project_path}"\n'
        f'{indent}task_id_in = "{task_id_in_search}"\n'
        f'{indent}task_id_in_et = "{task_id_in_search}"\n'
        f'{indent}task_id_out = "{task}"\n'
        f'{indent}subject_id = "{subject_id}"\n'
        f'{indent}session_id = "{session_id}"\n'
        f'{indent}run_id = "{run_id}"\n'
        f'{indent}site_code = "{site}"\n'
    )
    lines = template_content.split('\n')
    in_parameters = False
    param_lines_start = None
    param_lines_end = None
    for i, line in enumerate(lines):
        if '@app.cell' in line and 'parameters' in line:
            in_parameters = True
        elif in_parameters and 'def parameters():' in line:
            # The next line after this is the first parameter
            param_lines_start = i + 1
        elif in_parameters and param_lines_start and 'return' in line:
            param_lines_end = i
            break
    if param_lines_start and param_lines_end:
        # Replace the parameter block
        lines[param_lines_start:param_lines_end] = param_block.split('\n')
        output_content = '\n'.join(lines)
    else:
        # Fallback to old method if parsing fails
        output_content = template_content.replace(
            "# __Q1K_PARAMETERS__", param_block
        )
    out_notebook.write_text(output_content)



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
