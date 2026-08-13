"""CLI for Stage 1: BIDS conversion and report generation."""

import argparse
import glob
import os
import subprocess
from pathlib import Path

from q1k.init.tools import VALID_TASKS

#from tools import VALID_TASKS


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
        "--session", default="01",
        help="Session ID (default: 01).",
    )
    parser.add_argument(
        "--run", default="1",
        help="Run ID (default: 1).",
    )
    parser.add_argument(
        "--site", default="HSJ", choices=["HSJ", "MHC", "NIM"],
        help="Site code (default: HSJ).",
    )
    return parser


'''
def compute_subject_id_out(subject_id):
    """Convert Q1K subject ID to BIDS-compatible subject ID."""
    parts = re.split(r'_{1,2}', subject_id)

    subject_number = parts[0]
    subject_relation = parts[1] if len(parts) > 1 else ""

    # Handle MHC format: 1525-XXXX
    if subject_number.startswith(("1025-", "1525-")):
        family_code = subject_number[5:9]
    elif subject_number.startswith(("100", "200")):
        family_code = subject_number[3:].zfill(4)
    else:
        family_code = subject_number

    return family_code + subject_relation'''



def compute_subject_id_out(subject_id):
    """Convert Q1K subject ID to BIDS-compatible subject ID.

    Examples:
    - Q1K_HSJ_10046_F1 -> HSJ10046F1
    - Q1K_MHC_20034_P -> MHC20034P
    - Q1K_HSJ_1525_20034_S2 -> HSJ152520034S2
    """
    parts = subject_id.split("_")

    # Remove the first part ('Q1K') and join the rest without underscores
    if parts[0].upper() == "Q1K":
        return "".join(parts[1:])
    else:
        return subject_id

'''def compute_subject_id_out(subject_id):
    """Convert Q1K subject ID to BIDS-compatible subject ID.

    Examples:
    - Q1K_HSJ_1525-10046_F1 → HSJ_1525-10046_F1
    - Q1K_MHC_20034_P → MHC_20034_P
    - Q1K_HSJ_1525_20034_S2 → HSJ_1525_20034_S2
    - Q1K_HSJ_10046_F1 → HSJ_10046_F1
    """
    parts = subject_id.split("_")

    # Remove the first part ('Q1K') and keep everything else
    if parts[0].upper() == "Q1K":
        subject_id_out = "_".join(parts[1:])
    else:
        subject_id_out = subject_id

    return subject_id_out'''



def run_init(project_path, task, subject_id, session_id, run_id, site):
    """Run the BIDS initialization for a single subject.

    This generates a per-subject marimo notebook with the processing
    results, and exports it as HTML for quick review.
    """
    subject_id_out = compute_subject_id_out(subject_id)
    #report_dir = Path(project_path) / "reports" / "init" / task
    #report_dir.mkdir(parents=True, exist_ok=True

    report_dir = Path(project_path) / "derivatives" / "init" / task / f"sub-{subject_id_out}" / f"ses-{session_id}" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    notebook_template = Path(__file__).parent.parent / "notebooks" / "init_report.py"
    out_notebook = report_dir / f"sub-{subject_id_out}_ses-{session_id}_task-{task}_run-{run_id}_init.py"

    subject_id_out = compute_subject_id_out(subject_id)
    if task == "RS":

        task_id_in_search = "RS_"
    else:
        task_id_in_search = task

    # Copy template and inject parameters with proper indentation
    template_content = notebook_template.read_text()
    indent = "    "  # 4 spaces
    param_block = (
        f'{indent}project_path = "{project_path}"\n'
        f'{indent}task_id_in = "{task_id_in_search}"\n'
        f'{indent}task_id_in_et = "{task_id_in_search}"\n'
        f'{indent}task_id_out = "{task}"\n'
        f'{indent}subject_id = "{subject_id}"\n'
        f'{indent}subject_id_out = "{subject_id_out}"\n'
        f'{indent}session_id = "{session_id}"\n'
        f'{indent}run_id = "{run_id}"\n'
        f'{indent}site_code = "{site}"'
    )
    # Replace the placeholder parameter block
    lines = template_content.split('\n')
    in_params = False
    param_start = None
    param_end = None

    for i, line in enumerate(lines):
        if 'def parameters():' in line:
            in_params = True
            param_start = i + 1
        elif in_params and 'return' in line:
            param_end = i
            break

    if param_start and param_end:
        # Replace the parameter lines
        lines[param_start:param_end] = param_block.split('\n')
        output_content = '\n'.join(lines)
    else:
        # Fallback
        output_content = template_content.replace("# __Q1K_PARAMETERS__", param_block)
    out_notebook.write_text(output_content)
    # Export HTML report
    out_html = report_dir / f"sub-{subject_id_out}_ses-{session_id}_task-{task}_run-{run_id}_report.html"
    try:
        result = subprocess.run(
            ["marimo", "export", "html", str(out_notebook),
             "-o", str(out_html)],
            check=True,timeout =1800
        )
        if result.returncode == 0:
                print(f"Report saved: {out_html}")
        else:
                print(f"Warning: Notebook execution returned code {result.returncode}")
                if result.stderr:
                    print(f"stderr: {result.stderr[:500]}")
                print(f"Marimo notebook saved: {out_notebook}")
    except subprocess.TimeoutExpired:
        print("Warning: Notebook execution timed out after 10 minutes")
        print(f"Marimo notebook saved: {out_notebook}")
    except FileNotFoundError:
        print("Warning: 'marimo' command not found - install with: pip install marimo")
        print(f"Marimo notebook saved but not executed: {out_notebook}")
    except Exception as e:
        print(f"Warning: Unexpected error executing notebook: {e}")
        print(f"Marimo notebook saved: {out_notebook}")

    # Run the notebook to execute the cells
    '''try:
        print(f"Running notebook: {out_notebook}")
        subprocess.run(
            ["marimo", "run", str(out_notebook)],
            check=True,
            capture_output=True,
            text=True
        )
        print("Notebook executed successfully")
    except subprocess.CalledProcessError as e:
        print(f"Error running notebook: {e.stderr}")
    except FileNotFoundError:
        print("Warning: 'marimo' command not found - install with: pip install marimo")'''
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
        sourcedata = os.path.join(args.project_path, "sourcedata",args.site, "eeg")
        #pattern = os.path.join(sourcedata, "Q1K*", f"*{args.task}*.mff")
        #files = glob.glob(pattern)

        files = []
        for subject_dir in glob.glob(os.path.join(sourcedata, "Q1K*")):
            mff_dirs = glob.glob(os.path.join(subject_dir, f"*{args.task}*.mff"))
            files.extend(mff_dirs)
        if not files:
            print(f"No source files found for task {args.task}")
            return

        for f in files:

            subject_id = os.path.basename(os.path.dirname(f))



            print(f"Processing {subject_id}...")
            try:
                run_init( args.project_path, args.task, subject_id,
                    args.session, args.run, args.site,)
            except Exception as e:
                print(f"Error processing {subject_id}: {e}")
                print(f"Full error: {str(e)}")



if __name__ == "__main__":
    main()
