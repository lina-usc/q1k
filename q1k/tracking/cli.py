"""CLI for pipeline tracking and data loss reporting."""

import argparse

from q1k.config import VALID_TASKS


def create_parser():
    parser = argparse.ArgumentParser(
        prog="q1k-tracking",
        description=(
            "Pipeline tracking: cross-reference REDCap demographics "
            "with pipeline stage outputs to monitor data flow and "
            "generate data loss reports."
        ),
    )
    parser.add_argument(
        "--project-path", required=True,
        help="Path to the project experimental directory.",
    )
    parser.add_argument(
        "--redcap-dir", required=True,
        help=(
            "Path to REDCap export directory (containing "
            "Demographics and TaskCompletion CSVs)."
        ),
    )
    parser.add_argument(
        "--task", default=None,
        choices=[t for t in VALID_TASKS if t != "RSRio"],
        help="Specific task to track (default: all tasks).",
    )
    parser.add_argument(
        "--sharepoint", default=None,
        help=(
            "Path to SharePoint Excel file for comparison mode. "
            "When provided, generates a data loss comparison report."
        ),
    )
    parser.add_argument(
        "--mni-upload-date", default=None,
        help=(
            "MNI site upload cutoff date (YYYY-MM-DD). Subjects "
            "with visits after this date are classified as "
            "'Not uploaded'."
        ),
    )
    parser.add_argument(
        "--hsj-upload-date", default=None,
        help=(
            "HSJ site upload cutoff date (YYYY-MM-DD). Subjects "
            "with visits after this date are classified as "
            "'Not uploaded'."
        ),
    )
    parser.add_argument(
        "--output-dir", default=None,
        help=(
            "Output directory for tracking CSVs and reports. "
            "Default: tracking/ sibling of project-path."
        ),
    )
    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()

    from q1k.tracking.tools import (
        generate_all_tracking_csvs,
        generate_data_loss_excel,
        generate_tracking_csv,
    )

    if args.sharepoint:
        # Comparison mode: generate Excel report
        print("Generating data loss comparison report...")
        out_path = generate_data_loss_excel(
            project_path=args.project_path,
            redcap_dir=args.redcap_dir,
            sharepoint_path=args.sharepoint,
            output_dir=args.output_dir,
            mni_upload_date=args.mni_upload_date,
            hsj_upload_date=args.hsj_upload_date,
        )
        print(f"Data loss report: {out_path}")

    elif args.task:
        # Single task tracking CSV
        print(f"Generating tracking CSV for {args.task}...")
        out_path = generate_tracking_csv(
            project_path=args.project_path,
            task=args.task,
            redcap_dir=args.redcap_dir,
            output_dir=args.output_dir,
        )
        print(f"Tracking CSV: {out_path}")

    else:
        # All tasks tracking CSVs
        print("Generating tracking CSVs for all tasks...")
        results = generate_all_tracking_csvs(
            project_path=args.project_path,
            redcap_dir=args.redcap_dir,
            output_dir=args.output_dir,
        )
        for task, path in results.items():
            if path:
                print(f"  {task}: {path}")
            else:
                print(f"  {task}: FAILED")


if __name__ == "__main__":
    main()
