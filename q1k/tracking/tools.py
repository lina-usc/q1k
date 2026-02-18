"""Pipeline tracking and data loss reporting utilities.

Provides functions to:
- Load and merge REDCap demographic and task-completion exports
- Scan the BIDS filesystem for outputs at each pipeline stage
- Merge demographics with pipeline stage data to track data flow
- Compare automated tracking with manually curated SharePoint sheets
- Generate per-task tracking CSVs and data loss Excel reports
"""

import glob
import os
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from q1k.bids import q1k_to_bids
from q1k.config import (
    PIPELINE_STAGES,
    REDCAP_DEMO_COLUMNS,
    REDCAP_FAIL_COLUMNS,
    REDCAP_SESSION_COLUMNS,
    REDCAP_TASK_COLUMNS,
    VALID_TASKS,
)
from q1k.io import get_project_site_path


# ── REDCap data loading ─────────────────────────────────────────────


def _find_csv(directory, keyword):
    """Find a CSV file containing *keyword* in its name."""
    for f in Path(directory).glob("*.csv"):
        if keyword in f.name:
            return f
    raise FileNotFoundError(
        f"No CSV containing '{keyword}' found in {directory}"
    )


def load_redcap_task_completion(redcap_dir):
    """Load the REDCap EEG session / task-completion CSV.

    Parameters
    ----------
    redcap_dir : str or Path
        Directory containing the REDCap CSV exports.

    Returns
    -------
    pd.DataFrame
        DataFrame with standardised columns: ``record_id``,
        ``q1k_ID``, ``et_id``, ``visit_date``, per-task completion
        flags, per-task failure reasons, and session metadata.
    """
    csv_path = _find_csv(redcap_dir, "TaskCompletion")
    df = pd.read_csv(csv_path)

    # Rename columns
    rename = {
        "q1k_proband_id_1": "proband_id",
        "q1k_relative_idgenerated_1": "relative_id",
    }
    rename.update(REDCAP_SESSION_COLUMNS)
    rename.update(REDCAP_TASK_COLUMNS)
    rename.update(REDCAP_FAIL_COLUMNS)
    df = df.rename(columns=rename)

    # Build q1k_ID from proband or relative ID
    if "proband_id" in df.columns and "relative_id" in df.columns:
        df["q1k_ID"] = df["proband_id"].combine_first(
            df["relative_id"]
        )
    elif "proband_id" in df.columns:
        df["q1k_ID"] = df["proband_id"]

    # Merge TO and MMN (same paradigm)
    if "MMN" in df.columns and "TO" in df.columns:
        df["TO"] = df["TO"].combine_first(df["MMN"])
        df = df.drop(columns=["MMN"], errors="ignore")

    # Collapse duplicate record_ids (take first non-null per group)
    df = df.groupby("record_id", as_index=False).first()

    # Filter to attempted EEG sessions
    if "eeg_attempt" in df.columns:
        df = df.loc[df["eeg_attempt"] == 1].copy()

    # Derive site from q1k_ID
    df["site"] = df["q1k_ID"].apply(_infer_site)

    # Derive group (proband / parent / sibling) from ID suffix
    df["group"] = df["q1k_ID"].apply(_infer_group)

    # Generate BIDS ID
    df["bids_id"] = df["q1k_ID"].apply(_safe_q1k_to_bids)

    return df


def load_redcap_demographics(redcap_dir):
    """Load the REDCap demographics CSV.

    Parameters
    ----------
    redcap_dir : str or Path
        Directory containing the REDCap CSV exports.

    Returns
    -------
    pd.DataFrame
        DataFrame with standardised demographic columns.
    """
    csv_path = _find_csv(redcap_dir, "Demographics")
    df = pd.read_csv(csv_path)

    rename = {
        "q1k_proband_id_1": "proband_id",
        "q1k_relative_idgenerated_1": "relative_id",
    }
    rename.update(REDCAP_DEMO_COLUMNS)
    df = df.rename(columns=rename)

    # Build q1k_ID
    if "proband_id" in df.columns and "relative_id" in df.columns:
        df["q1k_ID"] = df["proband_id"].combine_first(
            df["relative_id"]
        )
    elif "proband_id" in df.columns:
        df["q1k_ID"] = df["proband_id"]

    # Collapse duplicates
    df = df.groupby("record_id", as_index=False).first()

    # Decode sex: 1=female, 2=male, 0=other
    if "sex" in df.columns:
        df["sex"] = df["sex"].map({1: "female", 2: "male", 0: "other"})

    # Decode phase 3 consent: 2=Complete
    if "phase_3_consented" in df.columns:
        df["phase_3_consented"] = (
            df["phase_3_consented"] == 2
        ).map({True: "Yes", False: "No"})

    return df


def load_redcap(redcap_dir):
    """Load and merge REDCap demographics + task completion data.

    Parameters
    ----------
    redcap_dir : str or Path
        Directory containing both REDCap CSV exports.

    Returns
    -------
    pd.DataFrame
        Merged DataFrame with demographics and task completion data.
    """
    task_df = load_redcap_task_completion(redcap_dir)
    demo_df = load_redcap_demographics(redcap_dir)

    # Merge on record_id
    merged = task_df.merge(
        demo_df[["record_id", "sex", "ndd", "asd", "adhd",
                 "phase_3_consented"]].drop_duplicates("record_id"),
        on="record_id",
        how="left",
        suffixes=("", "_demo"),
    )

    # Prefer demographics sex over session sex
    if "sex_demo" in merged.columns:
        merged["sex"] = merged["sex_demo"].combine_first(
            merged["sex"]
        )
        merged = merged.drop(columns=["sex_demo"])

    return merged


# ── Pipeline stage scanning ──────────────────────────────────────────


def _extract_subject_from_path(filepath):
    """Extract BIDS subject ID from a file path."""
    match = re.search(r"sub-([^_/\\]+)", filepath)
    if match:
        return match.group(1)
    return None


def _extract_subject_from_mff(filepath):
    """Extract subject ID from a raw .mff source path.

    Source paths look like: ``Q1K_HSJ_10043_P/.../*.mff``
    """
    basename = os.path.basename(filepath)
    # Try to find Q1K ID in the path
    match = re.search(r"Q1K_\w+_\d+[_-]\w+", filepath)
    if match:
        return _safe_q1k_to_bids(match.group(0))
    return basename


def scan_pipeline_stages(project_path, task):
    """Scan the BIDS filesystem for outputs at each pipeline stage.

    For a given task, checks which subjects have outputs at each
    stage of the processing pipeline.

    Parameters
    ----------
    project_path : str or Path
        Path to the project experimental directory.
    task : str
        Task code (e.g., ``"RS"``, ``"VEP"``).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``subject_id`` and one boolean
        column per stage in :data:`~q1k.config.PIPELINE_STAGES`.
    """
    pp = Path(project_path)

    # Use RS_ glob for RS to avoid matching RSRio
    task_glob = f"{task}_" if task == "RS" else task

    # Define glob patterns for each stage
    stage_patterns = {
        "EEG Raw Files": str(
            pp / "**" / "sourcedata" / "eeg"
            / "Q*" / f"Q1K*{task_glob}*.mff"
        ),
        "BIDS": str(
            pp / "sub*" / "*" / "eeg" / f"*task-{task_glob}*_eeg.edf"
        ),
        "Pylossless": str(
            pp / "derivatives" / "pylossless"
            / "sub*" / "*" / "eeg" / f"*task-{task_glob}*_eeg.edf"
        ),
        "ET_sync_loss": str(
            pp / "derivatives" / "pylossless"
            / "derivatives" / "sync_loss"
            / "sub*" / "*" / "eeg" / f"*task-{task_glob}*_eeg.edf"
        ),
        "Segmentation": str(
            pp / "derivatives" / "pylossless"
            / "derivatives" / "sync_loss"
            / "derivatives" / "segment"
            / "epoch_fif_files" / task / "sub*"
        ),
        "Autoreject": str(
            pp / "derivatives" / "pylossless"
            / "derivatives" / "sync_loss"
            / "derivatives" / "segment"
            / "derivatives" / "autorej"
            / "epoch_fif_files" / task / "sub*"
        ),
    }

    # Collect subjects per stage
    all_subjects = set()
    stage_subjects = {}

    for stage, pattern in stage_patterns.items():
        files = glob.glob(pattern, recursive=True)
        subjects = set()
        for f in files:
            if stage == "EEG Raw Files":
                subj = _extract_subject_from_mff(f)
            else:
                subj = _extract_subject_from_path(f)
            if subj:
                subjects.add(subj)
        stage_subjects[stage] = subjects
        all_subjects.update(subjects)

    # Build DataFrame
    rows = []
    for subj in sorted(all_subjects):
        row = {"subject_id": subj}
        for stage in PIPELINE_STAGES:
            row[stage] = subj in stage_subjects.get(stage, set())
        rows.append(row)

    return pd.DataFrame(rows)


# ── Merging and analysis ────────────────────────────────────────────


def has_skipped_steps(row, steps=None):
    """Check if a subject has non-contiguous pipeline progression.

    Returns ``True`` if there are gaps in the completed stages
    (e.g., passed BIDS but skipped Pylossless yet shows
    Segmentation).

    Parameters
    ----------
    row : pd.Series
        A row from the tracking DataFrame with boolean columns
        for each pipeline stage.
    steps : list of str, optional
        Ordered stage names. Defaults to
        :data:`~q1k.config.PIPELINE_STAGES`.

    Returns
    -------
    bool
    """
    if steps is None:
        steps = PIPELINE_STAGES

    completed = [i for i, s in enumerate(steps) if row.get(s, False)]
    if len(completed) <= 1:
        return False
    return (max(completed) - min(completed) + 1) != len(completed)


def compute_last_stage(row, steps=None):
    """Determine the last completed pipeline stage for a subject.

    Parameters
    ----------
    row : pd.Series
        A row with boolean columns for each pipeline stage.
    steps : list of str, optional
        Ordered stage names.

    Returns
    -------
    str
        Name of the last completed stage, or ``"None"`` if no
        stage is completed.
    """
    if steps is None:
        steps = PIPELINE_STAGES

    last = "None"
    for stage in steps:
        if row.get(stage, False):
            last = stage
    # Rename final stage for clarity
    if last == "Autoreject":
        last = "Completed"
    return last


def merge_tracking(demographics_df, pipeline_df):
    """Merge demographics with pipeline stage scanning results.

    Parameters
    ----------
    demographics_df : pd.DataFrame
        From :func:`load_redcap` — must have ``bids_id`` column.
    pipeline_df : pd.DataFrame
        From :func:`scan_pipeline_stages` — must have
        ``subject_id`` column.

    Returns
    -------
    pd.DataFrame
        Merged DataFrame with ``Last_stage`` and ``discrepancy``
        columns added.
    """
    merged = demographics_df.merge(
        pipeline_df,
        left_on="bids_id",
        right_on="subject_id",
        how="outer",
    )

    # Compute last stage and discrepancy
    merged["Last_stage"] = merged.apply(compute_last_stage, axis=1)
    merged["discrepancy"] = merged.apply(has_skipped_steps, axis=1)

    return merged


# ── Report generation ────────────────────────────────────────────────


def generate_tracking_csv(
    project_path, task, redcap_dir, output_dir=None
):
    """Generate a per-task tracking CSV.

    Loads REDCap data, scans pipeline stages, merges, and saves.

    Parameters
    ----------
    project_path : str or Path
        Path to the project experimental directory.
    task : str
        Task code.
    redcap_dir : str or Path
        Path to REDCap export directory.
    output_dir : str or Path, optional
        Output directory. Defaults to ``tracking/`` sibling of
        ``project_path``.

    Returns
    -------
    Path
        Path to the saved CSV file.
    """
    if output_dir is None:
        output_dir = Path(project_path).parent / "tracking"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    demo_df = load_redcap(redcap_dir)
    pipeline_df = scan_pipeline_stages(project_path, task)
    merged = merge_tracking(demo_df, pipeline_df)

    # Replace True/False with Yes/No for pipeline stage columns
    for stage in PIPELINE_STAGES:
        if stage in merged.columns:
            merged[stage] = merged[stage].map(
                {True: "Yes", False: "No"}
            )

    # Add task-specific fail reason column if available
    fail_col = f"{task}_fail_reason"
    if fail_col not in merged.columns:
        merged[fail_col] = np.nan

    # Add manual check column
    merged["Manual Check"] = np.nan

    # Save
    date_str = datetime.now().strftime("%Y_%m_%d")
    out_path = output_dir / f"Data_tracking_{task}_detailed_{date_str}.csv"
    merged.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")

    return out_path


def generate_all_tracking_csvs(
    project_path, redcap_dir, output_dir=None, tasks=None
):
    """Generate tracking CSVs for all (or specified) tasks.

    Parameters
    ----------
    project_path : str or Path
        Path to the project experimental directory.
    redcap_dir : str or Path
        Path to REDCap export directory.
    output_dir : str or Path, optional
        Output directory.
    tasks : list of str, optional
        Tasks to process. Defaults to all tasks except RSRio.

    Returns
    -------
    dict
        Mapping of task code to output CSV path.
    """
    if tasks is None:
        tasks = [t for t in VALID_TASKS if t != "RSRio"]

    results = {}
    for task in tasks:
        try:
            path = generate_tracking_csv(
                project_path, task, redcap_dir, output_dir
            )
            results[task] = path
        except Exception as e:
            print(f"Error generating tracking for {task}: {e}")
            results[task] = None

    return results


# ── SharePoint comparison ────────────────────────────────────────────


def compare_with_sharepoint(
    automated_df,
    sharepoint_path,
    sheet_name="Resting state",
    mni_upload_date=None,
    hsj_upload_date=None,
):
    """Compare automated tracking with a SharePoint Excel sheet.

    Detects changes in ``Last_stage`` between the automated output
    and the manually maintained SharePoint sheet, and applies
    automatic classification rules.

    Parameters
    ----------
    automated_df : pd.DataFrame
        Automated tracking DataFrame (from :func:`merge_tracking`).
    sharepoint_path : str or Path
        Path to the SharePoint Excel file.
    sheet_name : str
        Sheet name to read from the Excel file.
    mni_upload_date : str, optional
        Upload cutoff date for MNI site (``"YYYY-MM-DD"``).
    hsj_upload_date : str, optional
        Upload cutoff date for HSJ site (``"YYYY-MM-DD"``).

    Returns
    -------
    pd.DataFrame
        Updated DataFrame with ``changed``, ``Reason``,
        ``Further details``, ``Fixed``, and ``Manual Check``
        columns.
    """
    sp_df = pd.read_excel(sharepoint_path, sheet_name=sheet_name)

    # Extract relevant columns from SharePoint
    sp_cols = ["q1k_ID", "Last_stage", "Reason", "Further details",
               "Fixed", "Manual Check"]
    sp_subset = sp_df[[c for c in sp_cols if c in sp_df.columns]].copy()

    if "Last_stage" in sp_subset.columns:
        sp_subset = sp_subset.rename(
            columns={"Last_stage": "Last_stage_prev"}
        )

    # Merge
    merged = automated_df.merge(
        sp_subset, on="q1k_ID", how="left"
    )

    # Detect changes
    if "Last_stage_prev" in merged.columns:
        merged["changed"] = (
            merged["Last_stage"] != merged["Last_stage_prev"]
        ).map({True: "Yes", False: "No"})
    else:
        merged["changed"] = "Yes"

    # Apply automatic classification rules
    merged = _apply_classification_rules(
        merged, mni_upload_date, hsj_upload_date
    )

    return merged


def _apply_classification_rules(df, mni_upload_date, hsj_upload_date):
    """Apply automatic classification rules to the comparison data."""
    if "Reason" not in df.columns:
        df["Reason"] = np.nan

    for idx, row in df.iterrows():
        # Skip rows that already have a reason
        if pd.notna(row.get("Reason")):
            continue

        # Rule 1: Completed but no Phase 3 consent
        if (row.get("Last_stage") == "Completed"
                and row.get("phase_3_consented") == "No"):
            df.at[idx, "Reason"] = "Complete but no consent"
            continue

        # Rule 2/3: Not uploaded yet (visit after upload date)
        visit_date = row.get("visit_date")
        site = row.get("site")
        if pd.notna(visit_date) and isinstance(visit_date, str):
            if (site == "mni" and mni_upload_date
                    and visit_date > mni_upload_date):
                df.at[idx, "Reason"] = "Not uploaded"
                continue
            if (site == "hsj" and hsj_upload_date
                    and visit_date > hsj_upload_date):
                df.at[idx, "Reason"] = "Not uploaded"
                continue

        # Rule 4: New/changed participant not yet checked
        if row.get("changed") == "Yes":
            df.at[idx, "Reason"] = "New participant, not checked yet"

    return df


def generate_data_loss_excel(
    project_path,
    redcap_dir,
    sharepoint_path=None,
    output_dir=None,
    mni_upload_date=None,
    hsj_upload_date=None,
):
    """Generate a multi-sheet Excel data loss report.

    Parameters
    ----------
    project_path : str or Path
        Path to the project experimental directory.
    redcap_dir : str or Path
        Path to REDCap export directory.
    sharepoint_path : str or Path, optional
        Path to SharePoint Excel for comparison. If *None*, generates
        a fresh report without comparison.
    output_dir : str or Path, optional
        Output directory.
    mni_upload_date : str, optional
        MNI upload cutoff date.
    hsj_upload_date : str, optional
        HSJ upload cutoff date.

    Returns
    -------
    Path
        Path to the saved Excel file.
    """
    if output_dir is None:
        output_dir = Path(project_path).parent / "tracking"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tasks = [t for t in VALID_TASKS if t != "RSRio"]
    task_sheet_names = {
        "RS": "Resting state",
        "GO": "Gap Overlap",
        "TO": "Tone Oddball",
        "VEP": "Visual Evoked Potential",
        "AEP": "Auditory Evoked Potential",
        "NSP": "Naturalistic Social Preference",
        "PLR": "Pupil Light Reflex",
        "VS": "Visual Search",
    }

    date_str = datetime.now().strftime("%Y_%m_%d")
    out_path = (
        output_dir / f"Sharepoint_dataloss_report_{date_str}.xlsx"
    )

    demo_df = load_redcap(redcap_dir)

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        for task in tasks:
            pipeline_df = scan_pipeline_stages(project_path, task)
            merged = merge_tracking(demo_df, pipeline_df)

            # Replace True/False with Yes/No
            for stage in PIPELINE_STAGES:
                if stage in merged.columns:
                    merged[stage] = merged[stage].map(
                        {True: "Yes", False: "No"}
                    )

            if sharepoint_path:
                sheet = task_sheet_names.get(task, task)
                try:
                    merged = compare_with_sharepoint(
                        merged, sharepoint_path,
                        sheet_name=sheet,
                        mni_upload_date=mni_upload_date,
                        hsj_upload_date=hsj_upload_date,
                    )
                except Exception as e:
                    print(
                        f"Warning: Could not compare {task} "
                        f"with SharePoint: {e}"
                    )

            sheet_name = task_sheet_names.get(task, task)
            merged.to_excel(
                writer, sheet_name=sheet_name, index=False
            )

    print(f"Saved: {out_path}")
    return out_path


# ── Helper functions ─────────────────────────────────────────────────


def _safe_q1k_to_bids(q1k_id):
    """Convert Q1K ID to BIDS, returning None on failure."""
    if pd.isna(q1k_id):
        return None
    try:
        return q1k_to_bids(str(q1k_id))
    except (ValueError, IndexError):
        return None


def _infer_site(q1k_id):
    """Infer site code from a Q1K ID string."""
    if pd.isna(q1k_id):
        return None
    q1k_id = str(q1k_id)
    if "HSJ" in q1k_id or "NIM" in q1k_id:
        return "hsj"
    if "MHC" in q1k_id:
        return "mni"
    return "unknown"


def _infer_group(q1k_id):
    """Infer family role from a Q1K ID suffix."""
    if pd.isna(q1k_id):
        return None
    q1k_id = str(q1k_id)
    if "_P" in q1k_id:
        return "proband"
    if "_F" in q1k_id:
        return "father"
    if "_M" in q1k_id:
        return "mother"
    if "_S" in q1k_id:
        return "sibling"
    return "unknown"
