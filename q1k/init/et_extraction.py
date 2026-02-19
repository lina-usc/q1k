"""Eye-tracking data extraction and organization utilities.

Handles the extraction, conversion (EDF → ASC via eyelinkio), and
organization of eye-tracking data into BIDS-compatible directory
structures.  Provides functions for:

- Standardizing participant IDs between EEG and ET naming conventions
- Loading demographic data from REDCap exports
- Converting Eyelink ``.edf`` files to ``.asc`` using ``eyelinkio``
- Organizing ET files alongside EEG data in the project structure
"""

import glob
import os
import shutil
from pathlib import Path

import pandas as pd

from q1k.bids import eb_id_transform, format_id


def build_eeg_lookup(project_path, sites=("HSJ", "MHC")):
    """Build a lookup table mapping EEG subject IDs to ET IDs.

    Scans the ``sourcedata/eeg/`` directories for all sites and
    extracts truncated IDs suitable for matching with eye-tracking
    participant folder names.

    Parameters
    ----------
    project_path : str or Path
        Path to the project experimental directory (containing
        ``sourcedata/``).
    sites : tuple of str
        Site codes to scan.

    Returns
    -------
    pd.DataFrame
        Lookup table with columns ``q1k_ID``, ``et_ID``,
        ``family_ID``, ``subject``, and ``bids_folder``.
    """
    eeg_subjects = []
    truncated_ids = []
    family_ids = []

    for site in sites:
        pattern = os.path.join(
            project_path, site, "sourcedata", "eeg", "*",
        )
        for path in glob.glob(pattern):
            subject_id = os.path.basename(path)
            if "Pilots" in subject_id or "__" in subject_id:
                continue

            eeg_subjects.append(subject_id)
            truncated = _extract_truncated_id(subject_id)
            truncated_ids.append(truncated)
            family_ids.append(truncated.split("_")[0])

    df = pd.DataFrame({
        "q1k_ID": eeg_subjects,
        "et_ID": truncated_ids,
        "family_ID": family_ids,
        "subject": truncated_ids,
    })

    # Standardize IDs
    df["et_ID"] = df["et_ID"].apply(format_id)

    # Create BIDS-compliant subject ID (no underscores)
    df["subject"] = df["et_ID"].str.replace("_", "", regex=False)
    df["bids_folder"] = "sub-" + df["subject"]

    return df


def _extract_truncated_id(subject_id):
    """Extract truncated ID from a full Q1K EEG subject ID.

    Handles the various naming conventions across sites::

        Q1K_HSJ_100123_F1 → 123_F1
        Q1K_MHC_200181_P  → 181_P
        Q1K_HSJ_1025xxx   → xxx (after 1025)
        Q1K_HSJ_1525xxx   → xxx (after 1525)
        Q1K_HSJ_3530xxx   → xxx (after 3530)
    """
    if "1025" in subject_id:
        return subject_id.split("1025")[1][1:]
    elif "3530" in subject_id:
        return subject_id.split("3530")[1][1:]
    elif "1525" in subject_id:
        return subject_id.split("1525")[1][1:]
    elif "HSJ" in subject_id:
        return subject_id.split("Q1K_HSJ_100")[1]
    elif "MHC" in subject_id:
        return subject_id.split("Q1K_MHC_200")[1]
    else:
        # Fallback: return as-is
        return subject_id


def load_demographics(redcap_dir):
    """Load demographic data from a REDCap export directory.

    Looks for a CSV file containing "Demographics" in the filename
    within ``redcap_dir``, then extracts and standardizes the Q1K ID
    columns.

    Parameters
    ----------
    redcap_dir : str or Path
        Directory containing the REDCap CSV export.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``record_id`` and ``q1k_ID``.

    Raises
    ------
    FileNotFoundError
        If no demographics file is found in the directory.
    """
    demo_file = None
    for f in glob.glob(os.path.join(str(redcap_dir), "*")):
        if "Demographics" in f:
            demo_file = f
            break

    if demo_file is None:
        raise FileNotFoundError(
            f"No demographics CSV found in {redcap_dir}"
        )

    demo = pd.read_csv(demo_file)
    demo = demo.rename(columns={
        "q1k_proband_id_1": "proband_id",
        "q1k_relative_idgenerated_1": "relative_id",
    })

    demo = demo[
        ["record_id", "proband_id", "relative_id"]
    ].drop_duplicates(subset=["record_id"])

    demo["q1k_ID"] = demo["proband_id"].combine_first(
        demo["relative_id"]
    )
    demo = demo.loc[~demo.q1k_ID.isna()].drop(
        columns=["proband_id", "relative_id"]
    )

    return demo


def convert_edf_to_asc(edf_path, asc_path):
    """Convert an Eyelink .edf file to .asc using eyelinkio.

    Parameters
    ----------
    edf_path : str or Path
        Path to the source ``.edf`` file.
    asc_path : str or Path
        Path for the output ``.asc`` file.
    """
    from eyelinkio.edf.to_asc import to_asc

    to_asc(str(edf_path), str(asc_path))


def organize_et_files(
    input_dir,
    output_dir,
    eeg_lookup_df,
    tasks=("GO", "NSP", "AS", "VS", "FSP", "REST", "RS",
           "VEP", "AEP"),
):
    """Organize eye-tracking files into BIDS-compatible directories.

    Copies ``.txt`` and ``.edf`` files from source ET directories into
    participant-specific BIDS folders, renaming them with standardized
    subject IDs.

    Parameters
    ----------
    input_dir : str or Path
        Root directory containing raw ET task folders.
    output_dir : str or Path
        Output directory for organized ET data.
    eeg_lookup_df : pd.DataFrame
        Lookup table from :func:`build_eeg_lookup`.
    tasks : tuple of str
        Task codes to process.

    Returns
    -------
    list of list
        Records of participants with ET data but no matching EEG,
        each as ``[et_id, site, task]``.
    """
    missing_eeg = []
    os.makedirs(output_dir, exist_ok=True)

    for task in tasks:
        # NSP/RS/REST/VEP/AEP use .edf files directly
        if task in ("NSP", "RS", "REST", "VEP", "AEP"):
            pattern = os.path.join(
                str(input_dir), "**",
                f"*{task}*/results/*/*.edf",
            )
            task_files = glob.glob(pattern, recursive=True)
        else:
            pattern = os.path.join(
                str(input_dir), "**",
                f"*{task}_pfp*.txt",
            )
            task_files = glob.glob(pattern, recursive=True)

        for task_file in task_files:
            et_id = os.path.basename(os.path.dirname(task_file))
            transformed_id = eb_id_transform(et_id)
            participant_dir = os.path.dirname(task_file)

            # Extract site from path
            if "experimental/" in task_file:
                site = task_file.split("experimental/")[1].split("/")[0]
            else:
                site = "unknown"

            # Match to EEG subject
            match = eeg_lookup_df.loc[
                eeg_lookup_df["et_ID"] == transformed_id
            ]

            if not match.empty:
                subj = match.iloc[0]["subject"]
                bids_id = f"sub-{subj}"
                out_folder = os.path.join(output_dir, bids_id)
            else:
                if ("Q" in et_id or "q" in et_id) and \
                        "Pilots" not in task_file:
                    missing_eeg.append([et_id, site, task])
                subj = et_id
                out_folder = os.path.join(output_dir, et_id)

            os.makedirs(out_folder, exist_ok=True)

            # Copy relevant files
            for fname in os.listdir(participant_dir):
                src = os.path.join(participant_dir, fname)
                if fname.endswith(".txt") and task in fname:
                    dst = os.path.join(
                        out_folder, f"{subj}_{task}.txt"
                    )
                    if not os.path.exists(dst):
                        shutil.copy2(src, dst)
                elif fname.endswith(".edf"):
                    dst = os.path.join(
                        out_folder, f"{subj}_{task}.edf"
                    )
                    if not os.path.exists(dst):
                        shutil.copy2(src, dst)

    return missing_eeg


def batch_convert_edf_to_asc(bids_et_dir, eeg_lookup_df,
                              sourcedata_dir):
    """Convert all .edf files in BIDS ET folders to .asc.

    For each subject's BIDS folder, converts ``.edf`` files to
    ``.asc`` using ``eyelinkio`` and copies the result to the
    corresponding non-BIDS ``sourcedata`` folder.

    Parameters
    ----------
    bids_et_dir : str or Path
        Directory containing ``sub-*`` folders with ``.edf`` files.
    eeg_lookup_df : pd.DataFrame
        Lookup table from :func:`build_eeg_lookup`.
    sourcedata_dir : str or Path
        Root of the non-BIDS sourcedata directory tree.
    """
    for bids_folder in glob.glob(
        os.path.join(str(bids_et_dir), "sub-*")
    ):
        subject_id = os.path.basename(bids_folder).replace(
            "sub-", ""
        )
        match = eeg_lookup_df.loc[
            eeg_lookup_df["subject"] == subject_id
        ]
        if match.empty:
            print(f"Subject {subject_id} not found. Skipping.")
            continue

        q1k_id = match.iloc[0]["q1k_ID"]
        if "MHC" in q1k_id:
            site = "MHC"
        elif "HSJ" in q1k_id:
            site = "HSJ"
        else:
            site = "HSJ"

        non_bids_path = os.path.join(
            str(sourcedata_dir), site, "et", q1k_id,
        )
        os.makedirs(non_bids_path, exist_ok=True)

        for fname in os.listdir(bids_folder):
            if not fname.lower().endswith(".edf"):
                continue

            src = os.path.join(bids_folder, fname)
            asc_name = Path(fname).stem + ".asc"
            asc_bids = os.path.join(bids_folder, asc_name)
            asc_dest = os.path.join(non_bids_path, asc_name)

            if os.path.exists(asc_dest):
                continue

            if not os.path.exists(asc_bids):
                convert_edf_to_asc(src, asc_bids)
                print(f"  Converted: {fname} -> {asc_name}")

            shutil.copy(asc_bids, asc_dest)
            print(f"  Copied to: {asc_dest}")
