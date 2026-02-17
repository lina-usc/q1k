"""Unified Slurm job submission utilities."""

import glob
import os
import re
import subprocess
from pathlib import Path

from q1k.bids import extract_bids_info


def submit_slurm_job(script, job_name, output_dir, task_id, *args):
    """Submit a job to the Slurm scheduler.

    Parameters
    ----------
    script : str or Path
        Path to the Slurm batch script (.sh file).
    job_name : str
        Name for the Slurm job.
    output_dir : str or Path
        Directory for Slurm output files. A subdirectory named after
        ``task_id`` will be used.
    task_id : str
        Task identifier (used for output subdirectory).
    *args : str
        Additional arguments passed to the batch script.

    Returns
    -------
    sbatch_command : list[str]
        The command that was run.
    result : subprocess.CompletedProcess
        The result of the subprocess call.
    """
    output_path = Path(output_dir) / task_id
    output_path.mkdir(parents=True, exist_ok=True)

    out_file = f"{job_name}.out"

    sbatch_command = [
        "sbatch",
        f"--job-name={job_name}",
        f"--output={output_path / out_file}",
        str(script),
        *[str(a) for a in args],
    ]

    print("Running command:", " ".join(sbatch_command))

    try:
        result = subprocess.run(
            sbatch_command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        print("Job submitted successfully!")
        print(result.stdout.decode())
    except subprocess.CalledProcessError as e:
        print("Error submitting job:", e.stderr.decode())
        raise

    return sbatch_command, result


def find_unprocessed(input_pattern, output_pattern):
    """Find input files that have not yet been processed.

    Compares input files against output files using subject IDs
    extracted from BIDS filenames.

    Parameters
    ----------
    input_pattern : str
        Glob pattern for input files.
    output_pattern : str
        Glob pattern for existing output files.

    Returns
    -------
    list[str]
        Paths to input files that have no corresponding output.
    """
    input_files = glob.glob(input_pattern, recursive=True)
    output_files = glob.glob(output_pattern, recursive=True)

    # Extract subject IDs from output files
    processed_subjects = set()
    for f in output_files:
        basename = os.path.basename(f)
        match = re.search(r"sub-([^_]+)", basename)
        if match:
            processed_subjects.add(match.group(1))

    # Filter input files
    unprocessed = []
    for f in input_files:
        basename = os.path.basename(f)
        match = re.search(r"sub-([^_]+)", basename)
        if match and match.group(1) not in processed_subjects:
            unprocessed.append(f)

    return unprocessed
