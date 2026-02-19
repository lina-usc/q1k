"""PyLossless pipeline execution for artifact rejection.

Runs the PyLossless pipeline on a single BIDS EEG file and saves
the derivatives.
"""

import mne_bids
import pylossless as ll

from q1k.config import EOG_CHANNELS


def run_pylossless(project_path, subject_id, session_id, task_id, run_id,
                   out_path):
    """Run PyLossless on a single subject/task.

    Parameters
    ----------
    project_path : str
        BIDS root directory.
    subject_id : str
        BIDS subject identifier.
    session_id : str
        BIDS session identifier.
    task_id : str
        BIDS task identifier.
    run_id : str
        BIDS run identifier.
    out_path : str
        Output directory for pylossless derivatives.
    """
    bids_path = mne_bids.BIDSPath(
        subject=subject_id,
        session=session_id,
        task=task_id,
        run=run_id,
        datatype="eeg",
        root=project_path,
    )

    print(f"Running on: {subject_id}")

    raw = mne_bids.read_raw_bids(bids_path=bids_path)
    raw.load_data()

    # Mark EOG channels as bad
    raw.info["bads"].extend(EOG_CHANNELS)

    # Run pylossless with default config
    config = ll.config.Config()
    config.load_default()
    pipeline = ll.LosslessPipeline(config=config)
    pipeline.run_with_raw(raw)

    # Save derivatives
    out_bids = mne_bids.BIDSPath(
        subject=subject_id,
        session=session_id,
        task=task_id,
        run=run_id,
        suffix="eeg",
        extension=".edf",
        datatype="eeg",
        root=out_path,
    )
    pipeline.save(pipeline.get_derivative_path(out_bids), overwrite=True)
    print(f"Saved pylossless derivatives for {subject_id}")
