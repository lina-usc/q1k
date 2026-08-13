"""PyLossless pipeline execution for artifact rejection.

Runs the PyLossless pipeline on a single BIDS EEG file and saves
the derivatives.
"""
import os

import mne_bids
import pylossless as ll


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
        root=os.path.join(project_path, "derivatives", "init", task_id),
    )

    print(f"Running on: {subject_id}")
    print(f"BIDS PATH IS : --{bids_path}")

    raw = mne_bids.read_raw_bids(bids_path=bids_path)
    raw.load_data()

    # Mark EOG channels as bad
    eog_chans = ['E125', 'E126', 'E127', 'E128']
    raw.info["bads"].extend(eog_chans)

    # Run pylossless (matches original)
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
        root=os.path.join(out_path, task_id),
    )


    pipeline.save(out_bids, overwrite=True)
    print(f"Saved pylossless derivatives for {subject_id}")



    '''
    mne_bids.write_raw_bids(
    pipeline.raw,
    out_bids,
    overwrite=True,
    format='EDF',
    allow_preload=True,
    physical_range='channelwise' ) # ← Preserves original amplitude
    print(f"Saved pylossless derivatives for {subject_id}")'''
