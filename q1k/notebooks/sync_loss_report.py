import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium")


@app.cell
def parameters():
    # __Q1K_PARAMETERS__
    # The above comment is replaced by the CLI with actual values.
    project_path = ""
    task_id = ""
    subject_id = ""
    session_id = "01"
    run_id = "1"
    et_sync = False
    return project_path, task_id, subject_id, session_id, run_id, et_sync


@app.cell
def imports():
    import warnings
    from pathlib import Path

    import mne
    import os
    import mne_bids
    import numpy as np
    import plotly.express as px
    import plotly.io as pio
    import pylossless as ll
    warnings.filterwarnings("ignore")

    from q1k.bids import write_bids_eeg
    from q1k.config import EOG_CHANNELS
    from q1k.sync_loss.tools import apply_ll, eeg_et_combine
    return (mne, mne_bids, np, ll, px, pio, Path, warnings,
            apply_ll, eeg_et_combine, write_bids_eeg, EOG_CHANNELS)


@app.cell
def header(subject_id, task_id):
    import marimo as mo
    mo.md(f"## Sync + Lossless Report: {subject_id} - {task_id}")
    return (mo,)


@app.cell
def load_data(mne, mne_bids, ll, project_path, subject_id, session_id,
              task_id, run_id):
    pylossless_path = "derivatives/pylossless"
    init_path = "derivatives/init"
    # Read raw BIDS data
    bids_path = mne_bids.BIDSPath(
        subject=subject_id, session=session_id, task=task_id,
        run=run_id, datatype="eeg", suffix="eeg", root=str(Path(project_path)/ init_path),
    )
    eeg_raw = mne_bids.read_raw_bids(bids_path=bids_path, verbose=False)
    eeg_raw.load_data()
    device_info = eeg_raw.info["device_info"]

    # Read pylossless derivatives
    bids_ll_path = mne_bids.BIDSPath(
        subject=subject_id, session=session_id, task=task_id,
        run=run_id, datatype="eeg", suffix="eeg",
        root=str(Path(project_path)/ pylossless_path),
    )
    ll_state = ll.LosslessPipeline()
    ll_state = ll_state.load_ll_derivative(bids_ll_path)
    eeg_ll_raw = ll_state.raw.copy()

    # Crop raw to match lossless state
    start_time = eeg_ll_raw.times[0]
    end_time = eeg_ll_raw.times[-1]
    eeg_raw = eeg_raw.copy().crop(tmin=start_time, tmax=end_time)
    eeg_raw.set_annotations(eeg_ll_raw.annotations)

    return (eeg_raw, bids_ll_path, ll_state, device_info,
            pylossless_path)


@app.cell
def filter_data(mne, eeg_raw, EOG_CHANNELS):
    eeg_filt_raw = eeg_raw.copy()
    eeg_filt_raw.load_data()
    eeg_filt_raw.info["bads"].extend(EOG_CHANNELS)
    eeg_filt_raw = eeg_raw.filter(l_freq=1.0, h_freq=90.0, picks="eeg")
    eeg_filt_raw.notch_filter(freqs=60, picks="eeg", method="fir",
                         fir_design="firwin")
    return (eeg_filt_raw,)


@app.cell
def sync_et(mne, np, Path, eeg_filt_raw, et_sync, eeg_et_combine, project_path,
            bids_ll_path, task_id, subject_id, session_id, run_id):
    if et_sync:
        bids_ll_path_str = str(bids_ll_path.fpath)
        # Building path to _et.fif using BIDS convention
        clean_subject_id = subject_id.removeprefix("sub-")
        et_base = Path(project_path) / "sourcedata" / site_code / "et"
        et_fif_filename = f"sub-{subject_id}_ses-{session_id}_task-{task_id}_run-{run_id}_et.fif"
        et_fif_path = (
            Path(project_path)
            / "derivatives" / "init" / f"sub-{subject_id}"
            / f"ses-{session_id}"
            / "et"
            /et_fif_filename
        )
        print(f"Looking for ET file: {et_fif_path}")
        if not et_fif_path.exists():
            raise FileNotFoundError(
                f"ET .fif file not found: {et_fif_path}\n"
                f"Run init_report.py first to generate this file."
            )
        et_raw = mne.io.read_raw_fif(str(et_fif_path), preload=True)
        #Set ch_names for BAD_ACQ_skip
        ch_types = et_raw.get_channel_types()
        ch_names = et_raw.ch_names
        eye_ch = tuple(
            n for n, t in zip(ch_names, ch_types)
            if t in ('eyegaze', 'pupil')
        )
        for ann in et_raw.annotations:
            if ann['description'] == 'BAD_ACQ_SKIP':
                ann['ch_names'] = eye_ch
        # Interpolate blinks
        mne.preprocessing.eyetracking.interpolate_blinks(
            et_raw, match=("BAD_blink",),
            buffer=(0.05, 0.2), interpolate_gaze=True,
        )
        data = et_raw.get_data()
        data[np.isnan(data)] = 0
        et_raw._data = data

        '''# Get sync events
        eeg_events, eeg_event_dict = mne.events_from_annotations(eeg_filt_raw)
        et_events, et_event_dict = mne.events_from_annotations(et_raw)
        print(f"EEG event keys: {list(eeg_event_dict.keys())}")
        print(f"ET event keys : {list(et_event_dict.keys())}")

        if "eeg_sync_time" not in eeg_event_dict:
            raise ValueError("'eeg_sync_time' not found in EEG annotations. "
                             "Re-run generate_et_fif.py for this subject.")
        if "et_sync_time" not in et_event_dict:
            raise ValueError("'et_sync_time' not found in ET .fif annotations. "
                            "Re-run init stage for this subject.")
        eeg_sync_value = eeg_event_dict["eeg_sync_time"]
        et_sync_value = et_event_dict["et_sync_time"]
        eeg_syncs = eeg_events[eeg_events[:, 2] == eeg_sync_value]
        et_syncs = et_events[et_events[:, 2] == et_sync_value]

        eeg_sync_times = eeg_syncs[:, 0] / eeg_filt_raw.info["sfreq"]
        et_sync_times = et_syncs[:, 0] / et_raw.info["sfreq"]
        print(f"EEG sync points: {len(eeg_sync_times)}")
        print(f"ET  sync points: {len(et_sync_times)}")'''
        # Get sync events
        eeg_events, eeg_event_dict = mne.events_from_annotations(eeg_filt_raw)
        et_events, et_event_dict = mne.events_from_annotations(et_raw)

        # EEG sync: use eeg_sync_time if present, else use GO stim events
        if "eeg_sync_time" in eeg_event_dict:
            print("Using eeg_sync_time from EEG annotations.")
            eeg_syncs = eeg_events[eeg_events[:, 2] == eeg_event_dict["eeg_sync_time"]]
            eeg_sync_times = eeg_syncs[:, 0] / eeg_filt_raw.info["sfreq"]
        else:
            print("eeg_sync_time not found — using GO stim events as sync.")
            stim_ids = [v for k, v in eeg_event_dict.items()
                        if k in ("dtoc_d", "dtbc_d", "dtgc_d")]
            if not stim_ids:
                raise ValueError(f"No GO stim events found. Available: {list(eeg_event_dict.keys())}")
            eeg_stims = eeg_events[np.isin(eeg_events[:, 2], stim_ids)]
            eeg_sync_times = eeg_stims[:, 0] / eeg_filt_raw.info["sfreq"]

        # ET sync: must have et_sync_time
        if "et_sync_time" not in et_event_dict:
            raise ValueError(f"'et_sync_time' not in ET .fif. Available: {list(et_event_dict.keys())}")
        et_syncs = et_events[et_events[:, 2] == et_event_dict["et_sync_time"]]
        et_sync_times = et_syncs[:, 0] / et_raw.info["sfreq"]

        print(f"EEG sync points: {len(eeg_sync_times)}")
        print(f"ET  sync points: {len(et_sync_times)}")

        # Trim to equal length if needed
        if len(eeg_sync_times) != len(et_sync_times):
            n = min(len(eeg_sync_times), len(et_sync_times))
            print(f"WARNING: trimming to {n} sync points")
            eeg_sync_times = eeg_sync_times[:n]
            et_sync_times  = et_sync_times[:n]
        # Combine
        eeg_sync_raw, et_raw = eeg_et_combine(
            eeg_filt_raw, et_raw, eeg_sync_times, et_sync_times,
            eeg_events, eeg_event_dict, et_events, et_event_dict,
        )
    else:
        eeg_sync_raw = eeg_filt_raw

    return (eeg_sync_raw,)


@app.cell
def apply_lossless(apply_ll, bids_ll_path, ll_state, eeg_sync_raw):
    eeg_loss_raw = apply_ll(bids_ll_path, ll_state, eeg_sync_raw)
    return (eeg_loss_raw,)


@app.cell
def save_output(mne, eeg_loss_raw, write_bids_eeg, subject_id,
                session_id, task_id, project_path, pylossless_path, Path):
    # Convert ET channel types to misc for BIDS compatibility
    mapping = {ch: "misc" for ch, ct in zip(eeg_loss_raw.ch_names,
               eeg_loss_raw.get_channel_types()) if ct in ("eyegaze","pupil")}
    if mapping:
        eeg_loss_raw.set_channel_types(mapping)

    eeg_loss_events, eeg_loss_event_dict = mne.events_from_annotations(
        eeg_loss_raw
    )
    eeg_loss_events[:, 0] -= eeg_loss_raw.first_samp

    sync_loss_path = "derivatives/sync_loss/"
    loss_path = str(Path(project_path) / sync_loss_path)

    eeg_bids_path = write_bids_eeg(
        eeg_loss_raw, eeg_loss_events, eeg_loss_event_dict,
        subject_id, session_id, task_id, loss_path,
    )
    return (eeg_bids_path,)


if __name__ == "__main__":
    app.run()
