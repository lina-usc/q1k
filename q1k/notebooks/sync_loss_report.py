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
    import mne
    import mne_bids
    import numpy as np
    import pylossless as ll
    import plotly.express as px
    import plotly.io as pio
    from pathlib import Path
    import warnings
    warnings.filterwarnings("ignore")

    from q1k.sync_loss.tools import apply_ll, eeg_et_combine
    from q1k.bids import write_bids_eeg
    from q1k.config import EOG_CHANNELS
    return (mne, mne_bids, np, ll, px, pio, Path, warnings,
            apply_ll, eeg_et_combine, write_bids_eeg, EOG_CHANNELS)


@app.cell
def header(subject_id, task_id):
    import marimo as mo
    mo.md(f"## Sync + Lossless Report: {subject_id} - {task_id}")
    return (mo,)


@app.cell
def load_data(mne, mne_bids, ll, project_path, subject_id, session_id,
              task_id):
    pylossless_path = "derivatives/pylossless/"

    # Read raw BIDS data
    bids_path = mne_bids.BIDSPath(
        subject=subject_id, session=session_id, task=task_id,
        run="1", datatype="eeg", suffix="eeg", root=project_path,
    )
    eeg_raw = mne_bids.read_raw_bids(bids_path=bids_path, verbose=False)
    device_info = eeg_raw.info["device_info"]

    # Read pylossless derivatives
    bids_ll_path = mne_bids.BIDSPath(
        subject=subject_id, session=session_id, task=task_id,
        run="1", datatype="eeg", suffix="eeg",
        root=project_path + pylossless_path,
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
    eeg_raw.load_data()
    eeg_raw.info["bads"].extend(EOG_CHANNELS)
    eeg_raw = eeg_raw.filter(l_freq=1.0, h_freq=90.0, picks="eeg")
    eeg_raw.notch_filter(freqs=60, picks="eeg", method="fir",
                         fir_design="firwin")
    return (eeg_raw,)


@app.cell
def sync_et(mne, np, eeg_raw, et_sync, eeg_et_combine, project_path,
            bids_ll_path, task_id):
    if et_sync:
        bids_ll_path_str = str(bids_ll_path.fpath)
        et_bids_path = bids_ll_path_str.replace(".edf", ".fif")
        et_bids_path = et_bids_path.replace("eeg", "et")
        et_bids_path = et_bids_path.replace("derivatives/pylossless/", "")

        et_raw = mne.io.read_raw_fif(et_bids_path, preload=True)

        # Interpolate blinks
        mne.preprocessing.eyetracking.interpolate_blinks(
            et_raw, match=("BAD_blink", "BAD_ACQ_SKIP"),
            buffer=(0.05, 0.2), interpolate_gaze=True,
        )
        data = et_raw.get_data()
        data[np.isnan(data)] = 0
        et_raw._data = data

        # Get sync events
        eeg_events, eeg_event_dict = mne.events_from_annotations(eeg_raw)
        et_events, et_event_dict = mne.events_from_annotations(et_raw)

        eeg_sync_value = eeg_event_dict["eeg_sync_time"]
        et_sync_value = et_event_dict["et_sync_time"]
        eeg_syncs = eeg_events[eeg_events[:, 2] == eeg_sync_value]
        et_syncs = et_events[et_events[:, 2] == et_sync_value]

        eeg_sync_times = eeg_syncs[:, 0] / eeg_raw.info["sfreq"]
        et_sync_times = et_syncs[:, 0] / et_raw.info["sfreq"]

        # Combine
        eeg_raw, et_raw = eeg_et_combine(
            eeg_raw, et_raw, eeg_sync_times, et_sync_times,
            eeg_events, eeg_event_dict, et_events, et_event_dict,
        )
    return (eeg_raw,)


@app.cell
def apply_lossless(apply_ll, bids_ll_path, ll_state, eeg_raw):
    eeg_loss_raw = apply_ll(bids_ll_path, ll_state, eeg_raw)
    return (eeg_loss_raw,)


@app.cell
def save_output(mne, eeg_loss_raw, write_bids_eeg, subject_id,
                session_id, task_id, project_path, pylossless_path):
    eeg_loss_events, eeg_loss_event_dict = mne.events_from_annotations(
        eeg_loss_raw
    )
    eeg_loss_events[:, 0] -= eeg_loss_raw.first_samp

    sync_loss_path = "derivatives/sync_loss/"
    loss_path = project_path + pylossless_path + sync_loss_path

    eeg_bids_path = write_bids_eeg(
        eeg_loss_raw, eeg_loss_events, eeg_loss_event_dict,
        subject_id, session_id, task_id, loss_path,
    )
    return (eeg_bids_path,)


if __name__ == "__main__":
    app.run()
