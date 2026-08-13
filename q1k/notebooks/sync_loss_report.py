import marimo
from q1k.sync_loss.tools import apply_ll, eeg_et_combine
#from q1k.init.tools import eeg_et_align

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
        run=run_id, datatype="eeg", suffix="eeg",root=str(Path(project_path) / init_path / task_id), 
    )
    eeg_raw = mne_bids.read_raw_bids(bids_path=bids_path, verbose=False)
    eeg_raw.load_data()
    device_info = eeg_raw.info["device_info"]

    # Read pylossless derivatives
    bids_ll_path = mne_bids.BIDSPath(
        subject=subject_id, session=session_id, task=task_id,
        run=run_id, datatype="eeg", suffix="eeg",
        root=os.path.join(project_path, pylossless_path, task_id),
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
def sync_et(mne, np, Path, eeg_filt_raw, et_sync, eeg_et_combine, project_path, task_id, subject_id, session_id, run_id):
    eeg_events, eeg_event_dict = mne.events_from_annotations(eeg_filt_raw)
    et_events = None
    et_event_dict = None
            
    if et_sync:
        #bids_ll_path_str = str(bids_ll_path.fpath)
        #et_fif_path = bids_ll_path_str.replace(".edf", ".fif")
        #et_fif_path = et_fif_path.replace("eeg", "et")
        #et_fif_path = et_fif_path.replace("derivatives/pylossless/", "")
        #et_fif_path = Path(et_fif_path)
        et_fif_path = (
            Path(project_path) / "derivatives" / "init" / task_id / 
            f"sub-{subject_id}" / f"ses-{session_id}" / "et" / 
            f"sub-{subject_id}_ses-{session_id}_task-{task_id}_run-{run_id}_et.fif"
        )
        print(f"ET file path: {et_fif_path}")

        if not et_fif_path.exists():
            raise FileNotFoundError(
                f"ET .fif not found: {et_fif_path}\n"
                f"Run generate_et_fif.py first to create it."
                f"This subject may be excluded from sync-loss for {task_id} or rerun init if ET is recoverable."
                )
            
        et_raw = mne.io.read_raw_fif(et_fif_path, preload=True)
        # Set ch_names for BAD_ACQ_SKIP
        ch_types = et_raw.get_channel_types()
        ch_names = et_raw.ch_names
        eye_ch = tuple(
            n for n, t in zip(ch_names, ch_types)
            if t in ("eyegaze", "pupil")
        )
        
        for ann in et_raw.annotations:
            if ann["description"] == "BAD_ACQ_SKIP":
                ann["ch_names"] = eye_ch
        
        # Interpolate blinks
        mne.preprocessing.eyetracking.interpolate_blinks(
            et_raw,
            match=("BAD_blink",),
            buffer=(0.05, 0.2),
            interpolate_gaze=True,
        )
        
        data = et_raw.get_data()
        data[np.isnan(data)] = 0
        et_raw._data = data

        # Get sync events
        #eeg_events, eeg_event_dict = mne.events_from_annotations(eeg_filt_raw)
        et_events, et_event_dict = mne.events_from_annotations(et_raw)
        if "et_sync_time" not in et_event_dict:
            raise ValueError(
                f"'et_sync_time' not found in ET .fif annotations. "
                f"Available: {list(et_event_dict.keys())}"
            )
        if "eeg_sync_time" in eeg_event_dict:
            eeg_sync_value = eeg_event_dict["eeg_sync_time"]
            eeg_syncs = eeg_events[eeg_events[:, 2] == eeg_sync_value]
        elif task_id.upper() == "PLR" and "DIN2" in eeg_event_dict:
            print("Using original DIN2 EEG events as PLR sync")
            eeg_sync_value = eeg_event_dict["DIN2"]
            eeg_syncs = eeg_events[eeg_events[:, 2] == eeg_sync_value]
        else:
            raise ValueError(
                f"'eeg_sync_time' not found in EEG annotations. "
                f"Available: {list(eeg_event_dict.keys())}"
            )

        #eeg_sync_value = eeg_event_dict["eeg_sync_time"]
        et_sync_value = et_event_dict["et_sync_time"]

        #eeg_syncs = eeg_events[eeg_events[:, 2] == eeg_sync_value]
        et_syncs = et_events[et_events[:, 2] == et_sync_value]

        # ET sync: must have et_sync_time
        if "et_sync_time" not in et_event_dict:
            raise ValueError(f"'et_sync_time' not in ET .fif. Available: {list(et_event_dict.keys())}")




        #eeg_sync_time is missing in original pipeline... here 
        


        
        # Get sync time values
        #eeg_sync_time_value = eeg_event_dict['eeg_sync_time']
        #et_sync_time_value = et_event_dict['et_sync_time']
        
        # Filter rows where event matches sync_time value
        #eeg_syncs = eeg_events[eeg_events[:, 2] == eeg_sync_time_value]
        #et_syncs = et_events[et_events[:, 2] == et_sync_time_value]

        # Convert event sample index to time (seconds)
        eeg_sync_times = eeg_syncs[:, 0] / eeg_filt_raw.info['sfreq']
        et_sync_times = et_syncs[:, 0] / et_raw.info['sfreq']


 
        print(f"EEG sync points: {len(eeg_sync_times)}")
        print(f"ET sync points: {len(et_sync_times)}")
        if len(eeg_sync_times) != len(et_sync_times):
            raise ValueError(
                f"EEG/ET sync count mismatch: "
                f"EEG={len(eeg_sync_times)}, ET={len(et_sync_times)}"
            )

        # Combine EEG and ET
        #print("before:", np.nanmax(np.abs(et_raw.get_data("pupil_left"))))
        eeg_sync_raw, et_raw = eeg_et_combine(
            eeg_filt_raw, et_raw, eeg_sync_times, et_sync_times,
            eeg_events, eeg_event_dict, et_events, et_event_dict,
        )
        #print("after:", np.nanmax(np.abs(et_raw.get_data("pupil_left"))))
    else:
        eeg_sync_raw = eeg_filt_raw
    # One line to verify sync added ET data
    print(f">>> ET CHANNELS IN RAW: {[ch for ch in eeg_sync_raw.ch_names if 'pupil' in ch.lower() or 'gaze' in ch.lower()]}")

    return (eeg_sync_raw,eeg_events, eeg_event_dict, et_events, et_event_dict)


@app.cell
def plot_channel_groups(eeg_sync_raw, et_sync, mne):
    if et_sync:
        _frontal_ch = ["E11"]
        _occipital_ch = ["E62"]
        _din_ch = ["DIN"]

        _pupil_name_sync = next(
            (ch for ch in ("pupil_left", "pupil_right") if ch in eeg_sync_raw.ch_names),
            None,
        )
        _pupil_ch_sync = [_pupil_name_sync] if _pupil_name_sync is not None else []

        _scale_dict_sync = dict(eeg=1e-4, misc=1e3, pupil=1e3)

        _picks_idx_sync = mne.pick_channels(
            eeg_sync_raw.ch_names,
            _din_ch + _frontal_ch + _occipital_ch + _pupil_ch_sync,
            ordered=True,
        )

        eeg_sync_raw.plot(
            start=0,
            duration=20,
            order=_picks_idx_sync,
            scalings=_scale_dict_sync,
        )

    return

@app.cell
def apply_lossless(apply_ll, bids_ll_path, ll_state, eeg_sync_raw):
    eeg_loss_raw = apply_ll(bids_ll_path, ll_state, eeg_sync_raw)
    print(f"\n>>> FINAL CHANNELS: {eeg_loss_raw.ch_names}")
    print(f">>> PUPIL PRESENT: {any('pupil' in ch for ch in eeg_loss_raw.ch_names)}")
    return (eeg_loss_raw,)



@app.cell
def plot_channel_groups_lossless(eeg_loss_raw, et_sync, mne):
    if et_sync:
        _frontal_ch2 = ["E11"]
        _occipital_ch2 = ["E62"]
        _din_ch2 = ["DIN"]

        _pupil_name_loss = next(
            (ch for ch in ("pupil_left", "pupil_right") if ch in eeg_loss_raw.ch_names),
            None,
        )
        _pupil_ch_loss = [_pupil_name_loss] if _pupil_name_loss is not None else []

        _scale_dict_loss = dict(eeg=1e-4, misc=1e3, pupil=1e3)
        _plot_ch_loss = [
            ch for ch in (_din_ch2 + _frontal_ch2 + _occipital_ch2 + _pupil_ch_loss)
            if ch in eeg_loss_raw.ch_names
        ]

        _picks_idx_loss = mne.pick_channels(
            eeg_loss_raw.ch_names,
            _plot_ch_loss,
            ordered=True,
        )

        eeg_loss_raw.plot(
            start=0,
            duration=20,
            order=_picks_idx_loss,
            scalings=_scale_dict_loss,
        )

    return


@app.cell
def save_output(mne, eeg_loss_raw, write_bids_eeg, subject_id,
                session_id, task_id, project_path, Path, run_id):
    '''
    eeg_loss_raw_save = eeg_loss_raw.copy()  # don't mutate the shared object
    mapping = {ch: "misc" for ch, ct in zip(eeg_loss_raw_save.ch_names,
               eeg_loss_raw_save.get_channel_types()) if ct in ("eyegaze","pupil")}
    if mapping:
        eeg_loss_raw_save.set_channel_types(mapping)
    # use eeg_loss_raw_save for everything below in this cell'''


    
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
    loss_path = str(Path(project_path) / sync_loss_path / task_id)
    # loss_path = project_path + pylossless_path + sync_loss_path

    eeg_bids_path = write_bids_eeg(
        eeg_loss_raw, eeg_loss_events, eeg_loss_event_dict,
        subject_id, session_id, task_id, loss_path, run_id
        
    )
    return (eeg_bids_path,)


@app.cell
def plot_eeg_events(eeg_events, eeg_event_dict, eeg_filt_raw):
    
    figo = px.scatter(x=eeg_events[:,0], y=eeg_events[:,2])
    figo.update_layout(title='Original EEG event times')
    figo.update_xaxes(title_text='Time of event (ms)')
    figo.update_yaxes(title_text='Event index')
    figo.show()
    
    # Also save to file
    figo.write_html("eeg_event_times.html")
    return

@app.cell
def plot_et_events(et_sync, et_events):
    
    if et_sync:
        figt = px.scatter(x=et_events[:,0], y=et_events[:,2])
        figt.update_layout(title='Original ET event times')
        figt.update_xaxes(title_text='Time of event (ms)')
        figt.update_yaxes(title_text='Event index')
        figt.show()
        
        # Also save to file
        figt.write_html("et_event_times.html")
    return


if __name__ == "__main__":
    app.run()
