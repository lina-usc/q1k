import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium")


@app.cell
def parameters():
    # __Q1K_PARAMETERS__
    # The above comment is replaced by the CLI with actual values.
    # For interactive use, set your parameters here:
    project_path = ""
    task_id_in = ""
    task_id_in_et = ""
    task_id_out = ""
    subject_id = ""
    session_id = "01"
    run_id = "1"
    site_code = "HSJ"
    return (project_path, task_id_in, task_id_in_et, task_id_out,
            subject_id, session_id, run_id, site_code)


@app.cell
def imports():
    import warnings
    from pathlib import Path

    import mne
    import mne_bids
    import numpy as np
    import plotly.express as px
    import plotly.graph_objects as go
    warnings.filterwarnings("ignore")

    from q1k.config import EOG_CHANNELS, NO_DIN_OFFSET_TASKS
    from q1k.init.tools import (
        eeg_et_combine,
        eeg_event_test,
        et_event_test,
        get_event_dict,
        show_sync_offsets,
    )
    return (mne, mne_bids, np, px, go, Path, warnings,
            get_event_dict, eeg_event_test, et_event_test,
            show_sync_offsets, eeg_et_combine, EOG_CHANNELS,
            NO_DIN_OFFSET_TASKS)


@app.cell
def setup_paths(project_path, subject_id, session_id, task_id_in,
                task_id_out, Path):
    import marimo as mo

    # Map task to event dict offset and DIN strings
    task_config = {
        "AEP": {"offset": 1, "din_str": ("DIN4", "DIN4")},
        "ap": {"offset": 1, "din_str": ("DIN4", "DIN4")},
        "as": {"offset": 1, "din_str": ("DIN2", "DIN2")},
        "TO": {"offset": 1, "din_str": ("DIN2", "DIN2")},
        "go": {"offset": 1, "din_str": ("DIN2", "DIN2")},
        "GO": {"offset": 1, "din_str": ("DIN2", "DIN2")},
        "mn": {"offset": 1, "din_str": ("DIN4", "DIN4")},
        "vp": {"offset": 0, "din_str": ("DIN2", "DIN3")},
        "VEP": {"offset": 0, "din_str": ("DIN2", "DIN3")},
        "RSRio": {"offset": 1, "din_str": ("DIN2", "DIN2")},
        "rest": {"offset": 1, "din_str": ("DIN2", "DIN2")},
        "RS": {"offset": 1, "din_str": ("DIN2", "DIN2")},
    }

    cfg = task_config.get(task_id_out, {"offset": 1, "din_str": ("DIN2", "DIN2")})
    event_dict_offset = cfg["offset"]
    din_str = cfg["din_str"]

    pp = Path(project_path)
    subject_session = f"{subject_id}_{session_id[1]}"
    session_path_eeg = pp / "sourcefiles" / subject_session / f"{subject_session}_eeg"
    session_file_name_eeg = list(session_path_eeg.glob(f"*_{task_id_in}_*.mff"))

    et_dir = f"{subject_id}_eyetracking_{session_id[1]}"
    session_path_et = pp / "sourcefiles" / subject_session / et_dir
    session_file_name_et = list(session_path_et.glob(f"*_{task_id_in}_*.asc"))

    mo.md(f"## Q1K Init Report: {subject_id} - {task_id_out}")
    return (event_dict_offset, din_str, session_file_name_eeg,
            session_file_name_et, pp, mo)


@app.cell
def read_eeg(mne, session_file_name_eeg, event_dict_offset, get_event_dict):
    raw = mne.io.read_raw_egi(session_file_name_eeg[0])
    eeg_events = mne.find_events(raw, shortest_event=1)
    eeg_event_dict = get_event_dict(raw, eeg_events, event_dict_offset)
    return raw, eeg_events, eeg_event_dict


@app.cell
def plot_events(px, eeg_events):
    fig = px.scatter(
        x=eeg_events[:, 0], y=eeg_events[:, 2],
        title="Original EEG event times",
        labels={"x": "Time (ms)", "y": "Event index"},
    )
    fig
    return (fig,)


@app.cell
def process_events(eeg_event_test, eeg_events, eeg_event_dict, din_str,
                    task_id_out, NO_DIN_OFFSET_TASKS):
    if task_id_out in NO_DIN_OFFSET_TASKS:
        import numpy as _np
        print(f"{task_id_out}: skipping stimulus DIN event offset")
        eeg_events_processed = eeg_events
        eeg_stims = _np.empty((0, 3))
        eeg_iti = _np.array([])
        eeg_din_offset = []
        eeg_event_dict_updated = eeg_event_dict
        new_events = _np.empty((0, 3))
    else:
        (eeg_events_processed, eeg_stims, eeg_iti,
         eeg_din_offset, eeg_event_dict_updated,
         new_events) = eeg_event_test(
            eeg_events, eeg_event_dict, din_str,
            task_name=task_id_out,
        )
    return (eeg_events_processed, eeg_stims, eeg_iti,
            eeg_din_offset, eeg_event_dict_updated, new_events)


@app.cell
def plot_stim_iti(px, eeg_stims, eeg_iti, task_id_out,
                  NO_DIN_OFFSET_TASKS):
    if task_id_out in NO_DIN_OFFSET_TASKS:
        print(f"{task_id_out}: skipping stimulus DIN ITI display")
        fig = None
    else:
        fig = px.scatter(
            x=eeg_stims[1:, 0], y=eeg_iti,
            title="Stimulus DIN Inter-Trial Intervals",
            labels={"x": "Time (ms)", "y": "ITI (ms)"},
        )
        fig
    return (fig,)


@app.cell
def write_bids(mne, mne_bids, raw, eeg_events_processed,
               eeg_event_dict_updated, subject_id, session_id,
               task_id_out, pp, NO_DIN_OFFSET_TASKS):
    raw.info["line_freq"] = 60
    raw.info["device_info"]["type"] = (
        raw.info["device_info"]["type"].replace(" ", "_")
    )

    # RSRio retains stim channels; other tasks remove them
    if task_id_out in NO_DIN_OFFSET_TASKS:
        print("Retaining stim channels for RSRio task...")
    else:
        stim_chs = [
            ch for ch, ct in zip(
                raw.info["ch_names"],
                raw.get_channel_types(),
            )
            if ct == "stim"
        ]
        if stim_chs:
            raw.drop_channels(stim_chs)

    bids_path = mne_bids.BIDSPath(
        subject=subject_id, session=session_id,
        task=task_id_out,
        run="1", datatype="eeg", root=str(pp),
    )

    mne_bids.write_raw_bids(
        raw=raw, bids_path=bids_path,
        events=eeg_events_processed,
        event_id=eeg_event_dict_updated,
        format="EDF", overwrite=True,
    )
    return (bids_path,)


if __name__ == "__main__":
    app.run()
