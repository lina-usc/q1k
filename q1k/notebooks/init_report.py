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
                task_id_out, Path, site_code):
    import marimo as mo

    # Map task to event dict offset and DIN strings
    task_config = {
        "AEP": {"offset": 1, "din_str": ("DIN4", "DIN4")},
        "ap": {"offset": 1, "din_str": ("DIN4", "DIN4")},
        "as": {"offset": 1, "din_str": ("DIN2", "DIN2")},
        "TO": {"offset": 1, "din_str": ("DIN2", "DIN2")},
        "go": {"offset": 1, "din_str": ("DIN4", "DIN5")},
        "GO": {"offset": 1, "din_str": ("DIN4", "DIN5")},
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
    subject_versions = [subject_id, subject_id.replace('_', '')]
    session_path_eeg = None
    for subj_version in subject_versions:
       session_path_eeg = pp / "sourcedata"/site_code/"eeg" /subj_version # subject_session / f"{subject_session}_eeg"
       break
    if session_path_eeg is None:
       raise FileNotFoundError(f"Could not find source for {subject_id}")
    session_file_name_eeg = [d for d in session_path_eeg.iterdir() 
                             if d.is_dir() and d.name.endswith('.mff') and task_id_in in d.name]

    #session_file_name_eeg = list(session_path_eeg.glob(f"*_{task_id_in}_*.mff"))

    #et_dir = f"{subject_id}_eyetracking_{session_id[1]}"
    #session_path_et = pp / "sourcedata"/site_code/"et"/subject_id # / subject_session / et_dir
    #session_file_name_et = list(session_path_et.glob(f"*_{task_id_in}*.asc"))
    # First try: direct lookup in the subject's ET directory
    session_path_et = pp / "sourcedata"/site_code/"et"/subject_id
    session_file_name_et = list(session_path_et.glob(f"*_{task_id_in}*.asc"))
    # Second try: use mapping CSV for subjects where direct lookup fails
    if not session_file_name_et:
        import csv
        mapping_file = pp / "et_mapping_lookup.csv"
        if mapping_file.exists():
            # Extract core subject ID (e.g., "1525-1212_P" from "Q1K_HSJ_1525-1212_P")
            subject_core = '_'.join(subject_id.split('_')[2:]) if len(subject_id.split('_')) >= 3 else subject_id
            with open(mapping_file, 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    if row and not row[0].startswith('#'):
                        eeg_id, et_filename = row[0].strip(), row[1].strip()
                        if eeg_id == subject_core:
                            # Search for this ET file in all ET directories
                            et_base = pp / "sourcedata"/site_code/"et"
                            if et_base.exists():
                                for et_dir in et_base.iterdir():
                                    matches = list(et_dir.glob(et_filename))
                                    if matches:
                                        session_file_name_et = matches
                                        print(f"✓ ET found via mapping: {matches[0].name}")
                                        break
                            break
    mo.md(f"## Q1K Init Report: {subject_id} - {task_id_out}")
    return (event_dict_offset, din_str, session_file_name_eeg,
            session_file_name_et, pp, mo)


@app.cell
def read_eeg(mne, session_file_name_eeg, event_dict_offset, get_event_dict):
    raw = mne.io.read_raw_egi(session_file_name_eeg[0])
    eeg_events = mne.find_events(raw, shortest_event=1)
    eeg_event_dict = get_event_dict(raw, eeg_events, event_dict_offset)
    # Make sure all events have descriptions
    unique_events = set(eeg_events[:, 2])
    for event_id in unique_events:
        if event_id not in eeg_event_dict.values():
            # Add missing event with generic name
            eeg_event_dict[f'event_{int(event_id)}'] = int(event_id)
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
        fig_stim = None
    else:
        fig_stim = px.scatter(
            x=eeg_stims[1:, 0], y=eeg_iti,
            title="Stimulus DIN Inter-Trial Intervals",
            labels={"x": "Time (ms)", "y": "ITI (ms)"},
        )
        fig_stim
    return (fig_stim,)


@app.cell
def write_bids(mne, mne_bids, raw, eeg_events_processed,
               eeg_event_dict_updated, subject_id, session_id,
               task_id_out, pp, NO_DIN_OFFSET_TASKS):
    import re
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
    # Extract numeric part from subject_id (e.g., "Q1K_HSJ_10083_M1" -> "10083M1")
    match = re.search(r'(\d+)_?([A-Z0-9]+)$', subject_id)
    if match:
        bids_subject = match.group(1) + match.group(2)
    else:
        bids_subject = subject_id  # fallback
    print(f"Using BIDS subject ID: {bids_subject}")
    print(f"Using BIDS subject ID: {bids_subject}")
    print(f"Session ID: {session_id}")
    print(f"Task: {task_id_out}")
    print(f"Root path: {pp}")
    print(f"Events processed: {len(eeg_events_processed) if eeg_events_processed is not None else 'None'}")   
    bids_path = mne_bids.BIDSPath(
        subject=bids_subject, session=session_id,
        task=task_id_out,
        run="1", datatype="eeg", root=str(pp/"derivatives"/"init"),
    )

    mne_bids.write_raw_bids(
        raw=raw, bids_path=bids_path,
        events=eeg_events_processed,
        event_id=eeg_event_dict_updated,
        format="EDF", overwrite=True,
    )
    # Strip BOM from all TSV files written by mne_bids (known mne-bids bug)
    import pandas as pd
    bids_dir = pp /"derivatives"/"init"/ f"sub-{bids_subject}" / f"ses-{session_id}" / "eeg"
    for tsv_file in bids_dir.glob("*.tsv"):
        df_tsv = pd.read_csv(tsv_file, sep='\t', encoding='utf-8-sig')
        df_tsv.to_csv(tsv_file, sep='\t', index=False, encoding='utf-8')
    return (bids_path,)



@app.cell
def read_et(mne, session_file_name_et, task_id_out):
    ET_TASKS = {"VEP", "GO", "PLR", "VS", "NSP"}
    et_sync = task_id_out in ET_TASKS
    if et_sync and session_file_name_et:
        from q1k.init.tools import et_read, et_clean_events
        et_raw, et_raw_df, et_annot_events, et_annot_event_dict = et_read(
            str(session_file_name_et[0]), blink_interp=False, fill_nans=False, resamp=False,
        )
        et_annot_event_dict, et_annot_events = et_clean_events(et_annot_event_dict, et_annot_events)
        print(f"ET loaded: {session_file_name_et[0]}")
        print(f"ET annotations: {list(et_annot_event_dict.keys())}")
    else:
        et_raw = et_raw_df = et_annot_events = et_annot_event_dict = None
        print(f"ET sync disabled for task: {task_id_out}")
    return et_sync, et_raw, et_raw_df, et_annot_events, et_annot_event_dict


@app.cell
def process_et_events(et_sync, et_raw, et_raw_df, et_annot_events,
                      et_annot_event_dict, task_id_out):
    import numpy as _np
    if et_sync and et_raw is not None:
        task_key = "vp" if task_id_out == "VEP" else task_id_out.lower()
        _, et_events_out, et_stims_out, et_iti_out = et_event_test(et_raw_df, task_name=task_key)
        print(f"ET stimulus events: {len(et_stims_out)}")
    else:
        et_events_out = et_stims_out = et_iti_out = None
    return et_events_out, et_stims_out, et_iti_out


@app.cell
def align_and_save_et(et_sync, et_raw, et_annot_events, et_annot_event_dict,
                      et_stims_out, eeg_events_processed, eeg_event_dict_updated,
                      raw, task_id_out, bids_path):
    import numpy as _np
    import mne as _mne
    if et_sync and et_raw is not None and et_stims_out is not None and len(et_stims_out) > 0:
        from q1k.init.tools import eeg_et_align, et_events_to_annot, write_et

        # Get EEG events from annotations (stim channels dropped during BIDS write)
        eeg_events_ann, eeg_event_dict_ann = _mne.events_from_annotations(raw)

        # Extract GO stimulus events (dtoc_d, dtbc_d, dtgc_d)
        stim_keys = [k for k in eeg_event_dict_ann if k in ("dtoc_d", "dtbc_d", "dtgc_d")]
        if not stim_keys:
            # Fallback: use all processed events
            _eeg_stims_align = eeg_events_processed
            eeg_event_dict_use = eeg_event_dict_updated
            eeg_events_use = eeg_events_processed
        else:
            stim_ids = [eeg_event_dict_ann[k] for k in stim_keys]
            _eeg_stims_align = eeg_events_ann[_np.isin(eeg_events_ann[:, 2], stim_ids)]
            eeg_event_dict_use = eeg_event_dict_ann
            eeg_events_use = eeg_events_ann

        print(f"EEG stims for alignment: {len(_eeg_stims_align)}")

        et_sfreq = et_raw.info["sfreq"]
        et_stims_np = _np.column_stack([
            (et_stims_out["time"].values * et_sfreq).astype(int),
            _np.zeros(len(et_stims_out), dtype=int),
            _np.zeros(len(et_stims_out), dtype=int),
        ])

        (eeg_ed, et_ed, eeg_ev, et_ev,
         eeg_times, et_times) = eeg_et_align(
            eeg_event_dict_use.copy(), et_annot_event_dict.copy(),
            eeg_events_use.copy(), et_annot_events.copy(),
            _eeg_stims_align, et_stims_np,
            raw.info["sfreq"], et_sfreq, task_id_out,
        )
        print(f"Aligned: eeg={len(eeg_times)}, et={len(et_times)}")
        et_raw_annot = et_events_to_annot(et_raw, et_ed, et_ev)
        et_fif_path  = write_et(et_raw_annot, bids_path.fpath)
        print(f"ET .fif saved: {et_fif_path}")
    else:
        print("Skipping ET .fif save.")

if __name__ == "__main__":
    app.run()
