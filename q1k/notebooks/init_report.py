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
    subject_id_out = ""
    session_id = "01"
    run_id = "1"
    site_code = "HSJ"
    return (project_path, task_id_in, task_id_in_et, task_id_out,
            subject_id, subject_id_out, session_id, run_id, site_code)


@app.cell
def imports():
    import warnings
    from pathlib import Path

    import mne
    import mne_bids
    import marimo as mo
    import numpy as np
    import plotly.express as px
    import plotly.io as pio
    import plotly.graph_objects as go
    pio.renderers.default = "plotly_mimetype+notebook"
    warnings.filterwarnings("ignore")

    import q1k.init.tools as qit
    #import tools as qit

    return mne, mne_bids, np, px, go, Path, warnings, pio, qit, mo


@app.cell
def convert_edf_to_asc(project_path, site_code, subject_id, session_id,
                       task_id_in, Path):
    import eyelinkio
    import shutil
    pp = Path(project_path)
    asc_out = None
    ET_TASKS_INIT = {"GO", "PLR", "VS", "NSP"}

    if task_id_in.upper() not in ET_TASKS_INIT:
        print(f"Task {task_id_in} does not use eye-tracking — skipping EDF conversion.")
    else:
        et_dir = pp / "source_prime" / site_code / subject_id / "et" / task_id_in.upper()
        print(f"Looking for ET data in: {et_dir}")
        edf_file = None
        if et_dir.exists():
            edf_filey = list(et_dir.glob(f"{subject_id}_{task_id_in.upper()}.edf"))
            
    
            if edf_filey:
                edf_file = edf_filey[0] 
                print(f"Found EDF: {edf_file}")
                asc_out = et_dir / f"{edf_file.stem}.asc"
                #dest_dir = pp / "sourcedata" / site_code / "et" / subject_id
                #dest_dir.mkdir(parents=True, exist_ok=True)
                #asc_out = dest_dir / f"{edf_file.stem}.asc"
                
                if not asc_out.exists():
                    print(f"Converting to ASC from {edf_file}")
                    edf_data = eyelinkio.read_edf(str(edf_file))
                    edf_data.to_asc(str(asc_out))
                    print(f"Saved: {asc_out}")
                else:
                    print(f"ASC already exists: {asc_out}")
            else:
                print(f"No EDF found for {subject_id}_{task_id_in.upper()}.edf — ET data unavailable.")
        else:
            print(f"ET directory does not exist: {et_dir}")
            
    return (asc_out,)


@app.cell
def setup_paths(project_path, subject_id, session_id, task_id_in,
                task_id_out, site_code, qit, mo):
    print(f"DEBUG: Original subject_id = '{subject_id}'")
    din_str, event_dict_offset = qit.set_din_str(task_id_out)
    print("Initial DIN strings for " + task_id_out)
    print(din_str)

    subject_id_numeric = subject_id
    subject_id_out_bids = subject_id
    print(f"DEBUG: subject_id_numeric = '{subject_id_numeric}'")
    print(f"DEBUG: subject_id_out_bids = '{subject_id_out_bids}'")

    session_file_name_eeg, session_file_name_et = qit.generate_session_ids(
        "experimental", project_path, site_code, task_id_in, subject_id_numeric, "1"
    )

    print("EEG session file name: " + session_file_name_eeg[0] if session_file_name_eeg else "None")

    #base_source = Path(project_path) / "source_prime" / site_code / subject_id_numeric
    #eeg_task_path = base_source / "eeg" / task_id_in.upper()
    #session_file_name_eeg = []


    
    #session_file_name_eeg, session_file_name_et = qit.generate_session_ids(
    #    "experimental", project_path, site_code, task_id_in, subject_id_numeric, "1"
    #)
    '''
    if eeg_task_path.exists():
        # Look for .mff directories (they are folders, not files)
        mff_dirs = list(eeg_task_path.glob(f"{subject_id_numeric}_*.mff"))
        if mff_dirs:
            session_file_name_eeg = [str(mff_dirs[0])]
            print(f"Found EEG: {session_file_name_eeg[0]}")
        else:
            print(f"No EEG .mff directory found in {eeg_task_path}")
    else:
        print(f"EEG task path does not exist: {eeg_task_path}")

    #ET_TASKS = {"VEP", "GO", "PLR", "VS", "NSP"}
    #session_file_name_et = []
    
    if task_id_in.upper() in ET_TASKS:
        et_task_path = base_source / "et" / task_id_in.upper()
        if et_task_path.exists():
            edf_files = list(et_task_path.glob(f"{subject_id_numeric}_{task_id_in.upper()}.edf"))
            if edf_files:
                session_file_name_et = [str(edf_files[0])]
                print(f"Found ET: {session_file_name_et[0]}")
            else:
                print(f"No ET .edf file found in {et_task_path}")
        else:
            print(f"ET task path does not exist: {et_task_path}")
    else:
        print(f"Task {task_id_in} does not use eye-tracking") '''

    
    

    if session_file_name_et and session_file_name_et[0] if isinstance(session_file_name_et, list) else session_file_name_et:
        print("ET session file name: " + session_file_name_et[0])
    else:
        print("Could not find the session ET file.. abandoning ET sync portion of the initiation process.")

    mo.md(f"Q1K Init Report: **{subject_id}** — {task_id_out}")
    return event_dict_offset, session_file_name_eeg, session_file_name_et, mo, subject_id_out_bids


@app.cell
def set_et_sync(task_id_out):
    if task_id_out in {'GO', 'NSP', 'PLR', 'VS'}:
        et_sync = True
        print('init stage proceeds with et_sync enabled...')
    else:
        et_sync = False
        print('init stage proceeds with et_sync disabled...')
    return (et_sync,)


@app.cell
def read_eeg(mne, session_file_name_eeg, event_dict_offset, qit):
    if not session_file_name_eeg:
        raise FileNotFoundError(
            "No EEG .mff files found for this subject/task combination."
        )

    print('Reading: ' + session_file_name_eeg[0])
    eeg_raw = mne.io.read_raw_egi(session_file_name_eeg[0],preload=True)
    eeg_raw_fresh = eeg_raw.copy()
    device_info = eeg_raw.info['device_info']

    eeg_channel_types = eeg_raw.get_channel_types()
    print("EEG Channel Types:", eeg_channel_types)
    print("EEG Channel Names:", eeg_raw.info['ch_names'])

    eeg_events_raw = mne.find_events(eeg_raw, shortest_event=1)
    eeg_event_dict_raw = qit.get_event_dict(eeg_raw, eeg_events_raw, event_dict_offset)

    return eeg_raw, eeg_raw_fresh, device_info, eeg_events_raw, eeg_event_dict_raw


@app.cell
def plot_events(px, eeg_events_raw):
    fig= None
    if eeg_events_raw is not None and len(eeg_events_raw) > 0:
        fig = px.scatter(
            x=eeg_events_raw[:, 0], y=eeg_events_raw[:, 2],
            title="Original EEG event times",
            labels={"x": "Time (ms)", "y": "Event index"},
        )
        fig.show()
        fig
    else:
         print("No events to plot")
        
    return (fig,) 

'''
@app.cell
def process_events(qit, eeg_events_raw, eeg_event_dict_raw, task_id_out):
    import numpy as _np
    din_strp, _ = qit.set_din_str(task_id_out)
    if task_id_out.upper() == "GO":
        eeg_events_processed, eeg_stims, eeg_iti, eeg_din_offset, eeg_event_dict_updated, new_events = qit.eeg_event_test(eeg_events_raw.copy(), eeg_event_dict_raw.copy(), din_strp, task_name=task_id_out
    )
    else:

    din_strt = qit.din_check(eeg_event_dict_raw, din_strp)
    print(f"process_events received din_str = {din_strt}")
    if not din_strt:
        print('Required EEG DIN events are missing... skipping EEG DIN check')
    else:
        din_diffs, din_diffs_time = qit.get_din_diff(eeg_events_raw, eeg_event_dict_raw, din_strt)
        if not din_diffs:
            din_strt = ()
            print('No din_diffs found... setting din_str to ()')

    if not din_strt:
        print('Required EEG DIN events are missing... skipping EEG stimulus onset DIN process')
        eeg_events_processed = eeg_events_raw
        eeg_stims = _np.empty((0, 3), dtype=int)
        eeg_iti = _np.array([])
        eeg_din_offset = []
        eeg_event_dict_updated = eeg_event_dict_raw
        new_events = _np.empty((0, 3), dtype=int)
    else:
        eeg_events_cleaned, eeg_event_dict_cleaned = qit.eeg_clean_events(eeg_events_raw, eeg_event_dict_raw, din_strt)
        eeg_events_processed, eeg_stims, eeg_iti, eeg_din_offset, eeg_event_dict_updated, new_events = qit.eeg_task_events(
            eeg_events_cleaned, eeg_event_dict_cleaned, din_strt, task_name=task_id_out
        )

    column_values = eeg_events_processed[:, 2]
    unique_values, counts = _np.unique(column_values, return_counts=True)
    print("Counts of each event ID value:")
    for value, count in zip(unique_values, counts):
        print(f"Event ID: {value}, Count: {count}")
    print("Label\tValue")
    for label, value in eeg_event_dict_updated.items():
        print(f"{label}\t{value}")

    return eeg_events_processed, eeg_stims, eeg_iti, eeg_din_offset, eeg_event_dict_updated, new_events
'''


@app.cell
def process_events(qit, eeg_events_raw, eeg_event_dict_raw, task_id_out):
    import numpy as _np

    din_strp, _ = qit.set_din_str(task_id_out)

    # Original researcher GO behavior:
    # Do not run din_check/eeg_clean_events for GO.
    # Let original eeg_event_test handle GO directly.
    if task_id_out.upper() == "GO":
        eeg_events_processed, eeg_stims, eeg_iti, eeg_din_offset, eeg_event_dict_updated, new_events = qit.eeg_event_test(
            eeg_events_raw.copy(),
            eeg_event_dict_raw.copy(),
            din_strp,
            task_name=task_id_out,
        )

    else:
        din_strt = qit.din_check(eeg_event_dict_raw, din_strp)
        print(f"process_events received din_str = {din_strt}")

        if not din_strt:
            print("Required EEG DIN events are missing... skipping EEG DIN check")
        else:
            din_diffs, din_diffs_time = qit.get_din_diff(
                eeg_events_raw,
                eeg_event_dict_raw,
                din_strt,
            )
            if not din_diffs:
                din_strt = ()
                print("No din_diffs found... setting din_str to ()")

        if not din_strt:
            print("Required EEG DIN events are missing... skipping EEG stimulus onset DIN process")
            eeg_events_processed = eeg_events_raw
            eeg_stims = _np.empty((0, 3), dtype=int)
            eeg_iti = _np.array([])
            eeg_din_offset = []
            eeg_event_dict_updated = eeg_event_dict_raw
            new_events = _np.empty((0, 3), dtype=int)
        else:
            eeg_events_cleaned, eeg_event_dict_cleaned = qit.eeg_clean_events(
                eeg_events_raw,
                eeg_event_dict_raw,
                din_strt,
            )
            eeg_events_processed, eeg_stims, eeg_iti, eeg_din_offset, eeg_event_dict_updated, new_events = qit.eeg_task_events(
                eeg_events_cleaned,
                eeg_event_dict_cleaned,
                din_strt,
                task_name=task_id_out,
            )

    column_values = eeg_events_processed[:, 2]
    unique_values, counts = _np.unique(column_values, return_counts=True)

    print("Counts of each event ID value:")
    for value, count in zip(unique_values, counts):
        print(f"Event ID: {value}, Count: {count}")

    print("Label\tValue")
    for label, value in eeg_event_dict_updated.items():
        print(f"{label}\t{value}")

    return eeg_events_processed, eeg_stims, eeg_iti, eeg_din_offset, eeg_event_dict_updated, new_events

@app.cell
def plot_stim_iti(px, eeg_stims, eeg_iti):
    if eeg_stims is None or len(eeg_stims) < 2 or len(eeg_iti) == 0:
        print(
            f"WARNING: Insufficient DIN stimulus events to plot ITI "
            f"(found {len(eeg_stims) if eeg_stims is not None else 0} stim(s), need ≥ 2)."
        )
        fig_stim = None
    else:
        fig_stim = px.scatter(
            x=eeg_stims[1:, 0], y=eeg_iti,
            title="Stimulus DIN Inter-Trial Intervals",
            labels={"x": "Stimulus onset Time (ms)", "y": "ITI (ms)"},
        )
        fig_stim.show()
    return (fig_stim,)






@app.cell
def align_eeg_et(qit, et_sync, et_raw_obj, et_events_processed, et_event_dict_processed,
                 et_stims, eeg_events_processed, eeg_event_dict_updated,
                 eeg_stims, eeg_raw, task_id_out):
    """Align EEG and ET events, return aligned EEG events for BIDS writing."""
    
    # Initialize with original values (for non-ET cases)
    aligned_eeg_event_dict = eeg_event_dict_updated
    aligned_eeg_events = eeg_events_processed
    aligned_et_event_dict = None
    aligned_et_events = None
    eeg_times = None
    et_times = None
    
    if et_sync and et_raw_obj is not None and et_stims is not None and len(et_stims) > 0:
        # Perform alignment
        try: 
            aligned_eeg_event_dict, aligned_et_event_dict, aligned_eeg_events, aligned_et_events, eeg_times, et_times = qit.eeg_et_align(
                eeg_event_dict_updated, et_event_dict_processed,
                eeg_events_processed, et_events_processed,
                eeg_stims, et_stims,
                eeg_raw.info["sfreq"], et_raw_obj.info["sfreq"],
                task_id_out
            )
            print(f"✓ Aligned: eeg={len(eeg_times)} sync points, et={len(et_times)} sync points")
        except Exception as err:
            print(f"EEG-ET alignment failed; continuing EEG-only: {err}")
            aligned_eeg_event_dict = eeg_event_dict_updated
            aligned_eeg_events = eeg_events_processed
            aligned_et_event_dict = None
            aligned_et_events = None
            eeg_times = None
            et_times = None
        
        # Visualization
        if eeg_times is not None and et_times is not None:
            import plotly.graph_objects as goobj
            fig_sync = goobj.Figure()
            fig_sync.add_trace(goobj.Scatter(x=eeg_times, y=et_times, mode='markers'))
            fig_sync.update_layout(
                title="EEG-ET Sync: Stimulus Onset Times",
                xaxis_title="EEG time (ms)",
                yaxis_title="ET time (ms)"
            )
            fig_sync.show()
    else:
        print("⊘ Skipping EEG-ET alignment (no ET data or et_sync=False)")
    
    return (aligned_eeg_event_dict, aligned_eeg_events, 
            aligned_et_event_dict, aligned_et_events, et_raw_obj)















@app.cell
def write_bids(mne, eeg_raw, aligned_eeg_events, aligned_eeg_event_dict, 
               subject_id_out_bids, session_id, task_id_out, project_path, device_info, qit, Path):
    
    stim_channels = [ch_name for ch_name, ch_type in zip(
        eeg_raw.info['ch_names'], eeg_raw.get_channel_types()) if ch_type == 'stim']
    
    print(f"Stim channels to remove: {stim_channels}")
    eeg_raw.drop_channels(stim_channels)
    #derivatives_path = Path(project_path) / "derivatives" / "init" / task_id_out / subject_id_out_bids / f"ses-{session_id}"
    #derivatives_path.mkdir(parents=True, exist_ok=True)
    
    eeg_bids_path = qit.write_eeg(
        eeg_raw,
        aligned_eeg_event_dict,
        aligned_eeg_events,
        subject_id_out_bids,
        session_id,
        task_id_out,
        project_path,
        device_info
    )
    return (eeg_bids_path,)

'''
@app.cell
def read_et(mne, qit, et_sync, session_file_name_et):
    
    if et_sync and session_file_name_et and len(session_file_name_et) > 0:
        print(f"Reading ET from: {session_file_name_et[0]}")
        et_raw_obj, et_raw_df_obj, et_events_raw, et_event_dict_raw = qit.et_read(
            session_file_name_et[0], blink_interp=False, fill_nans=False, resamp=False
        )
        et_channel_types = et_raw_obj.get_channel_types()
        print("ET Channel Types:", et_channel_types)
        print("ET Channel Names:", et_raw_obj.info['ch_names'])
        print("ET event dict:", et_event_dict_raw)
    else:
        et_raw_obj = et_raw_df_obj = et_events_raw = et_event_dict_raw = None
        print("et_sync = False: not reading ET data")
    return et_raw_obj, et_raw_df_obj, et_events_raw, et_event_dict_raw
'''





@app.cell
def read_et(mne, qit, et_sync, session_file_name_et):
    if et_sync and session_file_name_et and len(session_file_name_et) > 0:
        print(f"Reading ET from: {session_file_name_et[0]}")
        try:
            et_raw_obj, et_raw_df_obj, et_events_raw, et_event_dict_raw = qit.et_read(
                session_file_name_et[0], blink_interp=False, fill_nans=False, resamp=False
            )
            et_channel_types = et_raw_obj.get_channel_types()
            print("ET Channel Types:", et_channel_types)
            print("ET Channel Names:", et_raw_obj.info["ch_names"])
            print("ET event dict:", et_event_dict_raw)
        except Exception as err:
            print(f"Could not read ET file; disabling ET sync for this run: {err}")
            et_raw_obj = et_raw_df_obj = et_events_raw = et_event_dict_raw = None
    else:
        et_raw_obj = et_raw_df_obj = et_events_raw = et_event_dict_raw = None
        print("et_sync = False: not reading ET data")

    return et_raw_obj, et_raw_df_obj, et_events_raw, et_event_dict_raw
'''
@app.cell
def process_et_events(qit, et_sync, et_raw_df_obj, et_events_raw, et_event_dict_raw, task_id_out):
    din_stret, _ = qit.set_din_str(task_id_out)
    if et_sync and et_raw_df_obj is not None:
        et_event_dict_cleaned, et_events_cleaned = qit.et_clean_events(et_event_dict_raw, et_events_raw)
        et_event_dict_processed, et_events_processed, et_raw_df_processed = qit.et_task_events(
            et_raw_df_obj, et_event_dict_cleaned, et_events_cleaned, task_id_out, din_stret
        )
        print("updated ET event dict:", et_event_dict_processed)
        stim_d_value = et_event_dict_processed['STIM_d']
        et_stims = et_events_processed[et_events_processed[:, 2] == stim_d_value]
        print('Number of stimulus onset DIN events: ' + str(len(et_stims)))
    else:
        et_stims = None
        et_events_processed = None
        et_event_dict_processed = None
        print("et_sync = False: not processing ET events")
    return et_stims, et_events_processed, et_event_dict_processed
'''

@app.cell
def process_et_events(qit, et_sync, et_raw_obj, et_raw_df_obj, et_events_raw, et_event_dict_raw, task_id_out):
    import numpy as _np

    et_stims = None
    et_events_processed = None
    et_event_dict_processed = None

    if et_sync and et_raw_df_obj is not None:
        try: 
            et_event_dict_cleaned, et_events_cleaned = qit.et_clean_events(
                et_event_dict_raw,
                et_events_raw,
            )
    
            et_raw_df_obj["DIN"] = et_raw_df_obj["DIN"].fillna(0)
            et_din_events = et_raw_df_obj.loc[et_raw_df_obj["DIN"].diff() > 0]
    
            if len(et_din_events) == 0:
                print(f"{task_id_out} ET data loaded, but no ET DIN/button events were found; skipping ET alignment.")
                et_events_processed = et_events_cleaned
                et_event_dict_processed = et_event_dict_cleaned
    
            elif task_id_out.upper() == "GO":
                _, et_events_out, et_stims_out, et_iti_out = qit.et_event_test(
                    et_raw_df_obj.copy(),
                    task_name="go",
                )
    
                print("Original GO ET stimulus events: " + str(len(et_stims_out)))
    
                et_event_dict_processed = et_event_dict_cleaned
                et_events_processed = et_events_cleaned
    
                if len(et_stims_out) == 0:
                    print("GO ET data loaded, but original GO parser found no reliable ET sync stims; skipping ET alignment.")
                    et_stims = None
                else:
                    et_stims = _np.column_stack([
                        (et_stims_out["time"].values * et_raw_obj.info["sfreq"]).astype(int),
                        _np.zeros(len(et_stims_out), dtype=int),
                        _np.zeros(len(et_stims_out), dtype=int),
                    ])
            elif task_id_out.upper() == "VEP":
                _, et_events_out, et_stims_out, et_iti_out = qit.et_event_test(
                    et_raw_df_obj.copy(),
                    task_name="vp",
                )
            
                print("Original VEP ET stimulus events: " + str(len(et_stims_out)))
            
                et_event_dict_processed = et_event_dict_cleaned
                et_events_processed = et_events_cleaned
            
                if len(et_stims_out) == 0:
                    print("VEP ET data loaded, but original VEP parser found no reliable ET sync stims; skipping ET alignment.")
                    et_stims = None
                else:
                    et_stims = _np.column_stack([
                        et_stims_out["index"].values.astype(int),
                        _np.zeros(len(et_stims_out), dtype=int),
                        _np.zeros(len(et_stims_out), dtype=int),
                    ])
    
            else:
                din_stret, _ = qit.set_din_str(task_id_out)
                et_event_dict_processed, et_events_processed, et_raw_df_processed = qit.et_task_events(
                    et_raw_df_obj,
                    et_event_dict_cleaned,
                    et_events_cleaned,
                    task_id_out,
                    din_stret,
                )
                print("updated ET event dict:", et_event_dict_processed)
                stim_d_value = et_event_dict_processed["STIM_d"]
                et_stims = et_events_processed[et_events_processed[:, 2] == stim_d_value]
                print("Number of stimulus onset DIN events: " + str(len(et_stims)))
        except Exception as err:
            print(f"ET event processing failed; continuing EEG-only: {err}")
            et_stims = None
            et_events_processed = None
            et_event_dict_processed = None
            

    else:
        print("et_sync = False: not processing ET events")

    return et_stims, et_events_processed, et_event_dict_processed



'''
@app.cell
def process_et_events(qit, et_sync, et_raw_obj, et_raw_df_obj, et_events_raw, et_event_dict_raw, task_id_out):
    import numpy as _np

    if et_sync and et_raw_df_obj is not None:
        et_event_dict_cleaned, et_events_cleaned = qit.et_clean_events(
            et_event_dict_raw,
            et_events_raw,
        )
        et_raw_df_obj["DIN"] = et_raw_df_obj["DIN"].fillna(0)
        et_din_events = et_raw_df_obj.loc[et_raw_df_obj["DIN"].diff() > 0]

        if len(et_din_events) == 0:
            print(f"{task_id_out} ET data loaded, but no ET DIN/button events were found; skipping ET alignment.")
            et_stims = None
            et_events_processed = et_events_cleaned
            et_event_dict_processed = et_event_dict_cleaned
            return et_stims, et_events_processed, et_event_dict_processed

        if task_id_out.upper() == "GO":
            _, et_events_out, et_stims_out, et_iti_out = qit.et_event_test(
                et_raw_df_obj.copy(),
                task_name="go",
            )

            print("Original GO ET stimulus events: " + str(len(et_stims_out)))

            et_event_dict_processed = et_event_dict_cleaned
            et_events_processed = et_events_cleaned

            if len(et_stims_out) == 0:
                print("GO ET data loaded, but original GO parser found no reliable ET sync stims; skipping ET alignment.")
                et_stims = None
            else:
                et_stims = _np.column_stack([
                    (et_stims_out["time"].values * et_raw_obj.info["sfreq"]).astype(int),
                    _np.zeros(len(et_stims_out), dtype=int),
                    _np.zeros(len(et_stims_out), dtype=int),
                ])

        else:
            din_stret, _ = qit.set_din_str(task_id_out)
            et_event_dict_processed, et_events_processed, et_raw_df_processed = qit.et_task_events(
                et_raw_df_obj,
                et_event_dict_cleaned,
                et_events_cleaned,
                task_id_out,
                din_stret,
            )
            print("updated ET event dict:", et_event_dict_processed)
            stim_d_value = et_event_dict_processed["STIM_d"]
            et_stims = et_events_processed[et_events_processed[:, 2] == stim_d_value]
            print("Number of stimulus onset DIN events: " + str(len(et_stims)))

    else:
        et_stims = None
        et_events_processed = None
        et_event_dict_processed = None
        print("et_sync = False: not processing ET events")

    return et_stims, et_events_processed, et_event_dict_processed
'''

@app.cell
def save_et(qit, et_sync, et_raw_obj, aligned_et_event_dict, 
            aligned_et_events, eeg_bids_path):
    """Save aligned ET data as .fif file (runs after EEG BIDS is written)."""
    
    if et_sync and et_raw_obj is not None and aligned_et_event_dict is not None:
        # Convert ET events to annotations
        try:
            et_raw_annot = qit.et_events_to_annot(
                et_raw_obj, 
                aligned_et_event_dict, 
                aligned_et_events
            )
            
            # Save ET .fif file
            et_out_path = qit.write_et(et_raw_annot, eeg_bids_path)
            print(f"✓ ET .fif saved: {et_out_path}")
        except Exception as err:
            print(f"ET save failed; EEG init output remains valid: {err}")
    else:
        print("⊘ Skipping ET save (no ET data or et_sync=False)")
    
    return ()









if __name__ == "__main__":
    app.run()



#def write_bids(mne, eeg_raw, eeg_events_processed, eeg_event_dict_updated, 
#               subject_id_out_bids, session_id, task_id_out, project_path, device_info, qit, Path):


'''

@app.cell
def align_and_save_et(qit, et_sync, et_raw_obj, et_events_processed, et_event_dict_processed,
                      et_stims, eeg_events_processed, eeg_event_dict_updated,
                      eeg_stims, eeg_raw, task_id_out, eeg_bids_path):
    #aligned_eeg_event_dict = eeg_event_dict_updated
    #aligned_eeg_events = eeg_events_processed
    if et_sync and et_raw_obj is not None and et_stims is not None and len(et_stims) > 0:
        aeeg_event_dict_updated, aet_event_dict_processed, aeeg_events_processed, aet_events_processed, eeg_times, et_times = qit.eeg_et_align(
            eeg_event_dict_updated, et_event_dict_processed,
            eeg_events_processed, et_events_processed,
            eeg_stims, et_stims,
            eeg_raw.info["sfreq"], et_raw_obj.info["sfreq"],
            task_id_out
        )
        aligned_eeg_event_dict, aet_event_dict_processed, aligned_eeg_events, aet_events_processed, eeg_times, et_times = qit.eeg_et_align(
            eeg_event_dict_updated, et_event_dict_processed,
            eeg_events_processed, et_events_processed,
            eeg_stims, et_stims,
            eeg_raw.info["sfreq"], et_raw_obj.info["sfreq"],
            task_id_out
        )
        print(f"Aligned: eeg={len(eeg_times)}, et={len(et_times)}")
        et_raw_annot = qit.et_events_to_annot(et_raw_obj, aet_event_dict_processed, aet_events_processed)
        et_out_path = qit.write_et(et_raw_annot, eeg_bids_path)
        print(f"ET .fif saved: {et_out_path}")
    else:
        print("Skipping ET sync and save")
    if 'eeg_times' in locals() and 'et_times' in locals():
        import plotly.graph_objects as goobj
        fig_sync = goobj.Figure()
        fig_sync.add_trace(goobj.Scatter(x=eeg_times, y=et_times, mode='markers'))
        fig_sync.update_layout(
            title="EEG-ET Sync: Stimulus Onset Times",
            xaxis_title="EEG time (ms)",
            yaxis_title="ET time (ms)"
        )
        fig_sync.show()
    
    return (aligned_eeg_event_dict, aligned_eeg_events) '''