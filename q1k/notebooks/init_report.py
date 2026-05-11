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
def convert_edf_to_asc(pp, site_code, subject_id, session_id,
                       task_id_in):
    import eyelinkio
    asc_out = None
    # Tasks that require eye-tracking data
    ET_TASKS_INIT = {"VEP", "GO", "PLR", "VS", "NSP"}

    # Skip conversion entirely for non-ET tasks
    if task_id_in.upper() not in ET_TASKS_INIT:
        print(f"Task {task_id_in} does not use eye-tracking — skipping EDF conversion.")
        asc_out = None

    # Extract the short ET subject prefix from Q1K_<SITE>_<NUM>_<SUFFIX> format
    parts = subject_id.split("_")
    if len(parts) >= 4:
        _et_id = parts[2] + parts[3]
    elif len(parts) >= 3:
        _et_id = parts[2]
    else:
        _et_id = subject_id
    if "-" in _et_id:
        _et_id = _et_id.split("-")[1]
    et_prefix = _et_id.replace("_", "")

    # including the task name in the glob pattern so we
    # never accidentally pick up an EDF for a different task (e.g. GO.edf
    # when processing RS).
    search_dirs = [
        pp / "sourcedata" / site_code / "et" / f"{task_id_in}_raw",
        pp / "sourcedata" / site_code / "et",
    ]
    edf_file = None
    for d in search_dirs:
        if d.exists():
            # Require task name in filename to avoid cross-task matches
            edfs = list(d.rglob(f"*{et_prefix}*{task_id_in}*.edf"))
            if edfs:
                edf_file = edfs[0]
                print(f"Found EDF: {edf_file}")
                break

    if edf_file:
        out_dir = (pp / "derivatives" / "init"
                   / f"sub-{subject_id}" / f"ses-{session_id}" / "et")
        out_dir.mkdir(parents=True, exist_ok=True)
        asc_out = out_dir / f"{edf_file.stem}.asc"
        if not asc_out.exists():
            print("Converting to ASC…")
            edf_data = eyelinkio.read_edf(str(edf_file))
            edf_data.to_asc(str(asc_out))
            print(f"Saved: {asc_out}")
        else:
            print(f"ASC already exists: {asc_out}")
    else:
        print(f"No EDF found for {subject_id} / {task_id_in} — ET data unavailable.")
    return (asc_out,)


@app.cell
def setup_paths(project_path, subject_id, session_id, task_id_in,
                task_id_out, Path, site_code, asc_out):
    import marimo as mo

    # Map every known task to its event-dict offset and DIN trigger strings
    task_config = {
        "AEP":   {"offset": 1, "din_str": ("DIN4", "DIN4")},
        "ap":    {"offset": 1, "din_str": ("DIN4", "DIN4")},
        "as":    {"offset": 1, "din_str": ("DIN2", "DIN2")},
        "TO":    {"offset": 1, "din_str": ("DIN2", "DIN2")},
        "go":    {"offset": 1, "din_str": ("DIN4", "DIN5")},
        "GO":    {"offset": 1, "din_str": ("DIN4", "DIN5")},
        "mn":    {"offset": 1, "din_str": ("DIN4", "DIN4")},
        "vp":    {"offset": 0, "din_str": ("DIN2", "DIN3")},
        "VEP":   {"offset": 0, "din_str": ("DIN2", "DIN3")},
        "RSRio": {"offset": 1, "din_str": ("DIN2", "DIN2")},
        "rest":  {"offset": 1, "din_str": ("DIN2", "DIN2")},
        "RS":    {"offset": 1, "din_str": ("DIN2", "DIN2")},
    }

    cfg = task_config.get(task_id_out, {"offset": 1, "din_str": ("DIN2", "DIN2")})
    event_dict_offset = cfg["offset"]
    din_str = cfg["din_str"]

    pp = Path(project_path)

    # ── Locate EEG source directory ──────────────────────────────────────
    # Trying both the raw subject ID and a version without underscores
    subject_versions = [subject_id, subject_id.replace("_", "")]
    session_path_eeg = None
    for subj_version in subject_versions:
        test_path = pp / "sourcedata" / site_code / "eeg" / subj_version
        if test_path.exists():
            session_path_eeg = test_path
            break
    if session_path_eeg is None:
        raise FileNotFoundError(
            f"Could not find EEG sourcedata for subject '{subject_id}' "
            f"under {pp / 'sourcedata' / site_code / 'eeg'}"
        )

    # Collecting .mff folders that match the task
    session_file_name_eeg = [
        d for d in session_path_eeg.iterdir()
        if d.is_dir() and d.name.endswith(".mff") and task_id_in in d.name
    ]
    if not session_file_name_eeg:
        print(f"WARNING: No .mff files found in {session_path_eeg} "
              f"matching task '{task_id_in}'")

    # ── Locating ET source file (.asc) ─────────────────────────────────────
    # ET is only relevant for tasks in ET_TASKS; for all others we skip
    # the search entirely to avoid picking up ET files from other tasks.
    ET_TASKS_SETUP = {"VEP", "GO", "PLR", "VS", "NSP"}
    session_file_name_et = []

    if task_id_out in ET_TASKS_SETUP:
        # 1. Check derivatives/init for a previously converted ASC
        init_et_dir = (pp / "derivatives" / "init"
                       / f"sub-{subject_id}" / f"ses-{session_id}" / "et")
        if init_et_dir.exists():
            subject_parts = subject_id.split("_")
            if len(subject_parts) >= 3:
                et_id_raw = "_".join(subject_parts[2:])
                et_file_prefix = et_id_raw.replace("_", "").replace("-", "")
            else:
                et_file_prefix = subject_id.replace("_", "")
            # Include task name in glob to avoid cross-task matches
            pattern = f"{et_file_prefix}*{task_id_in}*.asc"
            session_file_name_et = list(init_et_dir.rglob(pattern))
            if session_file_name_et:
                print(f"✓ ET from derivatives/init: {session_file_name_et[0]}")

        # 2. Fallback: search sourcedata/et
        if not session_file_name_et:
            et_base = pp / "sourcedata" / site_code / "et"
            subject_parts = subject_id.split("_")

            # FIX (BUG-4): et_id was used before assignment when the subject
            # ID had no '-' separator.  Now we always set et_id regardless of
            # whether a '-' is present.
            if len(subject_parts) >= 3:
                et_id_raw = "_".join(subject_parts[2:])
                if "-" in et_id_raw:
                    et_id = et_id_raw.split("-")[1]
                else:
                    et_id = et_id_raw          # ← was missing; caused NameError
            else:
                et_id = subject_id

            et_file_prefix = et_id.replace("_", "")
            if et_base.exists():
                # Include task name in glob pattern
                session_file_name_et = list(
                    et_base.rglob(f"{et_file_prefix}*{task_id_in}*.asc")
                )
                if session_file_name_et:
                    print(f"✓ ET from sourcedata: {session_file_name_et[0]}")
    else:
        print(f"Task {task_id_out} does not use eye-tracking — skipping ET file search.")

    print(f"EEG files found: {len(session_file_name_eeg)}")
    print(f"ET  files found: {len(session_file_name_et)}")
    if session_file_name_eeg:
        print(f"  EEG: {session_file_name_eeg[0].name}")
    if session_file_name_et:
        print(f"  ET:  {session_file_name_et[0].name}")

    mo.md(f"Q1K Init Report: **{subject_id}** — {task_id_out}")
    return (event_dict_offset, din_str, session_file_name_eeg,
            session_file_name_et, pp, mo, asc_out)


# When no .mff file is found (e.g. subject processed under wrong task code),
# the old code crashed with an opaque IndexError.
@app.cell
def read_eeg(mne, session_file_name_eeg, event_dict_offset, get_event_dict):
    if not session_file_name_eeg:
        raise FileNotFoundError(
            "No EEG .mff files found for this subject/task combination. "
            "Check that the .mff folder exists in sourcedata and that "
            "task_id_in matches the filename."
        )

    raw = mne.io.read_raw_egi(session_file_name_eeg[0])
    eeg_events = mne.find_events(raw, shortest_event=1)
    eeg_event_dict = get_event_dict(raw, eeg_events, event_dict_offset)

    # Ensure every numeric event ID has at least a generic label so
    # downstream code never hits a missing-key error
    unique_events = set(eeg_events[:, 2])
    for event_id in unique_events:
        if event_id not in eeg_event_dict.values():
            eeg_event_dict[f"event_{int(event_id)}"] = int(event_id)

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
    import numpy as _np

    if task_id_out in NO_DIN_OFFSET_TASKS:
        # Tasks like RS and RSRio do not use stimulus DIN offsets
        print(f"{task_id_out}: skipping stimulus DIN event offset")
        eeg_events_processed = eeg_events
        eeg_stims = _np.empty((0, 3), dtype=int)
        eeg_iti = _np.array([])
        eeg_din_offset = []
        eeg_event_dict_updated = eeg_event_dict
        new_events = _np.empty((0, 3), dtype=int)
    else:
        # wraping eeg_event_test in a try/except so that if the
        # expected DIN channel is absent in a recording we degrade gracefully
        # instead of propagating a bare KeyError through every downstream cell.
        try:
            (eeg_events_processed, eeg_stims, eeg_iti,
             eeg_din_offset, eeg_event_dict_updated,
             new_events) = eeg_event_test(
                eeg_events, eeg_event_dict, din_str,
                task_name=task_id_out,
            )
        except KeyError as _ke:
            print(
                f"WARNING: Expected DIN channel {_ke} not found in recording "
                f"for task '{task_id_out}'.  Processing without DIN offset."
            )
            eeg_events_processed = eeg_events
            eeg_stims = _np.empty((0, 3), dtype=int)
            eeg_iti = _np.array([])
            eeg_din_offset = []
            eeg_event_dict_updated = eeg_event_dict
            new_events = _np.empty((0, 3), dtype=int)

    return (eeg_events_processed, eeg_stims, eeg_iti,
            eeg_din_offset, eeg_event_dict_updated, new_events)


# guarding against empty or single-row eeg_stims before plotting.
# px.scatter raised "Cannot accept list of column references for both x and y"
# when the arrays had mismatched lengths (0 stims → 0-length ITI array but
# eeg_stims[1:,0] was also empty causing a shape mismatch in plotly).
@app.cell
def plot_stim_iti(px, eeg_stims, eeg_iti, task_id_out, NO_DIN_OFFSET_TASKS):
    if task_id_out in NO_DIN_OFFSET_TASKS:
        print(f"{task_id_out}: skipping stimulus DIN ITI display")
        fig_stim = None
    elif eeg_stims is None or len(eeg_stims) < 2 or len(eeg_iti) == 0:
        # Not enough stim events to compute ITI — show a warning instead of crashing
        print(
            f"WARNING: Insufficient DIN stimulus events to plot ITI "
            f"(found {len(eeg_stims)} stim(s), need ≥ 2)."
        )
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

    # RSRio retains stim channels; all other tasks drop them
    if task_id_out in NO_DIN_OFFSET_TASKS:
        print("Retaining stim channels for RSRio task…")
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

    # Build a clean numeric BIDS subject ID (e.g. "1248P") from the full
    # Q1K ID (e.g. "Q1K_HSJ_1248_P" or "Q1K_MHC_1525-1248_P")
    match = re.search(r"(\d+)_?([A-Z0-9]+)$", subject_id)
    if match:
        bids_subject = match.group(1) + match.group(2)
    else:
        bids_subject = subject_id  # fallback — should not happen in practice
    print(f"Using BIDS subject ID: {bids_subject}")
    print(f"Session ID:            {session_id}")
    print(f"Task:                  {task_id_out}")
    print(f"Root path:             {pp}")
    events_count = (len(eeg_events_processed)
                    if eeg_events_processed is not None else "None")
    print(f"Events processed:      {events_count}")

    bids_path = mne_bids.BIDSPath(
        subject=bids_subject, session=session_id,
        task=task_id_out,
        run="1", datatype="eeg", root=str(pp / "derivatives" / "init"),
    )

    mne_bids.write_raw_bids(
        raw=raw, bids_path=bids_path,
        events=eeg_events_processed,
        event_id=eeg_event_dict_updated,
        format="EDF", overwrite=True,
    )

    # Strip BOM from all TSV files written by mne-bids (known upstream bug)
    import pandas as pd
    bids_dir = (pp / "derivatives" / "init"
                / f"sub-{bids_subject}" / f"ses-{session_id}" / "eeg")
    for tsv_file in bids_dir.glob("*.tsv"):
        df_tsv = pd.read_csv(tsv_file, sep="\t", encoding="utf-8-sig")
        df_tsv.to_csv(tsv_file, sep="\t", index=False, encoding="utf-8")

    return (bids_path,)


@app.cell
def read_et(mne, session_file_name_et, task_id_out, asc_out):
    # Only tasks in ET_TASKS require eye-tracking synchronisation
    ET_TASKS_READ = {"VEP", "GO", "PLR", "VS", "NSP"}
    et_sync = task_id_out in ET_TASKS_READ

    if et_sync and (asc_out or session_file_name_et):
        from q1k.init.tools import et_clean_events, et_read
        et_file = asc_out if asc_out else session_file_name_et[0]
        et_raw, et_raw_df, et_annot_events, et_annot_event_dict = et_read(
            str(et_file), blink_interp=False, fill_nans=False, resamp=False,
        )
        et_annot_event_dict, et_annot_events = et_clean_events(
            et_annot_event_dict, et_annot_events
        )
        print(f"ET loaded: {et_file}")
        print(f"ET annotations: {list(et_annot_event_dict.keys())}")
    else:
        et_raw = et_raw_df = et_annot_events = et_annot_event_dict = None
        print(f"ET sync disabled for task: {task_id_out}")

    return et_sync, et_raw, et_raw_df, et_annot_events, et_annot_event_dict


@app.cell
def process_et_events(et_sync, et_raw, et_raw_df, et_annot_events,
                      et_annot_event_dict, task_id_out):
    if et_sync and et_raw is not None:
        task_key = "vp" if task_id_out == "VEP" else task_id_out.lower()
        _, et_events_out, et_stims_out, et_iti_out = et_event_test(
            et_raw_df, task_name=task_key
        )
        print(f"ET stimulus events: {len(et_stims_out)}")
    else:
        et_events_out = et_stims_out = et_iti_out = None

    return et_events_out, et_stims_out, et_iti_out


# stim_keys lookup used exact membership test against a tuple.
# Some recordings store events as "dtoc" (no "_d" suffix) so the tuple match
# missed them.  Replaced with startswith() to catch both "dtoc" and "dtoc_d".
@app.cell
def align_and_save_et(et_sync, et_raw, et_annot_events, et_annot_event_dict,
                      et_stims_out, eeg_events_processed, eeg_event_dict_updated,
                      raw, task_id_out, bids_path):
    import mne as _mne
    import numpy as _np

    if (et_sync and et_raw is not None
            and et_stims_out is not None and len(et_stims_out) > 0):
        from q1k.init.tools import eeg_et_align, et_events_to_annot, write_et

        # Rebuild event array from annotations (stim channels were dropped
        # during BIDS write, so we cannot use the raw stim channels here)
        eeg_events_ann, eeg_event_dict_ann = _mne.events_from_annotations(raw)

        # FIX (BUG-6): use startswith() so we match both "dtoc" and "dtoc_d"
        stim_keys = [
            k for k in eeg_event_dict_ann
            if k.startswith(("dtoc", "dtbc", "dtgc"))
        ]
        if not stim_keys:
            # Fallback: use all processed events when task-specific keys absent
            _eeg_stims_align = eeg_events_processed
            eeg_event_dict_use = eeg_event_dict_updated
            eeg_events_use = eeg_events_processed
        else:
            stim_ids = [eeg_event_dict_ann[k] for k in stim_keys]
            _eeg_stims_align = eeg_events_ann[
                _np.isin(eeg_events_ann[:, 2], stim_ids)
            ]
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
        et_fif_path = write_et(et_raw_annot, bids_path.fpath)
        print(f"ET .fif saved: {et_fif_path}")
    else:
        print("Skipping ET .fif save.")


if __name__ == "__main__":
    app.run()
