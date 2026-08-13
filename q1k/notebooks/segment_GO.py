import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium")


@app.cell
def parameters():
    # __Q1K_PARAMETERS__
    project_path = ""
    task_id = "GO"
    subject_id = ""
    session_id = "01"
    run_id = "1"
    derivative_base = "sync_loss"
    return (project_path, task_id, subject_id, session_id, run_id,
            derivative_base)


@app.cell
def imports():
    import warnings

    import matplotlib.pyplot as plt
    import mne
    import mne_bids
    import numpy as np
    warnings.filterwarnings("ignore")

    from q1k.io import get_segment_path, get_sync_loss_path
    from q1k.segment.tasks import TASK_PARAMS, segment_go
    return (mne, mne_bids, np, plt, warnings, segment_go,
            TASK_PARAMS, get_sync_loss_path, get_segment_path)


@app.cell
def header(subject_id, task_id):
    import marimo as mo
    mo.md(f"# Single Subject Segmentation Q1K - GO\n\n"
          f"**Subject:** {subject_id} | **Task:** {task_id}")
    return (mo,)


@app.cell
def load_data(mne, mne_bids, project_path, subject_id, session_id,
              task_id, derivative_base):
    from pathlib import Path as Path1

    pp1 = Path1(project_path)
    if derivative_base == "sync_loss":
        input_root = (pp1 / "derivatives" / "sync_loss"/task_id)
    else:
        input_root = (pp1 / "derivatives" / derivative_base)

    bids_path = mne_bids.BIDSPath(
        subject=subject_id, session=session_id, task=task_id,
        run="1", datatype="eeg", suffix="eeg", root=str(input_root),
    )
    print(f"Loading data from: {bids_path}")

    print(f"Root path: {input_root}")
    eeg_raw = mne_bids.read_raw_bids(bids_path=bids_path, verbose=False)
    return eeg_raw, bids_path


@app.cell
def get_events(mne_bids, mne,np, bids_path):
    import pandas as pd
    events_fname = bids_path.copy().update(suffix='events', extension='.tsv').fpath
    # Strip BOM then let mne.read_events do its normal job
    df = pd.read_csv(events_fname, sep='\t')
    # Building MNE events array [sample, 0, event_id] — same format as mne.read_events
    raw = mne.io.read_raw_edf(str(bids_path.fpath), preload=False, verbose=False)
    sfreq = raw.info['sfreq']
    #sfreq = mne.read_raw(str(bids_path.fpath), preload=False).info['sfreq']
    
    samples = (df['onset'].values * sfreq).astype(int)
    durations = np.zeros(len(samples), dtype=int)
    unique_types = sorted(df['trial_type'].unique())
    type_to_id = {t: i+1 for i, t in enumerate(unique_types)}
    event_ids = np.array([type_to_id[t] for t in df['trial_type']], dtype=int)
    eeg_events = np.column_stack([samples, durations, event_ids])
    unique_ids = np.unique(eeg_events[:, 2])
    
    #eeg_event_dict = {f"event_{int(i)}": int(i) for i in unique_ids}
    eeg_event_dict = {t: type_to_id[t] for t in unique_types}
    print(f"Found {len(df)} events in events.tsv")
    print(f"Unique trial types: {df['trial_type'].unique()}")
    return eeg_events, eeg_event_dict


@app.cell
def create_epochs(segment_go, eeg_raw, eeg_events, eeg_event_dict):

    epochs, event_id, conditions = segment_go(
        eeg_raw, eeg_events, eeg_event_dict,
    )
    return epochs, event_id, conditions


@app.cell
def save_epochs(epochs, bids_path, project_path, task_id,
                derivative_base):
    from pathlib import Path as Path2
    out_file = None
    if epochs is None:
        print("No epochs to save - skipping")
    else:
        epochs.drop_bad()
        pp = Path2(project_path)
        if derivative_base == "sync_loss":
            seg_path = (pp / "derivatives" / "segment")
        else:
            seg_path = (pp / "derivatives" / derivative_base)
    
        out_dir = seg_path / "epoch_fif_files" / task_id
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"{bids_path.basename}_epo.fif"
        epochs.save(str(out_file), overwrite=True)
    return (out_file,)


@app.cell
def plot_erp_joint(epochs, conditions):
    figs = []
    for cond1 in conditions:
        evoked = epochs[cond1].average()
        fig1 = evoked.plot_joint(title=f"ERP: {cond1}")
        figs.append(fig1)
    return (figs,)


@app.cell
def plot_erp_overlay(epochs, conditions, mne):
    fig2 = None
    if not conditions:
        print("No conditions found - skipping ERP overlay plot")
    else:
        evokeds1 = {cond2: epochs[cond2].average() for cond2 in conditions}
        fig2 = mne.viz.plot_compare_evokeds(
            evokeds1, picks=["E6"],
            title="GO ERP overlay (E6)",
        )
        fig2
    return (fig2,)


@app.cell
def plot_pupil_left_overlay(epochs, conditions, mne):
    fig = None
    if not conditions:
        print("No conditions found - skipping pupil overlay plot")
    else:
        _pupil_name = next(
            (ch for ch in ("pupil_left", "pupil_right") if ch in epochs.ch_names),
            None,
        )

        if _pupil_name is None:
            print("No pupil_left or pupil_right found - skipping pupil overlay plot")
        else:
            _pupil_idx = epochs.ch_names.index(_pupil_name)

            _pupil_evokeds = {}
            for _pupil_cond in conditions:
                _ep = epochs[_pupil_cond].copy().load_data()
                _ep = _ep.pick([_pupil_idx])
                _pupil_evokeds[_pupil_cond] = _ep.average(picks=[0])

            fig = mne.viz.plot_compare_evokeds(
                _pupil_evokeds,
                picks=[0],
                title=f"GO {_pupil_name} overlay",
            )

    return (fig,)

'''
@app.cell
def plot_tfr(epochs, conditions, mne, np):
    tfr_results = None
    if not conditions:
        print("No conditions found - skipping TFR plot")
    else:
        #memory issue : The Python kernel for file /lustre07/scratch/rsweety/white_paper/wd/derivatives/reports/segment/GO/HSJ0104P_GO_segment.py died unexpectedly.
        _freqs = np.arange(2, 51, 1)
        _n_cycles = _freqs / 2.0

        tfr_results = {}
        for _tfr_cond in conditions:
            _power, _itc = mne.time_frequency.tfr_morlet(
                epochs[_tfr_cond], freqs=_freqs, n_cycles=_n_cycles,
                return_itc=True,
            )
            tfr_results[_tfr_cond] = (_power, _itc)

        for _tfr_label, (_power, _itc) in tfr_results.items():
            _power.plot(title=f"TFR Power: {_tfr_label}", picks="eeg")
            _itc.plot(title=f"ITC: {_tfr_label}", picks="eeg")

    return (tfr_results,)
'''

if __name__ == "__main__":
    app.run()