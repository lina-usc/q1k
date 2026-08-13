import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium")


@app.cell
def parameters():
    # __Q1K_PARAMETERS__
    project_path = ""
    task_id = "VEP"
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
    from q1k.segment.tasks import TASK_PARAMS, segment_vep
    return (mne, mne_bids, np, plt, warnings, segment_vep,
            TASK_PARAMS, get_sync_loss_path, get_segment_path)


@app.cell
def header(subject_id, task_id):
    import marimo as mo
    mo.md(f"# Single Subject Segmentation Q1K - VEP\n\n"
          f"**Subject:** {subject_id} | **Task:** {task_id}")
    return (mo,)


@app.cell
def load_data(mne_bids, project_path, subject_id, session_id,
              task_id, derivative_base, run_id):
    from pathlib import Path as _Path

    _pp = _Path(project_path)
    if derivative_base == "sync_loss":
        input_root = _pp / "derivatives" / "sync_loss" / task_id
    else:
        input_root = _pp / "derivatives" / derivative_base / task_id

    bids_path = mne_bids.BIDSPath(
        subject=subject_id, session=session_id, task=task_id,
        run=run_id, datatype="eeg", suffix="eeg", root=str(input_root),
    )

    print(f"Loading data from: {bids_path.fpath}")
    eeg_raw = mne_bids.read_raw_bids(bids_path=bids_path, verbose=False)
    return eeg_raw, bids_path


@app.cell
def get_events(mne, eeg_raw):
    eeg_events, eeg_event_dict = mne.events_from_annotations(eeg_raw)
    return eeg_events, eeg_event_dict


@app.cell
def create_epochs(segment_vep, eeg_raw, eeg_events, eeg_event_dict):
    epochs, event_id, conditions = segment_vep(
        eeg_raw, eeg_events, eeg_event_dict,
    )
    return epochs, event_id, conditions


@app.cell
def save_epochs(epochs, bids_path, project_path, task_id,
                derivative_base):
    from pathlib import Path as _Path

    _pp = _Path(project_path)
    if derivative_base == "sync_loss":
        seg_path = (_pp / "derivatives" / "segment")
    else:
        seg_path = (_pp / "derivatives" / derivative_base)

    out_dir = seg_path / "epoch_fif_files" / task_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{bids_path.basename}_epo.fif"
    epochs.save(str(out_file), overwrite=True)
    return (out_file,)


@app.cell
def plot_erp_joint(epochs, conditions):
    figs = []
    for _vep_cond in conditions:
        _vep_evoked = epochs[_vep_cond].average()
        _vep_fig = _vep_evoked.plot_joint(title=f"ERP: {_vep_cond}")
        figs.append(_vep_fig)
    return (figs,)


@app.cell
def plot_erp_overlay(epochs, conditions, mne):
    evokeds = {
        _vep_overlay_cond: epochs[_vep_overlay_cond].average()
        for _vep_overlay_cond in conditions
    }
    fig_overlay = mne.viz.plot_compare_evokeds(
        evokeds,
        picks=["E70"],
        title="VEP ERP overlay (E70)",
    )
    return (fig_overlay,)

#memory is crashing again 
'''@app.cell
def plot_tfr(epochs, conditions, mne, np):
    freqs = np.arange(2, 51, 1)
    n_cycles = freqs / 2.0

    tfr_results = {}
    for cond in conditions:
        power, itc = mne.time_frequency.tfr_morlet(
            epochs[cond], freqs=freqs, n_cycles=n_cycles,
            return_itc=True,
        )
        tfr_results[cond] = (power, itc)

    for cond, (power, itc) in tfr_results.items():
        power.plot(title=f"TFR Power: {cond}", picks="eeg")
        itc.plot(title=f"ITC: {cond}", picks="eeg")

    return (tfr_results,)'''


if __name__ == "__main__":
    app.run()