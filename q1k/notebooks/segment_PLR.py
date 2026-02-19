import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium")


@app.cell
def parameters():
    # __Q1K_PARAMETERS__
    project_path = ""
    task_id = "PLR"
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
    from q1k.segment.tasks import TASK_PARAMS, segment_plr
    return (mne, mne_bids, np, plt, warnings, segment_plr,
            TASK_PARAMS, get_sync_loss_path, get_segment_path)


@app.cell
def header(subject_id, task_id):
    import marimo as mo
    mo.md(f"# Single Subject Segmentation Q1K - PLR\n\n"
          f"**Subject:** {subject_id} | **Task:** {task_id}")
    return (mo,)


@app.cell
def load_data(mne, mne_bids, project_path, subject_id, session_id,
              task_id, derivative_base):
    from pathlib import Path

    pp = Path(project_path)
    if derivative_base == "sync_loss":
        input_root = (pp / "derivatives" / "pylossless"
                      / "derivatives" / "sync_loss")
    else:
        input_root = (pp / "derivatives" / "pylossless"
                      / "derivatives" / derivative_base)

    bids_path = mne_bids.BIDSPath(
        subject=subject_id, session=session_id, task=task_id,
        run="1", datatype="eeg", suffix="eeg", root=str(input_root),
    )
    eeg_raw = mne_bids.read_raw_bids(bids_path=bids_path, verbose=False)
    return eeg_raw, bids_path


@app.cell
def get_events(mne, eeg_raw):
    eeg_events, eeg_event_dict = mne.events_from_annotations(eeg_raw)
    return eeg_events, eeg_event_dict


@app.cell
def create_epochs(segment_plr, eeg_raw, eeg_events, eeg_event_dict):
    epochs, event_id, conditions = segment_plr(
        eeg_raw, eeg_events, eeg_event_dict,
    )
    return epochs, event_id, conditions


@app.cell
def save_epochs(epochs, bids_path, project_path, task_id,
                derivative_base):
    from pathlib import Path

    pp = Path(project_path)
    if derivative_base == "sync_loss":
        seg_path = (pp / "derivatives" / "pylossless"
                    / "derivatives" / "sync_loss"
                    / "derivatives" / "segment")
    else:
        seg_path = (pp / "derivatives" / "pylossless"
                    / "derivatives" / derivative_base)

    out_dir = seg_path / "epoch_fif_files" / task_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{bids_path.basename}_epo.fif"
    epochs.save(str(out_file), overwrite=True)
    return (out_file,)


@app.cell
def plot_erp_joint(epochs, conditions):
    figs = []
    for cond in conditions:
        evoked = epochs[cond].average()
        fig = evoked.plot_joint(title=f"ERP: {cond}")
        figs.append(fig)
    return (figs,)


@app.cell
def plot_erp_overlay(epochs, conditions, mne):
    evokeds = {cond: epochs[cond].average() for cond in conditions}
    fig = mne.viz.plot_compare_evokeds(
        evokeds, picks=["E70"],
        title="PLR ERP overlay (E70)",
    )
    fig
    return (fig,)


@app.cell
def plot_pupil_left_overlay(epochs, conditions, mne):
    evokeds = {cond: epochs[cond].average() for cond in conditions}
    fig = mne.viz.plot_compare_evokeds(
        evokeds, picks=["pupil_left"],
        title="PLR pupil_left overlay",
    )
    fig
    return (fig,)


@app.cell
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

    return (tfr_results,)


if __name__ == "__main__":
    app.run()
