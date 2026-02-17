import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium")


@app.cell
def parameters():
    # __Q1K_PARAMETERS__
    project_path = ""
    task_id = "RS"
    subject_id = ""
    session_id = "01"
    run_id = "1"
    derivative_base = "sync_loss"
    return (project_path, task_id, subject_id, session_id, run_id,
            derivative_base)


@app.cell
def imports():
    import mne
    import mne_bids
    import numpy as np
    import matplotlib.pyplot as plt
    import warnings
    warnings.filterwarnings("ignore")

    from q1k.segment.tasks import (
        segment_resting_state, segment_rsrio,
        _detect_rsrio, TASK_PARAMS,
    )
    from q1k.config import FREQ_BANDS, FRONTAL_ROI
    from q1k.io import get_sync_loss_path, get_segment_path
    return (mne, mne_bids, np, plt, warnings,
            segment_resting_state, segment_rsrio,
            _detect_rsrio, TASK_PARAMS, FREQ_BANDS, FRONTAL_ROI,
            get_sync_loss_path, get_segment_path)


@app.cell
def header(subject_id, task_id):
    import marimo as mo
    mo.md(f"# Single Subject Segmentation Q1K - Resting State\n\n"
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
def detect_task(eeg_event_dict, _detect_rsrio, task_id):
    """Auto-detect whether this recording is RS or RSRio."""
    if _detect_rsrio(eeg_event_dict):
        actual_task = "RSRio"
        print("Detected: Resting-state Rio (eyeo/comm markers)")
    else:
        actual_task = task_id
        print(f"Detected: standard resting state ({actual_task})")
    return (actual_task,)


@app.cell
def plot_events(mne, eeg_events, eeg_event_dict, eeg_raw):
    fig = mne.viz.plot_events(
        eeg_events, sfreq=eeg_raw.info["sfreq"],
        first_samp=eeg_raw.first_samp, event_id=eeg_event_dict,
    )
    fig
    return (fig,)


@app.cell
def create_epochs(segment_resting_state, segment_rsrio,
                  eeg_raw, eeg_events, eeg_event_dict,
                  actual_task):
    if actual_task == "RSRio":
        epochs, event_id, conditions = segment_rsrio(
            eeg_raw, eeg_events, eeg_event_dict,
        )
    else:
        epochs, event_id, conditions = segment_resting_state(
            eeg_raw, eeg_events, eeg_event_dict,
        )
    return epochs, event_id, conditions


@app.cell
def save_epochs(epochs, bids_path, project_path, derivative_base,
                actual_task):
    from pathlib import Path

    pp = Path(project_path)
    if derivative_base == "sync_loss":
        seg_path = (pp / "derivatives" / "pylossless"
                    / "derivatives" / "sync_loss"
                    / "derivatives" / "segment")
    else:
        seg_path = (pp / "derivatives" / "pylossless"
                    / "derivatives" / derivative_base)

    out_dir = seg_path / "epoch_fif_files" / actual_task
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{bids_path.basename}_epo.fif"
    epochs.save(str(out_file), overwrite=True)
    print(f"Saved epochs to {out_file}")
    return (out_file,)


@app.cell
def plot_psd(epochs, np, FRONTAL_ROI):
    epo_power = epochs.compute_psd(fmin=0, fmax=50)
    fig1 = epo_power.plot(picks="eeg", exclude="bads", amplitude=False)
    fig2 = epo_power.plot(picks=FRONTAL_ROI, exclude="bads",
                          amplitude=False)
    fig3 = epo_power.plot_topomap(ch_type="eeg", agg_fun=np.median)
    return fig1, fig2, fig3


if __name__ == "__main__":
    app.run()
