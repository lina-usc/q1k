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
    mo.md(f"# Single Subject PLR Segmentation Q1K - PLR\n\n"
          f"Subject: {subject_id} | **Task:** {task_id}")
    return (mo,)


@app.cell
def load_data(mne, mne_bids, project_path, subject_id, session_id,
              task_id, derivative_base):
    from pathlib import Path
    import re
    pp_ad = Path(project_path)
    match = re.search(r'(\d+)_?([A-Z0-9]+)$', subject_id)
    if match:
        bids_subject = match.group(1) + match.group(2)
    else:
        # Fallback: remove underscores and Q1K prefix
        bids_subject = subject_id.replace('Q1K_', '').replace('_', '')
    print(f"Original subject_id: {subject_id}")
    print(f"BIDS subject_id: {bids_subject}")
    if derivative_base == "sync_loss":
        input_root = (pp_ad / "derivatives" / "sync_loss")
    else:
        input_root = (pp_ad / "derivatives" /  derivative_base)

    bids_path = mne_bids.BIDSPath(
        subject=bids_subject, session=session_id, task=task_id,
        run="1", datatype="eeg", suffix="eeg", root=str(input_root),
    )
    print(f"Loading from: {bids_path.fpath}")
    eeg_raw = mne_bids.read_raw_bids(bids_path=bids_path, verbose=False)
    print(f"✓ {len(eeg_raw.ch_names)} ch, {eeg_raw.n_times} samples")
    return eeg_raw, bids_path, pp_ad


@app.cell
def get_events(mne, eeg_raw):
    eeg_events, eeg_event_dict = mne.events_from_annotations(eeg_raw)
    return eeg_events, eeg_event_dict


@app.cell
def create_epochs(segment_plr, eeg_raw, eeg_events, eeg_event_dict):
    epochs, event_id, conditions = segment_plr(
        eeg_raw, eeg_events, eeg_event_dict,
    )
    print(f"Created {len(epochs.events)} epochs | conditions: {conditions}")
    return epochs, event_id, conditions


@app.cell
def save_epochs(epochs, bids_path, pp_ad, task_id,
                derivative_base):
    clean_epochs = epochs.copy().drop_bad()
    if derivative_base == "segment":
        seg_path = (pp_ad / "derivatives" / "segment")
    else:
        seg_path = (pp_ad / "derivatives" / "segment")

    out_dir = seg_path / "epoch_fif_files" / task_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{bids_path.basename}_epo.fif"
    clean_epochs.save(str(out_file), overwrite=True)
    print(f"Saved: {out_file}")
    return (out_file,)


@app.cell
def plot_erp_joint(epochs, conditions, pp_ad, bids_path, plt):
    figs = []
    fig_dir_joint = (pp_ad / "derivatives" / "segment"
                / "figures" / "PLR" / bids_path.basename)
    fig_dir_joint.mkdir(parents=True, exist_ok=True)
    for cond_joint in conditions:
        try:
            evoked = epochs[cond_joint].average()
            fig_joint = evoked.plot_joint(title=f"ERP: {cond_joint}")
            plott = fig_dir_joint / f"erp_joint_{cond_joint}.png"
            fig_joint.savefig(str(plott), dpi=150, bbox_inches="tight")
            plt.close(fig_joint)
            figs.append(fig_joint)
        except Exception as e:
            print(f" Error : {e}")
    return (figs,)


@app.cell
def plot_erp_overlay(epochs, conditions, mne, pp_ad, bids_path, plt):
    '''evokeds = {cond: epochs[cond].average() for cond in conditions}
    fig = mne.viz.plot_compare_evokeds(
        evokeds, picks=["E70"],
        title="PLR ERP overlay (E70)",
    )
    fig
    return (fig,)'''
    _fig_dirover = (pp_ad / "derivatives" / "segment"
                / "figures" / "PLR" / bids_path.basename)
    _fig_dirover.mkdir(parents=True, exist_ok=True)
    _pick = (["E70"] if "E70" in epochs.ch_names
             else [[ch for ch in epochs.ch_names
                    if ch.startswith(("E", "eeg"))][0]]
             if any(ch.startswith(("E", "eeg")) for ch in epochs.ch_names)
             else "eeg")
    _lbl = _pick[0] if isinstance(_pick, list) else "EEG"
    _fig_path = None
    try:
        _evokeds2 = {c: epochs[c].average() for c in conditions}
        _fig2 = mne.viz.plot_compare_evokeds(
            _evokeds2, picks=_pick, title=f"PLR ERP ({_lbl})", show=False
        )
        _fig_path = _fig_dirover / f"erp_overlay_{_lbl}.png"
        (_fig2[0] if isinstance(_fig2, tuple) else _fig2).savefig(
            str(_fig_path), dpi=150, bbox_inches="tight"
        )
        plt.close((_fig2[0] if isinstance(_fig2, tuple) else _fig2))
        print(f"   {_fig_path.name}")
    except Exception as _e:
        print(f"   overlay: {_e}")
    return (_fig_path,)


@app.cell
def plot_pupil_left_overlay(epochs, conditions, mne, pp_ad, bids_path, plt):
    '''evokeds = {cond: epochs[cond].average() for cond in conditions}
    fig = mne.viz.plot_compare_evokeds(
        evokeds, picks=["pupil_left"],
        title="PLR pupil_left overlay",
    )
    fig'''
    _fig_dir = (pp_ad / "derivatives" / "segment"
                / "figures" / "PLR" / bids_path.basename)
    _fig_dir.mkdir(parents=True, exist_ok=True)
    _fps = []
    for _ch in [ch for ch in epochs.ch_names if "pupil" in ch.lower()]:
        try:
            _evokeds3 = {c: epochs[c].average() for c in conditions}
            _fig3 = mne.viz.plot_compare_evokeds(
                _evokeds3, picks=[_ch],
                title=f"PLR {_ch}", show=False
            )
            _fp2 = _fig_dir / f"pupil_{_ch}_overlay.png"
            (_fig3[0] if isinstance(_fig3, tuple) else _fig3).savefig(
                str(_fp2), dpi=150, bbox_inches="tight"
            )
            plt.close((_fig3[0] if isinstance(_fig3, tuple) else _fig3))
            _fps.append(_fp2)
        except Exception as _e:
            print(f"  ✗ pupil {_ch}: {_e}")
    return (_fps,)
    #return (fig,)


@app.cell
def plot_tfr(epochs, conditions, mne, np, pp_ad, bids_path, plt):
    fig_dir_tfr = (pp_ad / "derivatives" / "segment"
                / "figures" / "PLR" / bids_path.basename)
    fig_dir_tfr.mkdir(parents=True, exist_ok=True)

    freqs = np.arange(2, 51, 2)
    n_cycles =freqs / 2.0

    tfr_results = {}
    for condtfr in conditions:
        try:
            power, itc = mne.time_frequency.tfr_morlet(
               epochs[condtfr], freqs=freqs, n_cycles=n_cycles,
               return_itc=True,picks="eeg", verbose=False, n_jobs=1
            )
            tfr_results[condtfr] = (power, itc)

            #for condtfr, (power, itc) in tfr_results.items():
                #power.plot(title=f"TFR Power: {condtfr}", picks="eeg")
                #itc.plot(title=f"ITC: {condtfr}", picks="eeg")
            fig_power = power.plot(title=f"TFR Power: {condtfr}", picks="eeg", show=False)
            fig_path_power = fig_dir_tfr / f"tfr_power_{condtfr}.png"
            (fig_power[0] if isinstance(fig_power, list) else fig_power).savefig(
                str(fig_path_power), dpi=150, bbox_inches="tight"
            )
            plt.close((fig_power[0] if isinstance(fig_power, list) else fig_power))
            del power
            # Plot ITC
            fig_itc = itc.plot(title=f"ITC: {condtfr}", picks="eeg", show=False)
            fig_path_itc = fig_dir_tfr / f"tfr_itc_{condtfr}.png"
            (fig_itc[0] if isinstance(fig_itc, list) else fig_itc).savefig(
                str(fig_path_itc), dpi=150, bbox_inches="tight"
            )
            plt.close((fig_itc[0] if isinstance(fig_itc, list) else fig_itc))
            del itc
            print(f"  ✓ Saved TFR plots for {condtfr}")
        except Exception as _e:
            print(f"  ✗ TFR {condtfr}: {_e}")
    return (tfr_results,)


if __name__ == "__main__":
    app.run()
