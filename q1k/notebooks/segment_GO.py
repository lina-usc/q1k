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
    import matplotlib
    matplotlib.use('Agg')  # Using non-interactive backend
    import matplotlib.pyplot as plt
    import mne
    import mne_bids
    import numpy as np
    warnings.filterwarnings("ignore")

    from q1k.io import get_segment_path, get_sync_loss_path
    from q1k.segment.tasks import TASK_PARAMS, segment_go
    return (mne, mne_bids, np, plt, warnings, segment_go,
            TASK_PARAMS, get_sync_loss_path, get_segment_path, matplotlib)


@app.cell
def header(subject_id, task_id):
    import marimo as mo
    mo.md(f"# Single Subject Segmentation Q1K - GO\n\n"
          f"**Subject:** {subject_id} | **Task:** {task_id}")
    return (mo,)


@app.cell
def load_data(mne, mne_bids, project_path, subject_id, session_id,
              task_id, derivative_base):
    from pathlib import Path

    _pp = Path(project_path)
    sync_path="derivatives/sync_loss"
    if derivative_base == "sync_loss":
        input_root = (_pp / "derivatives" / "sync_loss")
    else:
        input_root = (_pp / "derivatives" / derivative_base)

    bids_path = mne_bids.BIDSPath(
        subject=subject_id, session=session_id, task=task_id,
        run="1", datatype="eeg", suffix="eeg", root=str(input_root),
    )
    print(f"Loading data from: {bids_path.fpath}")
    eeg_raw = mne_bids.read_raw_bids(bids_path=bids_path, verbose=False)
    print(f"✓ Loaded raw data: {len(eeg_raw.ch_names)} channels, {eeg_raw.n_times} samples")
    return eeg_raw, bids_path


@app.cell
def get_events(np, bids_path, eeg_raw):
    import pandas as pd

    events_fname = bids_path.copy().update(suffix='events', extension='.tsv').fpath
    # Strip BOM then let mne.read_events do its normal job
    df = pd.read_csv(events_fname, sep='\t')
    # Building MNE events array [sample, 0, event_id] — same format as mne.read_events
    #sfreq = mne.read_raw(str(bids_path.fpath), preload=False).info['sfreq']
    sfreq = eeg_raw.info['sfreq']
    samples = (df['onset'].values * sfreq).astype(int)
    durations = np.zeros(len(samples), dtype=int)
    unique_types = sorted(df['trial_type'].unique())
    type_to_id = {t: i+1 for i, t in enumerate(unique_types)}
    event_ids = np.array([type_to_id[t] for t in df['trial_type']], dtype=int)
    eeg_events = np.column_stack([samples, durations, event_ids])
    unique_ids = np.unique(eeg_events[:, 2])
    eeg_event_dict = type_to_id
    print(f"Found {len(unique_types)} unique event types:")
    print(f"Events starting with 'd' or 'g': {[k for k in eeg_event_dict.keys() if k.startswith(('d', 'g'))]}")
    #eeg_event_dict = {f"event_{int(i)}": int(i) for i in unique_ids}
    return eeg_events, eeg_event_dict


@app.cell
def create_epochs(segment_go, eeg_raw, eeg_events, eeg_event_dict):
    epochs, event_id, conditions = segment_go(
        eeg_raw, eeg_events, eeg_event_dict,
    )
    print(f"Created {len(epochs.events)} epochs")
    print(f"Conditions: {conditions}")
    print(f"Channels: {epochs.ch_names[:10]}...")
    return epochs, event_id, conditions


@app.cell
def save_epochs(epochs, bids_path, project_path, task_id,
                derivative_base):
    from pathlib import Path as _Path
    epochs_clean = epochs.copy().drop_bad()
    _pp = _Path(project_path)
    seg_path = _pp / "derivatives" / "segment"
    out_dir = seg_path / "epoch_fif_files" / task_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{bids_path.basename}_epo.fif"
    epochs_clean.save(str(out_file), overwrite=True)
    print(f"Saved epochs to: {out_file}")
    return (out_file,)
'''
@app.cell
def plot_erp_joint(epochs, conditions):
    figs = []
    for cond1 in conditions:
        evoked = epochs[cond1].average()
        fig1 = evoked.plot_joint(title=f"ERP: {cond1}")
        figs.append(fig1)
    return (figs,)
'''

@app.cell
def plot_erp_joint(epochs, conditions, project_path, bids_path, plt):
    """Generate ERP joint plots and save as PNG files"""
    from pathlib import Path as _Path
    # Create figures directory
    _pp = _Path(project_path)
    _fig_dir4 = _pp / "derivatives" / "segment" / "figures" / "GO" / bids_path.basename
    _fig_dir4.mkdir(parents=True, exist_ok=True)
    print(f"\nGenerating ERP joint plots...")
    figs = []
    for _cond1 in conditions:
        try:
            evoked_5 = epochs[_cond1].average()
            # Create figure
            _fig4 = evoked_5.plot_joint(
                title=f"ERP: {_cond1}",
                show=False  # Don't display interactively
            )
            # Save figure
            _fig_path9 = _fig_dir4 / f"erp_joint_{_cond1}.png"
            _fig4.savefig(str(_fig_path9), dpi=150, bbox_inches='tight')
            plt.close(_fig4)
            figs.append(_fig_path9)
            print(f"  ✓ Saved: {fig_path.name}")
        except Exception as e:
            print(f"  ✗ Error plotting {_cond1}: {e}")
    return (figs,)

'''
@app.cell
def plot_erp_overlay(epochs, conditions, mne):
    #evokeds1 = {cond2: epochs[cond2].average() for cond2 in conditions}
    #fig2 = mne.viz.plot_compare_evokeds(evokeds1, picks=["E6"], title="GO ERP overlay (E6)",)
    # Check if E6 exists, otherwise use first EEG channel
    if "E6" in epochs.ch_names:
        pick_ch = ["E6"]
    else:
        eeg_channels = [ch for ch in epochs.ch_names if ch.startswith('E')or ch.startswith('eeg')]
        if eeg_channels:
            if len(eeg_channels)>5:
                pick_ch = [eeg_channels[5]]
            else:
                pick_ch =[eeg_channels[0]]
        else:
            pick_ch = "eeg"
    evokeds1 = {cond2: epochs[cond2].average() for cond2 in conditions}
    fig2 = mne.viz.plot_compare_evokeds(
        evokeds1, picks=pick_ch,
        title=f"GO ERP overlay ({pick_ch if isinstance(pick_ch, str) else pick_ch[0]})",
    )
    fig2
    return (fig2,)
'''
@app.cell
def plot_erp_overlay(epochs, conditions, mne, project_path, bids_path, plt):
    """Generate ERP overlay plot and save as PNG"""
    from pathlib import Path as _Path
    _pp = _Path(project_path)
    _fig_dir2 = _pp / "derivatives" / "segment" / "figures" / "GO" / bids_path.basename
    _fig_dir2.mkdir(parents=True, exist_ok=True)
    print(f"\nGenerating ERP overlay plot...")
    # Checking if E6 exists, otherwise taking EEG channel
    if "E6" in epochs.ch_names:
        pick_ch = ["E6"]
        ch_label = "E6"
    else:
        eeg_channels = [ch for ch in epochs.ch_names 
                       if ch.startswith('E') or ch.startswith('eeg')]
        if eeg_channels:
            pick_ch = [eeg_channels[5] if len(eeg_channels) > 5 else eeg_channels[0]]
            ch_label = pick_ch[0]
        else:
            pick_ch = "eeg"
            ch_label = "EEG"
    try:
        _evokeds1 = {_cond4: epochs[_cond4].average() for _cond4 in conditions}
        _fig2 = mne.viz.plot_compare_evokeds(
            _evokeds1, picks=pick_ch,
            title=f"GO ERP overlay ({ch_label})",
            show=False
        )
        # figure
        _fig_path2 = _fig_dir2 / f"erp_overlay_{ch_label}.png"
        # Handling both figure types (matplotlib or array of axes)
        if isinstance(_fig2, tuple):
            _fig2[0].savefig(str(_fig_path2), dpi=150, bbox_inches='tight')
            plt.close(_fig2[0])
        else:
            _fig2.savefig(str(_fig_path2), dpi=150, bbox_inches='tight')
            plt.close(_fig2)
        print(f"  ✓ Saved: {_fig_path2.name}")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        _fig_path2 = None
    return (_fig_path2,)

'''
@app.cell
def plot_pupil_left_overlay(epochs, conditions, mne):
    # Check if pupil_left channel exists
    fig = None
    if "pupil_left" in epochs.ch_names:
        try:
            evokeds = {cond: epochs[cond].average() for cond in conditions}
            fig = mne.viz.plot_compare_evokeds(
                evokeds, picks=["pupil_left"],
                title="GO pupil_left overlay",
            )
        except ValueError as e:
            print(f"Cannot plot pupil_left: {e}")
    else:
        import marimo as _mo
        _mo.md("*No pupil_left channel found (eye-tracking not available)*")
    return (fig,)
'''

@app.cell
def plot_pupil_left_overlay(epochs, conditions, mne, project_path, bids_path, plt):
    """Generate pupil_left overlay plot if available"""
    from pathlib import Path as _Path
    import marimo as _mo
    _pp = _Path(project_path)
    _fig_dir3 = _pp / "derivatives" / "segment" / "figures" / "GO" / bids_path.basename
    _fig_dir3.mkdir(parents=True, exist_ok=True)
    _fig_path3 = None
    if "pupil_left" in epochs.ch_names:
        print(f"\nGenerating pupil_left overlay plot...")
        try:
            _evokeds2 = {_cond3: epochs[_cond3].average() for _cond3 in conditions}
            _fig3 = mne.viz.plot_compare_evokeds(
                _evokeds2, picks=["pupil_left"],
                title="GO pupil_left overlay",
                show=False
            )
            _fig_path3 = _fig_dir3 / "pupil_left_overlay.png"
            if isinstance(_fig3, tuple):
                _fig3[0].savefig(str(_fig_path3), dpi=150, bbox_inches='tight')
                plt.close(_fig3[0])
            else:
                _fig3.savefig(str(_fig_path3), dpi=150, bbox_inches='tight')
                plt.close(_fig3)
            print(f"  ✓ Saved: {_fig_path3.name}")
        except ValueError as e:
            print(f"  ✗ Cannot plot pupil_left: {e}")
    else:
        print("\nNo pupil_left channel (eye-tracking not available)")
    return (_fig_path3,)


'''
@app.cell
def plot_tfr(epochs, conditions, mne, np):
    freqs = np.arange(2, 51, 1)
    n_cycles = freqs / 2.0

    tfr_results = {}
    for cond in conditions:
        power, itc = mne.time_frequency.tfr_morlet(
            epochs[cond], freqs=freqs, n_cycles=n_cycles,
            return_itc=True, picks="eeg",
        )
        tfr_results[cond] = (power, itc)

    for cond3, (power, itc) in tfr_results.items():
        power.plot(title=f"TFR Power: {cond3}", picks="eeg")
        itc.plot(title=f"ITC: {cond3}", picks="eeg")

    return (tfr_results,)
'''


@app.cell
def plot_tfr(epochs, conditions, mne, np, project_path, bids_path, plt):
    """Generate time-frequency analysis plots"""
    from pathlib import Path as _Path
    _pp = _Path(project_path)
    _fig_dir1 = _pp / "derivatives" / "segment" / "figures" / "GO" / bids_path.basename
    _fig_dir1.mkdir(parents=True, exist_ok=True)
    print(f"\nGenerating TFR plots...")
    freqs = np.arange(2, 51, 1)
    n_cycles = freqs / 2.0

    tfr_results = {}
    for _cond2 in conditions:
        print(f"  Processing {_cond2}...")
        try:
            power, itc = mne.time_frequency.tfr_morlet(
                epochs[_cond2], freqs=freqs, n_cycles=n_cycles,
                return_itc=True, picks="eeg", verbose=False
            )
            tfr_results[_cond2] = (power, itc)
            # Plotting and save power
            fig_power = power.plot(
                title=f"TFR Power: {_cond2}", 
                picks="eeg",
                show=False
            )
            fig_path_power = _fig_dir1 / f"tfr_power_{_cond2}.png"
            if isinstance(fig_power, list):
                fig_power[0].savefig(str(fig_path_power), dpi=150, bbox_inches='tight')
                for f in fig_power:
                    plt.close(f)
            else:
                fig_power.savefig(str(fig_path_power), dpi=150, bbox_inches='tight')
                plt.close(fig_power)
            print(f"    ✓ Saved power: {fig_path_power.name}")
            fig_itc = itc.plot(
                title=f"ITC: {_cond2}",
                picks="eeg",
                show=False
            )
            fig_path_itc = _fig_dir1 / f"tfr_itc_{_cond2}.png"
            if isinstance(fig_itc, list):
                fig_itc[0].savefig(str(fig_path_itc), dpi=150, bbox_inches='tight')
                for f in fig_itc:
                    plt.close(f)
            else:
                fig_itc.savefig(str(fig_path_itc), dpi=150, bbox_inches='tight')
                plt.close(fig_itc)
            print(f"    ✓ Saved ITC: {fig_path_itc.name}")
        except Exception as e:
            print(f"    ✗ Error: {e}")

    return (tfr_results,)


@app.cell
def summary(project_path, bids_path):
    """Print summary of saved files"""
    from pathlib import Path as _Path
    import marimo as _mo
    _pp = _Path(project_path)
    _fig_dir5 = _pp / "derivatives" / "segment" / "figures" / "GO" / bids_path.basename
    print("\n" + "="*60)
    print("SEGMENTATION COMPLETE")
    print("="*60)
    # Counting generated figures
    if _fig_dir5.exists():
        png_files = list(_fig_dir5.glob("*.png"))
        print(f"\n✓ Generated {len(png_files)} figure(s) in:")
        print(f"  {_fig_dir5}")
        for pngfile in sorted(png_files):
            print(f"    - {pngfile.name}")
    else:
        print("\n No figures directory created")
    return None

if __name__ == "__main__":
    app.run()
