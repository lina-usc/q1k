import marimo
import mne

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

    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    import mne
    import mne_bids
    import numpy as np
    warnings.filterwarnings("ignore")

    from q1k.io import get_segment_path, get_sync_loss_path
    from q1k.segment.tasks import TASK_PARAMS, segment_vep
    return (mne, mne_bids, np, plt, warnings, segment_vep,
            TASK_PARAMS, get_sync_loss_path, get_segment_path, matplotlib)


@app.cell
def header(subject_id, task_id):
    import marimo as mo
    mo.md(f"# Single Subject Segmentation Q1K - VEP\n\n"
          f"**Subject:** {subject_id} | **Task:** {task_id}")
    return (mo,)


@app.cell
def load_data(mne, mne_bids, project_path, subject_id, session_id,
              task_id, derivative_base):
    from pathlib import Path

    pp = Path(project_path)
    # Use sync_loss directly (not nested under pylossless)
    if derivative_base == "sync_loss":
        input_root = pp / "derivatives" / "sync_loss"
    else:
        input_root = pp / "derivatives" / derivative_base

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
    """Get events from TSV file or annotations"""
    import pandas as pd
    # Trying to load from TSV first
    events_fname = bids_path.copy().update(suffix='events', extension='.tsv').fpath
    if events_fname.exists():
        print(f"Reading events from: {events_fname}")
        df = pd.read_csv(events_fname, sep='\t')
        sfreq = eeg_raw.info['sfreq']
        samples = (df['onset'].values * sfreq).astype(int)
        durations = np.zeros(len(samples), dtype=int)
        unique_types = sorted(df['trial_type'].unique())
        type_to_id = {t: i+1 for i, t in enumerate(unique_types)}
        event_ids = np.array([type_to_id[t] for t in df['trial_type']], dtype=int)
        eeg_events = np.column_stack([samples, durations, event_ids])
        eeg_event_dict = type_to_id
        print(f"✓ Found {len(unique_types)} unique event types from TSV")
    else:
        # Fall back to annotations
        print("TSV not found, using annotations")
        eeg_events, eeg_event_dict = mne.events_from_annotations(eeg_raw)
        print(f"✓ Found {len(np.unique(eeg_events[:, 2]))} unique event types from annotations")
    print(f"  Total events: {len(eeg_events)}")
    print(f"  Event types: {list(eeg_event_dict.keys())[:5]}...")
    return eeg_events, eeg_event_dict


@app.cell
def create_epochs(segment_vep, eeg_raw, eeg_events, eeg_event_dict):
    print("Creating epochs...")
    epochs, event_id, conditions = segment_vep(
        eeg_raw, eeg_events, eeg_event_dict,
    )
    print(f"✓ Created {len(epochs.events)} epochs")
    print(f"  Conditions: {conditions}")
    print(f"  Event IDs: {event_id}")
    return epochs, event_id, conditions


@app.cell
def save_epochs(epochs, bids_path, project_path, task_id):
    from pathlib import Path

    pp = Path(project_path)
    seg_path = pp / "derivatives" / "segment"
    out_dir = seg_path / "epoch_fif_files" / task_id
    out_dir.mkdir(parents=True, exist_ok=True)
    epochs_clean = epochs.copy().drop_bad()
    print(f"Epochs after dropping bad: {len(epochs_clean)} / {len(epochs)}")
    out_file = out_dir / f"{bids_path.basename}_epo.fif"
    epochs_clean.save(str(out_file), overwrite=True)
    print(f"✓ Saved epochs to: {out_file}")
    return (out_file,)


@app.cell
def plot_erp_joint(epochs, conditions, project_path, bids_path, plt):
    """Generate ERP joint plots and save as PNG files"""
    from pathlib import Path
    pp = Path(project_path)
    fig_dir = pp / "derivatives" / "segment" / "figures" / "VEP" / bids_path.basename
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("\nGenerating ERP joint plots...")
    figs = []
    for cond in conditions:
        try:
            evoked = epochs[cond].average()
            fig = evoked.plot_joint(
                title=f"ERP: {cond}",
                show=False
            )
            fig_path = fig_dir / f"erp_joint_{cond}.png"
            fig.savefig(str(fig_path), dpi=150, bbox_inches='tight')
            plt.close(fig)
            figs.append(fig_path)
            print(f"  ✓ Saved: {fig_path.name}")
        except Exception as e:
            print(f"  ✗ Error plotting {cond}: {e}")
    return (figs,)


@app.cell
def plot_erp_overlay(epochs, conditions, mne, project_path, bids_path, plt):
    """Generate ERP overlay plot - VEP typically uses occipital channels"""
    from pathlib import Path
    pp = Path(project_path)
    fig_dir = pp / "derivatives" / "segment" / "figures" / "VEP" / bids_path.basename
    fig_dir.mkdir(parents=True, exist_ok=True)
    print("\nGenerating ERP overlay plot...")
    # Check for E70 (occipital), otherwise find suitable channel
    if "E70" in epochs.ch_names:
        pick_ch = ["E70"]
        ch_label = "E70"
    elif "Oz" in epochs.ch_names:
        pick_ch = ["Oz"]
        ch_label = "Oz"
    else:
        # Find any occipital-like channel
        occipital_ch = [ch for ch in epochs.ch_names
                       if any(occ in ch for occ in ['O', 'occip', 'E70', 'E75'])]
        if occipital_ch:
            pick_ch = [occipital_ch[0]]
            ch_label = occipital_ch[0]
        else:
            # Fallback to any EEG channel
            eeg_channels = [ch for ch in epochs.ch_names
                           if ch.startswith('E') or ch.startswith('eeg')]
            if eeg_channels:
                pick_ch = [eeg_channels[0]]
                ch_label = eeg_channels[0]
            else:
                pick_ch = "eeg"
                ch_label = "EEG"
    print(f"  Using channel: {ch_label}")
    try:
        evokeds = {cond: epochs[cond].average() for cond in conditions}
        fig = mne.viz.plot_compare_evokeds(
            evokeds, picks=pick_ch,
            title=f"VEP ERP overlay ({ch_label})",
            show=False
        )
        fig_path = fig_dir / f"erp_overlay_{ch_label}.png"
        if isinstance(fig, tuple):
            fig[0].savefig(str(fig_path), dpi=150, bbox_inches='tight')
            plt.close(fig[0])
        else:
            fig.savefig(str(fig_path), dpi=150, bbox_inches='tight')
            plt.close(fig)
        print(f"  ✓ Saved: {fig_path.name}")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        fig_path = None
    return (fig_path,)


@app.cell
def plot_pupil_overlay(epochs, conditions, mne, project_path, bids_path, plt):
    """Generate pupil overlay plots if available"""
    from pathlib import Path
    pp = Path(project_path)
    fig_dir = pp / "derivatives" / "segment" / "figures" / "VEP" / bids_path.basename
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig_paths = []
    # Check for pupil channels
    pupil_channels = [ch for ch in epochs.ch_names if 'pupil' in ch.lower()]
    if pupil_channels:
        print("\nGenerating pupil overlay plots...")
        for pupil_ch in pupil_channels:
            try:
                evokeds = {cond: epochs[cond].average() for cond in conditions}
                fig = mne.viz.plot_compare_evokeds(
                    evokeds, picks=[pupil_ch],
                    title=f"VEP {pupil_ch} overlay",
                    show=False
                )
                fig_path = fig_dir / f"pupil_{pupil_ch}_overlay.png"
                if isinstance(fig, tuple):
                    fig[0].savefig(str(fig_path), dpi=150, bbox_inches='tight')
                    plt.close(fig[0])
                else:
                    fig.savefig(str(fig_path), dpi=150, bbox_inches='tight')
                    plt.close(fig)
                fig_paths.append(fig_path)
                print(f"  ✓ Saved: {fig_path.name}")
            except Exception as e:
                print(f"  ✗ Error plotting {pupil_ch}: {e}")
    else:
        print("\nNo pupil channels (eye-tracking not available)")
    return (fig_paths,)


@app.cell
def plot_tfr(epochs, conditions, mne, np, project_path, bids_path, plt):
    """Generate time-frequency analysis plots"""
    from pathlib import Path
    pp = Path(project_path)
    fig_dir = pp / "derivatives" / "segment" / "figures" / "VEP" / bids_path.basename
    fig_dir.mkdir(parents=True, exist_ok=True)
    print("\nGenerating TFR plots...")
    freqs = np.arange(2, 51, 1)
    n_cycles = freqs / 2.0

    tfr_results = {}
    for cond in conditions:
        print(f"  Processing {cond}...")
        try:
            power, itc = mne.time_frequency.tfr_morlet(
                epochs[cond], freqs=freqs, n_cycles=n_cycles,
                return_itc=True, picks="eeg", verbose=False
            )
            tfr_results[cond] = (power, itc)
            # Plot and save power
            fig_power = power.plot(
                title=f"TFR Power: {cond}",
                picks="eeg",
                show=False
            )
            fig_path_power = fig_dir / f"tfr_power_{cond}.png"
            if isinstance(fig_power, list):
                fig_power[0].savefig(str(fig_path_power), dpi=150, bbox_inches='tight')
                for f in fig_power:
                    plt.close(f)
            else:
                fig_power.savefig(str(fig_path_power), dpi=150, bbox_inches='tight')
                plt.close(fig_power)
            print(f"    ✓ Saved power: {fig_path_power.name}")
            # Plot and save ITC
            fig_itc = itc.plot(
                title=f"ITC: {cond}",
                picks="eeg",
                show=False
            )
            fig_path_itc = fig_dir / f"tfr_itc_{cond}.png"
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
    from pathlib import Path
    pp = Path(project_path)
    fig_dir = pp / "derivatives" / "segment" / "figures" / "VEP" / bids_path.basename
    print("\n" + "="*60)
    print("SEGMENTATION COMPLETE")
    print("="*60)
    if fig_dir.exists():
        png_files = list(fig_dir.glob("*.png"))
        print(f"\n✓ Generated {len(png_files)} figure(s) in:")
        print(f"  {fig_dir}")
        for pngfile in sorted(png_files):
            print(f"    - {pngfile.name}")
    else:
        print("\n No figures directory created")
    return None


if __name__ == "__main__":
    app.run()
