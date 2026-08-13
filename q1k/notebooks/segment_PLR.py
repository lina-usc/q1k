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

    return (project_path, task_id, subject_id, session_id, run_id, derivative_base)

@app.cell
def imports():
    import warnings
    import matplotlib.pyplot as plt
    import mne
    import mne_bids
    import numpy as np
    warnings.filterwarnings("ignore")

    from q1k.segment.tasks import TASK_PARAMS, segment_plr
    return (mne, mne_bids, np, plt, warnings, segment_plr, TASK_PARAMS)

@app.cell
def header(subject_id, task_id):
    import marimo as mo
    mo.md(f"# Single Subject Segmentation Q1K - PLR\n\n"
          f"**Subject:** {subject_id} | **Task:** {task_id}")
    return (mo,)

@app.cell
def load_data(mne_bids, project_path, subject_id, session_id, task_id, derivative_base, run_id,):
    # Direct path construction (verified working)
    #root_path = Path(project_path) / "derivatives" / "sync_loss" / task_id
    #data_path = root_path / f"sub-{subject_id}" / f"ses-{session_id}" / "eeg" / f"sub-{subject_id}_ses-{session_id}_task-{task_id}_run-{run_id}_eeg.edf"
    
    #print(f"Loading data from: {data_path}")
    #print(f"File exists: {data_path.exists()}")
    #print(f"Root path: {root_path}")
    from pathlib import Path as _Path

    _pp = _Path(project_path)
    if derivative_base == "sync_loss":
        input_root = _pp / "derivatives" / "sync_loss" / task_id
    else:
        input_root = _pp / "derivatives" / derivative_base / task_id

    
    bids_path = mne_bids.BIDSPath(
        subject=subject_id, 
        session=session_id, 
        task=task_id,
        run=run_id, 
        datatype="eeg", 
        suffix="eeg", 
        root=str(input_root),
    )
    print(f"Loading data from: {bids_path.fpath}")
    
    eeg_raw = mne_bids.read_raw_bids(bids_path=bids_path, verbose=False)
  
    return eeg_raw, bids_path






@app.cell
def inspect_and_plot_channels(eeg_raw, mne):
    # Show channel types and names
    raw_channel_types = eeg_raw.get_channel_types()
    print("Channel Types:", raw_channel_types)
    print("Channel Names:", eeg_raw.info['ch_names'])
    
    # Define channel groups of interest (matching original)
    frontal = ["E19", "E11", "E4", "E12", "E5"]
    occipital = ["E61", "E62", "E78", "E67", "E72", "E77"]
    din = ["DIN"]
    #pupil = ["pupil_left"]
    #x_pos = ["xpos_left"]
    #y_pos = ["ypos_left"]
    _pupil_name = next((ch for ch in ("pupil_left", "pupil_right") if ch in epochs.ch_names), None)
    _xpos_name = next((ch for ch in ("xpos_left", "xpos_right") if ch in epochs.ch_names), None)
    _ypos_name = next((ch for ch in ("ypos_left", "ypos_right") if ch in epochs.ch_names), None)
    '''
    # Scale dict for plotting
    scale_dict = dict(eeg=1e-4, eyegaze=30, pupil=30)
    
    # Plot raw data with selected channels
    picks_idx = mne.pick_channels(
        eeg_raw.ch_names, 
        din + frontal + occipital + pupil + x_pos + y_pos, 
        ordered=True
    )
    eeg_raw.plot(start=0, duration=20, order=picks_idx, scalings=scale_dict)'''
    
    return frontal, occipital, din, pupil, x_pos, y_pos, scale_dict


@app.cell
def get_events(mne, eeg_raw):
    eeg_events, eeg_event_dict = mne.events_from_annotations(eeg_raw)
    return eeg_events, eeg_event_dict


@app.cell
def debug_events(eeg_events, eeg_event_dict):
    print("Event dict keys:", list(eeg_event_dict.keys()))
    
    # Look for PLR-specific events
    if 'plro_d' in eeg_event_dict:
        plro_events = eeg_events[eeg_events[:,2] == eeg_event_dict['plro_d']]
        print(f"Found {len(plro_events)} plro_d events")
        print(f"First 5 'plro_d' event samples: {plro_events[:5, 0]}")
    else:
        print("No 'plro' events found in event dict")
        print("Available events:", list(eeg_event_dict.keys()))
    
    if 'DIN2' in eeg_event_dict:
        din2_events = eeg_events[eeg_events[:,2] == eeg_event_dict['DIN2']]
        print(f"Found {len(din2_events)} DIN2 events")
    
    return
@app.cell
def create_epochs(segment_plr, eeg_raw, eeg_events, eeg_event_dict):
    epochs, event_id, conditions = segment_plr(
        eeg_raw, eeg_events, eeg_event_dict,
    )
    return epochs, event_id, conditions
'''
@app.cell
def create_epochs(mne, eeg_raw, eeg_events, eeg_event_dict, project_path, pylossless_path, sync_loss_path, segment_path, bids_path, Path):
    # Filter for 'plro_d' events only (matching original)
    plro_d_dict = {key: value for key, value in eeg_event_dict.items() if key == 'plro_d'}
    
    # Create epochs
    epochs = mne.Epochs(
        eeg_raw, 
        eeg_events, 
        event_id=plro_d_dict, 
        tmin=-2, 
        tmax=4.0, 
        on_missing='warn', 
        event_repeated='drop'
    )
    
    print(epochs)  # instead of display(epochs)
    
    # Save epochs to FIF file
    epochs_out_dir = Path(project_path) / segment_path / task_id / f"sub-{subject_id}" / f"ses-{session_id}" / "eeg" / "epoch_fif_files" 
    epochs_out_dir.mkdir(parents=True, exist_ok=True)
    epochs_out_file = epochs_out_dir / f"{bids_path.basename}_epo.fif"
    epochs.save(str(epochs_out_file), overwrite=True)
    
    print(f"Epochs saved to: {epochs_out_file}")
    
    return epochs, plro_d_dict
'''

@app.cell
def save_epochs(epochs, bids_path, project_path, task_id,
                derivative_base):
    from pathlib import Path as _Path

    #epochs.drop_bad()

    _pp = _Path(project_path)
    if derivative_base == "sync_loss":
        seg_path = _pp / "derivatives" / "segment"
    else:
        seg_path = _pp / "derivatives" / derivative_base

    out_dir = seg_path / "epoch_fif_files" / task_id
    out_dir.mkdir(parents=True, exist_ok=True)

    out_file = out_dir / f"{bids_path.basename}_epo.fif"
    epochs.save(str(out_file), overwrite=True)

    return (out_file,)



'''
@app.cell
def create_evoked_and_save(epochs, mne, project_path, pylossless_path, sync_loss_path, segment_path, bids_path, Path):
    
    # Check channel types in epochs (matching original)
    epochs_channel_types = epochs.get_channel_types()
    print("EEG Channel Types:", epochs_channel_types)
    print("EEG Channel Names:", epochs.info['ch_names'])
    # Check what channel types you actually have
    print("Channel types dict:", dict(zip(epochs.ch_names, epochs.get_channel_types())))
    #picks = mne.pick_types(epochs.info, meg=False, eeg=True, misc=True)

    picks_correct = mne.pick_types(epochs.info, meg=False, eeg=True, misc=True)
    print(f"Selected {len(picks_correct)} channels")
    
    # Create evoked for 'plro_d'
    evokeds = {'plro_d': epochs['plro_d'].average(picks=['eeg', 'misc'])}
    #evokeds = {'plro_d': epochs['plro_d'].average(picks)}
    # Save evoked to FIF file
    erp_out_dir = Path(project_path) / segment_path / task_id / f"sub-{subject_id}" / f"ses-{session_id}" / "eeg" / "erp_fif_files" 
    erp_out_dir.mkdir(parents=True, exist_ok=True)
    erp_out_file = erp_out_dir / f"{bids_path.basename}_erp.fif"
    
    mne.write_evokeds(str(erp_out_file), list(evokeds.values()), overwrite=True)
    
    print(f"Evoked saved to: {erp_out_file}")
    
    return evokeds









@app.cell
def plot_erp_joint(evokeds):
    # Plot ERP envelopes and topographies (matching original)
    evokeds['plro_d'].plot_joint(picks=['eeg'], title='6Hz ERP')
    
    return


@app.cell
def plot_erp_overlays(evokeds, mne):
    # Plot ERP overlay for EEG channel E70 (matching original)
    mne.viz.plot_compare_evokeds(evokeds, picks=['E70'], combine='mean')


    #evokeds_pupillist = {'plro_d': [evokeds['plro_d']]}
    # Plot ERP overlay for pupil channel (matching original)
    mne.viz.plot_compare_evokeds(evokeds, picks=['pupil_left'], combine='mean')

    
    return
'''




@app.cell
def compute_and_plot_tfr(epochs, mne, np, plt):
    ch_name = 'E70'
    
    decim = 2
    freqs = np.arange(2, 50, 2)  # define frequencies of interest
    n_cycles = freqs / 2
    
    pow_1, itc_1 = mne.time_frequency.tfr_morlet(
        epochs['plro_d'],
        freqs,
        picks=ch_name,
        n_cycles=n_cycles,
        decim=decim,
        return_itc=True,
        average=True,
    )
    
    itc_dat_1 = itc_1.data[0, :, :]  # only 1 channel as 3D matrix
    pow_dat_1 = pow_1.data[0, :, :]  # only 1 channel as 3D matrix
    
    times = 1e3 * epochs['plro_d'].times  # change unit to ms
    
    fig1, (ax1t, ax1b) = plt.subplots(2, 1, figsize=(6, 4))
    fig1.subplots_adjust(0.12, 0.08, 0.96, 0.94, 0.2, 0.43)
    
    ax1t.imshow(
        pow_dat_1,
        extent=[times[0], times[-1], freqs[0], freqs[-1]],
        aspect="auto",
        origin="lower",
        cmap="RdBu_r",
    )
    
    ax1b.imshow(
        itc_dat_1,
        extent=[times[0], times[-1], freqs[0], freqs[-1]],
        aspect="auto",
        origin="lower",
        cmap="RdBu_r",
    )
    
    ax1t.set_ylabel("Frequency (Hz)")
    ax1t.set_title(f"6Hz Induced power ({ch_name})")
    ax1b.set_title(f"6Hz Inter Trial Coherence ({ch_name})")
    ax1b.set_xlabel("Time (ms)")
    
    plt.show()
    
    return pow_1, itc_1

'''
@app.cell
def pupil_diagnostic_unique(eeg_raw, epochs, evokeds, mne):
    """Diagnostic with unique variable names - no conflicts"""
    import matplotlib.pyplot as plt_diag
    
    print("=" * 70)
    print("PUPIL DIAGNOSTIC")
    print("=" * 70)
    
    # =========================================================
    # 1. CHECK RAW DATA
    # =========================================================
    print("\n1. RAW DATA CHECK:")
    raw_has_pupil = 'pupil_left' in eeg_raw.ch_names
    if raw_has_pupil:
        pupil_raw_data, pupil_raw_times = eeg_raw['pupil_left']
        print(f"   ✓ pupil_left found in raw data")
        print(f"   Data range: {pupil_raw_data.min():.6f} to {pupil_raw_data.max():.6f}")
        print(f"   Mean: {pupil_raw_data.mean():.6f}")
        
        diag_fig1, diag_ax1 = plt_diag.subplots(figsize=(10, 3))
        diag_ax1.plot(pupil_raw_times, pupil_raw_data[0])
        diag_ax1.set_title("Raw Pupil Data (first 10 seconds)")
        diag_ax1.set_xlabel("Time (s)")
        diag_ax1.set_ylabel("Pupil diameter")
        diag_ax1.grid(True, alpha=0.3)
        plt_diag.show()
    else:
        print("   ✗ pupil_left NOT found in raw data!")
    
    # =========================================================
    # 2. CHECK EPOCHS
    # =========================================================
    print("\n2. EPOCHS CHECK:")
    epochs_has_pupil = 'pupil_left' in epochs.ch_names
    if epochs_has_pupil:
        pupil_epochs_data = epochs.get_data(picks=['pupil_left'])
        print(f"   ✓ pupil_left found in epochs")
        print(f"   Shape: {pupil_epochs_data.shape}")
        print(f"   Data range: {pupil_epochs_data.min():.6f} to {pupil_epochs_data.max():.6f}")
        
        diag_fig2, diag_ax2 = plt_diag.subplots(figsize=(10, 4))
        for i in range(min(3, len(pupil_epochs_data))):
            diag_ax2.plot(epochs.times, pupil_epochs_data[i, 0, :], label=f'Epoch {i+1}')
        diag_ax2.axvline(x=0, color='r', linestyle='--', label='Stimulus')
        diag_ax2.set_xlabel('Time (s)')
        diag_ax2.set_ylabel('Pupil diameter')
        diag_ax2.set_title('Pupil Data in Epochs (first 3 epochs)')
        diag_ax2.legend()
        diag_ax2.grid(True, alpha=0.3)
        plt_diag.show()
    else:
        print("   ✗ pupil_left NOT found in epochs!")
    
    # =========================================================
    # 3. CHECK CURRENT EVOKED (YOUR EXISTING METHOD)
    # =========================================================
    print("\n3. YOUR CURRENT EVOKED CHECK:")
    current_evoked_has_pupil = 'pupil_left' in evokeds['plro_d'].ch_names
    
    if current_evoked_has_pupil:
        pupil_evoked_data = evokeds['plro_d'].get_data(picks=['pupil_left'])[0]
        pupil_evoked_times = evokeds['plro_d'].times
        print(f"   ✓ pupil_left IS in your evoked!")
        print(f"   Data range: {pupil_evoked_data.min():.6f} to {pupil_evoked_data.max():.6f}")
        
        baseline_val = pupil_evoked_data[pupil_evoked_times < 0].mean()
        post_val = pupil_evoked_data[pupil_evoked_times > 0].mean()
        print(f"   Baseline mean: {baseline_val:.6f}")
        print(f"   Post-stimulus mean: {post_val:.6f}")
        
        diag_fig3, diag_ax3 = plt_diag.subplots(figsize=(10, 4))
        diag_ax3.plot(pupil_evoked_times, pupil_evoked_data)
        diag_ax3.axvline(x=0, color='r', linestyle='--', label='Stimulus')
        diag_ax3.set_title("Pupil from YOUR evoked object")
        diag_ax3.set_xlabel("Time (s)")
        diag_ax3.set_ylabel("Pupil diameter")
        diag_ax3.legend()
        diag_ax3.grid(True, alpha=0.3)
        plt_diag.show()
    else:
        print("    pupil_left NOT in your evoked!")
        print(f"   Available channels: {evokeds['plro_d'].ch_names[:10]}...")
    
    # =========================================================
    # 4. CREATE TEST EVOKED WITH CORRECT PICKS
    # =========================================================
    print("\n4. TEST EVOKED (with correct picks = pick_types):")
    diag_picks = mne.pick_types(epochs.info, meg=False, eeg=True, misc=True)
    print(f"   Selected {len(diag_picks)} channels")
    
    test_evoked_obj = epochs['plro_d'].average(picks=diag_picks)
    test_has_pupil = 'pupil_left' in test_evoked_obj.ch_names
    
    if test_has_pupil:
        test_pupil_data = test_evoked_obj.get_data(picks=['pupil_left'])[0]
        test_pupil_times = test_evoked_obj.times
        print(f"   ✓ pupil_left found in TEST evoked!")
        print(f"   Data range: {test_pupil_data.min():.6f} to {test_pupil_data.max():.6f}")
        
        test_baseline = test_pupil_data[test_pupil_times < 0].mean()
        test_post = test_pupil_data[test_pupil_times > 0].mean()
        print(f"   Baseline mean: {test_baseline:.6f}")
        print(f"   Post-stimulus mean: {test_post:.6f}")
        
        diag_fig4, diag_ax4 = plt_diag.subplots(figsize=(10, 4))
        diag_ax4.plot(test_pupil_times, test_pupil_data)
        diag_ax4.axvline(x=0, color='r', linestyle='--', label='Stimulus')
        diag_ax4.set_title('Pupil from CORRECTLY created evoked')
        diag_ax4.set_xlabel('Time (s)')
        diag_ax4.set_ylabel('Pupil diameter')
        diag_ax4.legend()
        diag_ax4.grid(True, alpha=0.3)
        plt_diag.show()
        
        # Try plot_compare_evokeds
        test_evokeds_dict = {'plro_d': [test_evoked_obj]}
        mne.viz.plot_compare_evokeds(test_evokeds_dict, picks=['pupil_left'], combine='mean')
    else:
        print("   ✗ pupil_left NOT in test evoked - data issue!")
    
    # =========================================================
    # 5. SUMMARY
    # =========================================================
    print("\n" + "=" * 70)
    print("CONCLUSION:")
    print("=" * 70)
    
    if not raw_has_pupil:
        print(" Pupil not in RAW data - check your recording/sync stage")
    elif not epochs_has_pupil:
        print(" Pupil lost during EPOCHING - check epoch creation parameters")
    elif not current_evoked_has_pupil and test_has_pupil:
        print("Pupil data exists and is good!")
        print("   → The problem is YOUR create_evoked_and_save function")
        print("   → Change: picks=['eeg', 'misc'] to: picks = mne.pick_types(epochs.info, meg=False, eeg=True, misc=True)")
    elif current_evoked_has_pupil:
        print(" Pupil IS in your evoked - the issue is in your plotting code")
        print("   → Try: evokeds_list = {'plro_d': [evokeds['plro_d']]}")
        print("   → Then: mne.viz.plot_compare_evokeds(evokeds_list, picks=['pupil_left'], combine='mean')")
    else:
        print("  Unknown issue - check data quality")
    
    return test_evoked_obj
'''
    
if __name__ == "__main__":
    app.run()
