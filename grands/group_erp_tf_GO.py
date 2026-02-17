import marimo

app = marimo.App()


@app.cell
def __():
    import mne
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.signal import medfilt
    from q1k.io import get_epoch_files
    return mne, plt, np, medfilt, get_epoch_files


@app.cell
def __(get_epoch_files):
    # Parameters
    task = "GO"
    conditions = ["dtgc", "dtbc", "dtoc"]
    roi = ["E83"]
    decim = 2
    freqs = np.arange(2, 50, 2)
    n_cycles = freqs / 2

    # Get epoch files
    epoch_files = get_epoch_files(task, file_pattern="*epo.fif")
    print(f"Found {len(epoch_files)} epoch files for {task}")
    for item in epoch_files:
        print(f"  {item}")
    return task, conditions, roi, decim, freqs, n_cycles, epoch_files


@app.cell
def __(mne, np, medfilt, epoch_files, conditions, roi, freqs, n_cycles, decim):
    # Load and process epoch files with eye-tracking data
    averaging_dict = {label: [] for label in conditions}

    def remove_spikes(data, threshold=0.5, kernel_size=17):
        """Detect and correct spikes using median filtering."""
        mean = np.mean(data)
        std = np.std(data)
        z_scores = np.abs((data - mean) / std)
        outlier_indices = np.where(z_scores > threshold)[0]
        filtered_data = medfilt(data, kernel_size=kernel_size)
        cleaned_data = data.copy()
        cleaned_data[outlier_indices] = filtered_data[outlier_indices]
        return cleaned_data

    for filepath in epoch_files:
        print(f"Loading: {filepath}")
        epochs = mne.read_epochs(filepath)
        epochs = epochs[conditions]

        # Check all required conditions are present
        required_conditions = {"dtbc", "dtoc", "dtgc"}
        available_conditions = set(epochs.event_id.keys())
        if not required_conditions.issubset(available_conditions):
            print(
                f"Missing conditions: {required_conditions - available_conditions}... skipping"
            )
            continue

        # Get raw data
        data = epochs.get_data()

        # Process x position (baseline correct and compute distance)
        x_data = epochs.get_data(picks=["xpos_left"])
        x_baseline_mean = np.mean(x_data[:, :, 800:1000], axis=2, keepdims=True)
        x_data_adjusted = x_data - x_baseline_mean

        # Process y position
        y_data = epochs.get_data(picks=["ypos_left"])
        y_baseline_mean = np.mean(y_data[:, :, 800:1000], axis=2, keepdims=True)
        y_data_adjusted = y_data - y_baseline_mean

        # Compute distance from origin
        origin_distance = np.sqrt(x_data_adjusted**2 + y_data_adjusted**2)
        velocity = np.diff(origin_distance, axis=-1)
        velocity = np.pad(velocity, ((0, 0), (0, 0), (0, 1)), mode="constant", constant_values=0)
        total_travel = np.cumsum(velocity, axis=1)

        # Insert computed signals into data array
        ch_id = epochs.ch_names.index("distance")
        data[:, ch_id, :] = origin_distance.squeeze(1)
        ch_id = epochs.ch_names.index("x_head")
        data[:, ch_id, :] = velocity.squeeze(1)
        ch_id = epochs.ch_names.index("y_head")
        data[:, ch_id, :] = total_travel.squeeze(1)

        # Create new Epochs object with modified data
        new_epochs = mne.EpochsArray(
            data,
            info=epochs.info,
            events=epochs.events,
            event_id=epochs.event_id,
            tmin=epochs.tmin,
        )

        # Rename channels
        new_epochs.rename_channels({"distance": "origin_dist", "x_head": "velocity", "y_head": "total_travel"})
        epochs = new_epochs

        # Clean velocity channel
        channels = ["velocity"]
        for ch in channels:
            idx = epochs.ch_names.index(ch)
            epochs._data[:, idx, :] = np.apply_along_axis(
                remove_spikes, axis=1, arr=epochs._data[:, idx, :]
            )

        # Compute TFR for each condition
        for condition in conditions:
            power, itc = mne.time_frequency.tfr_morlet(
                epochs[condition].pick(roi),
                n_cycles=n_cycles,
                return_itc=True,
                freqs=freqs,
                decim=decim,
            )
            averaging_dict[condition].append(
                (epochs[condition].average(picks=["eeg", "misc"]), power, itc)
            )

    print("Loaded all epochs")
    return averaging_dict


@app.cell
def __(mne, np, averaging_dict, conditions):
    # Plot grand average ERPs with topomaps
    def condition_summary(condition_label):
        print(f"Working on: {condition_label}")
        grand_average = mne.grand_average(
            [item[0] for item in averaging_dict[condition_label]]
        )
        grand_average.plot()
        times = np.arange(0, 1.0, 0.1)
        fig = grand_average.plot_topomap(times=times, colorbar=True)
        fig.suptitle(condition_label)
        return grand_average

    grand_averages = {}
    for condition in conditions:
        grand_averages[condition] = condition_summary(condition)

    return condition_summary, grand_averages


@app.cell
def __(mne, averaging_dict):
    # Compare ERPs across conditions on E6
    color_dict = {"dtgc": "blue", "dtbc": "gray", "dtoc": "red"}
    linestyle_dict = {"dtgc": "-", "dtbc": "-", "dtoc": "-"}

    evokeds = {
        "dtgc": [item[0] for item in averaging_dict["dtgc"]],
        "dtbc": [item[0] for item in averaging_dict["dtbc"]],
        "dtoc": [item[0] for item in averaging_dict["dtoc"]],
    }

    mne.viz.plot_compare_evokeds(
        evokeds,
        combine="mean",
        legend="lower right",
        picks="E6",
        show_sensors="upper right",
        colors=color_dict,
        linestyles=linestyle_dict,
        title="gap vs baseline vs overlap",
    )


@app.cell
def __(mne, averaging_dict):
    # Compare origin distance channel
    color_dict = {"dtgc": "blue", "dtbc": "gray", "dtoc": "red"}
    linestyle_dict = {"dtgc": "-", "dtbc": "-", "dtoc": "-"}

    evokeds = {
        "dtgc": [item[0] for item in averaging_dict["dtgc"]],
        "dtbc": [item[0] for item in averaging_dict["dtbc"]],
        "dtoc": [item[0] for item in averaging_dict["dtoc"]],
    }

    mne.viz.plot_compare_evokeds(
        evokeds,
        combine="mean",
        legend="lower right",
        picks="origin_dist",
        colors=color_dict,
        linestyles=linestyle_dict,
        title="gap vs baseline vs overlap",
    )


@app.cell
def __(mne, averaging_dict):
    # Compare velocity channel
    color_dict = {"dtgc": "blue", "dtbc": "gray", "dtoc": "red"}
    linestyle_dict = {"dtgc": "-", "dtbc": "-", "dtoc": "-"}

    evokeds = {
        "dtgc": [item[0] for item in averaging_dict["dtgc"]],
        "dtbc": [item[0] for item in averaging_dict["dtbc"]],
        "dtoc": [item[0] for item in averaging_dict["dtoc"]],
    }

    mne.viz.plot_compare_evokeds(
        evokeds,
        combine="mean",
        legend="lower right",
        picks="velocity",
        colors=color_dict,
        linestyles=linestyle_dict,
        title="gap vs baseline vs overlap",
    )


@app.cell
def __(mne, np, averaging_dict, freqs):
    # Time-frequency analysis with permutation cluster test
    def do_power_plotting(ersp=True):
        indexer = 1 if ersp else 2
        cond1 = mne.grand_average([item[indexer] for item in averaging_dict["dtgc"]])
        cond2 = mne.grand_average([item[indexer] for item in averaging_dict["dtbc"]])
        cond3 = mne.grand_average([item[indexer] for item in averaging_dict["dtoc"]])

        epochs_power_1 = np.array([item[indexer].data for item in averaging_dict["dtgc"]])[:, 0, :, :]
        epochs_power_2 = np.array([item[indexer].data for item in averaging_dict["dtbc"]])[:, 0, :, :]
        epochs_power_3 = np.array([item[indexer].data for item in averaging_dict["dtoc"]])[:, 0, :, :]

        times = 1e3 * averaging_dict["dtgc"][0][1].times
        fig1, (ax1t, ax1b, ax1c) = plt.subplots(3, 1, figsize=(6, 6))
        fig1.subplots_adjust(0.12, 0.08, 0.96, 0.94, 0.2, 0.35)

        ax1t.imshow(
            epochs_power_1.mean(axis=0),
            extent=[times[0], times[-1], freqs[0], freqs[-1]],
            aspect="auto",
            origin="lower",
            cmap="RdBu_r",
        )
        ax1t.set_title("Gap")

        ax1b.imshow(
            epochs_power_2.mean(axis=0),
            extent=[times[0], times[-1], freqs[0], freqs[-1]],
            aspect="auto",
            origin="lower",
            cmap="RdBu_r",
        )
        ax1b.set_title("Baseline")

        ax1c.imshow(
            epochs_power_3.mean(axis=0),
            extent=[times[0], times[-1], freqs[0], freqs[-1]],
            aspect="auto",
            origin="lower",
            cmap="RdBu_r",
        )

        ax1t.set_ylabel("Frequency (Hz)")
        ax1c.set_title("Overlap")
        ax1c.set_xlabel("Time (ms)")

    # Generate plots
    do_power_plotting(ersp=True)
    do_power_plotting(ersp=False)


if __name__ == "__main__":
    app.run()
