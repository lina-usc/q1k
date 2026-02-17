import marimo

app = marimo.App()


@app.cell
def __():
    import mne
    import matplotlib.pyplot as plt
    import numpy as np
    from q1k.io import get_epoch_files
    return mne, plt, np, get_epoch_files


@app.cell
def __(get_epoch_files):
    # Parameters
    task = "VEP"
    conditions = ["sv06_d", "sv15_d"]
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
def __(mne, np, epoch_files, conditions, roi, freqs, n_cycles, decim):
    # Load and process epoch files
    averaging_dict = {label: [] for label in conditions}

    for filepath in epoch_files:
        print(f"Loading: {filepath}")
        new_epoch = mne.read_epochs(filepath)

        for condition in conditions:
            power, itc = mne.time_frequency.tfr_morlet(
                new_epoch[condition].pick(roi),
                n_cycles=n_cycles,
                return_itc=True,
                freqs=freqs,
                decim=decim,
            )
            averaging_dict[condition].append(
                (new_epoch[condition].average(picks=["eeg", "misc"]), power, itc)
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
def __(mne, averaging_dict, conditions):
    # Compare ERPs across conditions
    color_dict = {"sv06_d": "blue", "sv15_d": "red"}
    linestyle_dict = {"sv06_d": "-", "sv15_d": "-"}

    evokeds = {
        "sv06_d": [item[0] for item in averaging_dict["sv06_d"]],
        "sv15_d": [item[0] for item in averaging_dict["sv15_d"]],
    }

    mne.viz.plot_compare_evokeds(
        evokeds,
        combine="mean",
        legend="lower right",
        picks=["E83"],
        show_sensors="upper right",
        colors=color_dict,
        linestyles=linestyle_dict,
        title="6Hz vs. 15Hz ERPs",
    )


@app.cell
def __(mne, averaging_dict):
    # Compare ERPs on pupil channel
    color_dict = {"sv06_d": "blue", "sv15_d": "red"}
    linestyle_dict = {"sv06_d": "-", "sv15_d": "-"}

    evokeds = {
        "sv06_d": [item[0] for item in averaging_dict["sv06_d"]],
        "sv15_d": [item[0] for item in averaging_dict["sv15_d"]],
    }

    mne.viz.plot_compare_evokeds(
        evokeds,
        combine="mean",
        legend="lower right",
        picks="pupil_left",
        colors=color_dict,
        linestyles=linestyle_dict,
        title="6Hz vs. 15Hz ERPs",
    )


@app.cell
def __(mne, np, averaging_dict, freqs):
    # Time-frequency analysis with permutation cluster test
    def do_power_plotting(ersp=True):
        indexer = 1 if ersp else 2
        cond1 = mne.grand_average([item[indexer] for item in averaging_dict["sv06_d"]])
        cond2 = mne.grand_average([item[indexer] for item in averaging_dict["sv15_d"]])

        epochs_power_1 = np.array([item[indexer].data for item in averaging_dict["sv06_d"]])[:, 0, :, :]
        epochs_power_2 = np.array([item[indexer].data for item in averaging_dict["sv15_d"]])[:, 0, :, :]

        times = 1e3 * averaging_dict["sv06_d"][0][1].times
        fig1, (ax1t, ax1b) = plt.subplots(2, 1, figsize=(6, 4))
        fig1.subplots_adjust(0.12, 0.08, 0.96, 0.94, 0.2, 0.43)

        ax1t.imshow(
            epochs_power_1.mean(axis=0),
            extent=[times[0], times[-1], freqs[0], freqs[-1]],
            aspect="auto",
            origin="lower",
            cmap="RdBu_r",
        )

        ax1b.imshow(
            epochs_power_2.mean(axis=0),
            extent=[times[0], times[-1], freqs[0], freqs[-1]],
            aspect="auto",
            origin="lower",
            cmap="RdBu_r",
        )

        ax1t.set_ylabel("Frequency (Hz)")
        ax1t.set_title("target Induced power 06Hz")
        ax1b.set_title("target Induced power 15Hz")
        ax1b.set_xlabel("Time (ms)")

        F_obs, clusters, cluster_p_values, H0 = mne.stats.permutation_cluster_test(
            [epochs_power_1, epochs_power_2],
            out_type="mask",
            n_permutations=100,
            threshold=6.0,
            tail=0,
        )

        times = 1e3 * averaging_dict["sv06_d"][0][1].times

        evoked_power_contrast = epochs_power_1.mean(axis=0) - epochs_power_2.mean(axis=0)
        signs = np.sign(evoked_power_contrast)

        F_obs_plot = np.nan * np.ones_like(F_obs)
        for c, p_val in zip(clusters, cluster_p_values):
            if p_val <= 0.05:
                F_obs_plot[c] = F_obs[c] * signs[c]
        max_F = np.nanmax(np.abs(F_obs_plot))

        fig, (ax, ax2) = plt.subplots(2, 1, figsize=(6, 4))
        ax.imshow(
            F_obs,
            extent=[times[0], times[-1], freqs[0], freqs[-1]],
            aspect="auto",
            origin="lower",
            cmap="gray",
        )

        ax.imshow(
            F_obs_plot,
            extent=[times[0], times[-1], freqs[0], freqs[-1]],
            aspect="auto",
            origin="lower",
            cmap="RdBu_r",
            vmin=-max_F,
            vmax=max_F,
        )
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Frequency (Hz)")
        ax.set_title("Induced power")

        evoked_contrast = mne.combine_evoked([cond1, cond2], weights=[1, -1])
        evoked_contrast.plot(axes=ax2)

    # Generate plots
    do_power_plotting(ersp=True)
    do_power_plotting(ersp=False)


if __name__ == "__main__":
    app.run()
