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
    task = "TO"
    conditions = ["stan1", "stan4", "dev"]
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
        epochs = mne.read_epochs(filepath)
        epochs.apply_baseline((0.2, 0))

        event_names = epochs.event_id.keys()

        # Select events for standard item 1
        selected_events_stan1 = [event for event in event_names if "stan" in event and "item1" in event]
        if selected_events_stan1:
            power, itc = mne.time_frequency.tfr_morlet(
                epochs[selected_events_stan1].pick(roi),
                n_cycles=n_cycles,
                return_itc=True,
                freqs=freqs,
                decim=decim,
            )
            averaging_dict["stan1"].append(
                (epochs[selected_events_stan1].average(picks=["eeg"]), power, itc)
            )

        # Select events for standard item 4
        selected_events_stan4 = [event for event in event_names if "stan" in event and "item4" in event]
        if selected_events_stan4:
            power, itc = mne.time_frequency.tfr_morlet(
                epochs[selected_events_stan4].pick(roi),
                n_cycles=n_cycles,
                return_itc=True,
                freqs=freqs,
                decim=decim,
            )
            averaging_dict["stan4"].append(
                (epochs[selected_events_stan4].average(picks=["eeg"]), power, itc)
            )

        # Select events for deviant
        selected_events_deviant = [event for event in event_names if "deviant" in event and "Dev" in event]
        if selected_events_deviant:
            power, itc = mne.time_frequency.tfr_morlet(
                epochs[selected_events_deviant].pick(roi),
                n_cycles=n_cycles,
                return_itc=True,
                freqs=freqs,
                decim=decim,
            )
            averaging_dict["dev"].append(
                (epochs[selected_events_deviant].average(picks=["eeg"]), power, itc)
            )

    print("Loaded all epochs")
    return averaging_dict


@app.cell
def __(mne, np, averaging_dict, conditions):
    # Plot grand average ERPs with topomaps
    def condition_summary(condition_label):
        if not averaging_dict[condition_label]:
            print(f"No data for: {condition_label}")
            return None

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
    color_dict = {"stan1": "blue", "stan4": "gray", "dev": "red"}
    linestyle_dict = {"stan1": "-", "stan4": "-", "dev": "-"}

    evokeds = {
        "stan1": [item[0] for item in averaging_dict["stan1"]],
        "stan4": [item[0] for item in averaging_dict["stan4"]],
        "dev": [item[0] for item in averaging_dict["dev"]],
    }

    mne.viz.plot_compare_evokeds(
        evokeds,
        combine="mean",
        legend="lower right",
        picks="E11",
        show_sensors="upper right",
        colors=color_dict,
        linestyles=linestyle_dict,
        title="standard item 1 vs standard item 4 vs deviant",
    )


if __name__ == "__main__":
    app.run()
