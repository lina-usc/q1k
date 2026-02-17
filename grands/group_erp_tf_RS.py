import marimo

app = marimo.App()


@app.cell
def __():
    import mne
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from q1k.io import get_epoch_files
    return mne, plt, np, pd, sns, get_epoch_files


@app.cell
def __(get_epoch_files):
    # Parameters
    task = "RS"
    frontal_roi = ["E18", "E19", "E23", "E24", "E27", "E3", "E4", "E10", "E118", "E123", "E124"]
    common_freqs = {
        "delta": (0, 4),
        "theta": (4, 7),
        "alpha": (8, 12),
        "beta": (13, 30),
        "gamma": (30, 45),
    }

    # Get epoch files
    epoch_files = get_epoch_files(task, file_pattern="*epo.fif")
    print(f"Found {len(epoch_files)} epoch files for {task}")
    return task, frontal_roi, common_freqs, epoch_files


@app.cell
def __(mne, np, pd, epoch_files, frontal_roi, common_freqs):
    # Process resting state data: compute PSD and extract band power
    band_specific_map = []
    whole_group_eeg = []

    for filepath in epoch_files[8:40]:  # Process subset of files
        print(f"Loading: {filepath}")
        new_epoch = mne.read_epochs(filepath)

        # Compute power spectral density
        epo_power = new_epoch.compute_psd(fmin=0, fmax=50)
        psds, freqs = epo_power.get_data(return_freqs=True)
        avg_psd = psds.mean(0)

        psd_df = pd.DataFrame(avg_psd)
        psd_df.columns = freqs
        psd_df.index = new_epoch.pick(picks=["eeg"]).ch_names

        # Group-level PSD
        group_psd_df = pd.DataFrame(psd_df.mean(0)).reset_index()
        group_psd_df.columns = ["freqs", "psd"]

        # Extract subject ID and diagnosis from file path
        subject_id = filepath.name.split("-")[1].split("_")[0]
        diagnosis = "asd" if "P" in subject_id else "control"

        group_psd_df["subject"] = subject_id
        group_psd_df["diagnosis"] = diagnosis
        whole_group_eeg.append(group_psd_df)

        # Extract band-specific power for frontal ROI
        for band, (fmin, fmax) in common_freqs.items():
            psd_band = psd_df[psd_df.columns[(psd_df.columns > fmin) & (psd_df.columns < fmax)]].mean(1)
            psd_band = pd.DataFrame(psd_band)
            psd_band = psd_band.loc[frontal_roi]
            psd_band.columns = ["psd"]
            psd_band["band"] = band
            psd_band["epoch_length"] = len(new_epoch)
            psd_band["subject"] = subject_id
            psd_band["diagnosis"] = diagnosis

            band_specific_map.append(psd_band)

    print("Loaded and processed all epochs")
    return band_specific_map, whole_group_eeg


@app.cell
def __(pd, whole_group_eeg, sns):
    # Plot whole group EEG PSD by diagnosis
    whole_group_df = pd.concat(whole_group_eeg)
    print(whole_group_df.head())

    sns.lineplot(data=whole_group_df, x="freqs", y="psd", hue="diagnosis", palette="viridis")
    plt.title("Group PSD by Diagnosis")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power")


@app.cell
def __(pd, band_specific_map, sns):
    # Plot band-specific power for alpha band
    group_df = pd.concat(band_specific_map)
    group_df = group_df.groupby(["band", "subject", "epoch_length", "diagnosis"]).mean().reset_index()

    alpha_df = group_df.loc[group_df.band == "alpha"]

    sns.stripplot(
        data=alpha_df,
        x="band",
        y="psd",
        hue="diagnosis",
        dodge=2,
        palette="viridis",
    )
    plt.title("Alpha Power by Diagnosis (Strip Plot)")

    sns.boxplot(
        data=alpha_df,
        x="band",
        y="psd",
        hue="diagnosis",
        palette="viridis",
    )
    plt.title("Alpha Power by Diagnosis (Box Plot)")


if __name__ == "__main__":
    app.run()
