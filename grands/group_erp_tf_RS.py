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
    from q1k.config import (
        FREQ_BANDS,
        FRONTAL_ROI,
        FRONTAL_LEFT_ROI,
        PARIETAL_LEFT_ROI,
        TEMPORAL_LEFT_ROI,
    )
    return (
        mne, plt, np, pd, sns, get_epoch_files,
        FREQ_BANDS, FRONTAL_ROI,
        FRONTAL_LEFT_ROI, PARIETAL_LEFT_ROI, TEMPORAL_LEFT_ROI,
    )


@app.cell
def __(get_epoch_files, FREQ_BANDS, FRONTAL_ROI):
    # Parameters
    task = "RS"
    frontal_roi = FRONTAL_ROI
    common_freqs = FREQ_BANDS

    # Get epoch files
    epoch_files = get_epoch_files(task, file_pattern="*epo.fif")
    print(f"Found {len(epoch_files)} epoch files for {task}")
    return task, frontal_roi, common_freqs, epoch_files


@app.cell
def __(mne, np, pd, epoch_files, frontal_roi, common_freqs):
    # Process resting state data: compute PSD and extract band power
    band_specific_map = []
    whole_group_eeg = []
    whole_brain_band_map = []

    print(f"Processing {len(epoch_files)} files...")
    for filepath in epoch_files:
        print(f"Loading: {filepath}")
        new_epoch = mne.read_epochs(filepath, preload=True)

        # Compute power spectral density
        epo_power = new_epoch.compute_psd(fmin=0, fmax=50)
        psds, freqs = epo_power.get_data(return_freqs=True)
        avg_psd = psds.mean(0)

        psd_df = pd.DataFrame(avg_psd)
        psd_df.columns = freqs
        psd_df.index = new_epoch.pick(picks=["eeg"]).ch_names

        # Group-level PSD (whole scalp average)
        group_psd_df = pd.DataFrame(psd_df.mean(0)).reset_index()
        group_psd_df.columns = ["freqs", "psd"]

        # Extract subject ID and diagnosis from file path
        subject_id = filepath.name.split("-")[1].split("_")[0]
        diagnosis = "asd" if "P" in subject_id else "control"

        group_psd_df["subject"] = subject_id
        group_psd_df["diagnosis"] = diagnosis
        group_psd_df["epoch_length"] = len(new_epoch)
        whole_group_eeg.append(group_psd_df)

        # Extract band-specific power
        for band, (fmin, fmax) in common_freqs.items():
            psd_band = psd_df[psd_df.columns[(psd_df.columns > fmin) & (psd_df.columns < fmax)]].mean(1)

            # Whole-brain band power
            wb_band = pd.DataFrame({"psd": [psd_band.mean()]})
            wb_band["band"] = band
            wb_band["epoch_length"] = len(new_epoch)
            wb_band["subject"] = subject_id
            wb_band["diagnosis"] = diagnosis
            whole_brain_band_map.append(wb_band)

            # Frontal ROI band power
            psd_band = pd.DataFrame(psd_band)
            psd_band = psd_band.loc[frontal_roi]
            psd_band.columns = ["psd"]
            psd_band["band"] = band
            psd_band["epoch_length"] = len(new_epoch)
            psd_band["subject"] = subject_id
            psd_band["diagnosis"] = diagnosis
            band_specific_map.append(psd_band)

    print(f"Processed {len(epoch_files)} epoch files")
    return band_specific_map, whole_group_eeg, whole_brain_band_map


@app.cell
def __(pd, whole_group_eeg, sns, plt):
    # Plot whole group EEG PSD by diagnosis
    whole_group_df = pd.concat(whole_group_eeg)
    print(whole_group_df.head())

    sns.lineplot(data=whole_group_df, x="freqs", y="psd", hue="diagnosis", palette="viridis")
    plt.title("Group PSD by Diagnosis (Whole Scalp)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power")
    return whole_group_df,


@app.cell
def __(pd, band_specific_map, whole_brain_band_map, sns, plt):
    # Frontal ROI band power
    group_df = pd.concat(band_specific_map)
    group_df = group_df.groupby(["band", "subject", "epoch_length", "diagnosis"]).mean().reset_index()

    # Whole-brain band power
    wb_df = pd.concat(whole_brain_band_map)

    alpha_df = group_df.loc[group_df.band == "alpha"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.stripplot(data=alpha_df, x="band", y="psd", hue="diagnosis", dodge=True, palette="viridis", ax=axes[0])
    sns.boxplot(data=alpha_df, x="band", y="psd", hue="diagnosis", palette="viridis", ax=axes[0])
    axes[0].set_title("Alpha Power — Frontal ROI")
    axes[0].get_legend().remove()

    wb_alpha = wb_df.loc[wb_df.band == "alpha"]
    sns.stripplot(data=wb_alpha, x="band", y="psd", hue="diagnosis", dodge=True, palette="viridis", ax=axes[1])
    sns.boxplot(data=wb_alpha, x="band", y="psd", hue="diagnosis", palette="viridis", ax=axes[1])
    axes[1].set_title("Alpha Power — Whole Brain")

    fig.tight_layout()
    return group_df, wb_df


@app.cell
def __(whole_group_df, group_df, wb_df):
    # Export results to CSV
    from pathlib import Path

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    whole_group_df.to_csv(results_dir / "rs_group_psd.csv", index=False)
    group_df.to_csv(results_dir / "rs_frontal_band_power.csv", index=False)
    wb_df.to_csv(results_dir / "rs_whole_brain_band_power.csv", index=False)
    print(f"Results saved to {results_dir}/")


if __name__ == "__main__":
    app.run()
