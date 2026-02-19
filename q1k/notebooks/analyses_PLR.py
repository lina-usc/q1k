"""PLR task — pupil/EEG correlation analysis with publication figure."""

import marimo

__generated_with = "0.10.0"
app = marimo.App(width="full")


@app.cell
def imports():
    import warnings
    from pathlib import Path

    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt
    import mne
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from tqdm import tqdm

    from q1k.analyses.tools import load_participants_tsv
    from q1k.io import get_epoch_files, get_project_site_path

    warnings.filterwarnings("ignore")
    mne.set_log_level(verbose="CRITICAL")
    return (
        mne, np, pd, plt, sns, gridspec, Path, tqdm, warnings,
        load_participants_tsv, get_epoch_files, get_project_site_path,
    )


@app.cell
def parameters():
    # Parameters — adjust for your environment
    participants_tsv = ""  # Path to participants.tsv

    task = "PLR"
    test_ch = "E70"
    return participants_tsv, task, test_ch


@app.cell
def load_epochs(get_project_site_path, task):
    """Get epoch file paths for PLR task."""
    root_path = get_project_site_path()
    pattern = (
        f"derivatives/pylossless/derivatives/sync_loss/derivatives/segment/"
        f"derivatives/autorej/epoch_fif_files/{task}/*_eeg_epo.fif"
    )
    epoch_paths = list(root_path.glob(pattern))
    print(f"Found {len(epoch_paths)} epoch files for {task}")
    return (epoch_paths,)


@app.cell
def build_figure(
    mne, np, pd, plt, sns, gridspec, tqdm, warnings,
    epoch_paths, test_ch,
):
    """Build multi-panel PLR publication figure."""
    if not epoch_paths:
        print("No epoch files found.")
        return ()

    def plot_epochs(epochs, axes, tmin=-1, tmax=2.5):
        axes[0].plot(
            epochs.times,
            epochs.get_data("pupil_left").squeeze().T * 1e3,
            alpha=0.1, color="k",
        )
        axes[0].plot(
            epochs.times,
            epochs.get_data("pupil_left").squeeze().T.mean(axis=1) * 1e3,
            color="r",
        )
        axes[0].set_xlim(tmin, tmax)
        axes[0].yaxis.set_label_coords(-0.1, 0.5)

        axes[1].plot(
            epochs.times,
            epochs.get_data(test_ch).squeeze().T * 1e6,
            alpha=0.1, color="k",
        )
        axes[1].plot(
            epochs.times,
            epochs.get_data(test_ch).squeeze().T.mean(axis=1) * 1e6,
            color="r",
        )
        axes[1].set_xlim(tmin, tmax)
        axes[1].set_ylabel("ERP (\u00b5V)")
        axes[1].yaxis.set_label_coords(-0.1, 0.5)

    # Build figure with gridspec
    fig = plt.figure(figsize=(10, 6))

    gs0 = gridspec.GridSpec(
        2, 2, figure=fig, width_ratios=[5, 0.85],
        height_ratios=[4, 2], hspace=0.4,
    )
    gs00 = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs0[0, :])

    row0 = np.array([fig.add_subplot(gs00[0, i]) for i in range(2)])
    row1 = np.array([
        fig.add_subplot(gs00[1, i], sharex=ax) for ax, i in zip(row0, range(2))
    ])
    axes = np.stack([row0, row1])

    for ax in row0:
        ax.tick_params(labelbottom=False)

    gs01 = gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=gs0[1, 0])
    gs02 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs0[1, 1])

    fig.text(0, 1, "a", weight="bold")
    fig.text(0.455, 1, "b", weight="bold")
    fig.text(0, 0.74, "c", weight="bold")
    fig.text(0.455, 0.74, "d", weight="bold")
    fig.text(0, 0.37, "e", weight="bold")
    fig.text(0.78, 0.37, "f", weight="bold")

    gs0.update(left=0.06)
    gs0.update(top=1.0)

    axes_topo = np.array([fig.add_subplot(gs01[i]) for i in range(5)])
    ax_corr_topo = fig.add_subplot(gs02[0])

    # Single subject example
    axes[0, 0].set_title("pupil")
    axes[0, 1].set_title(f"{test_ch} (O1)")

    # Find example subject
    example_epoch = None
    for path in epoch_paths:
        epochs = mne.read_epochs(path)["plro_d"]
        if len(epochs) > 0:
            example_epoch = epochs
            subject = path.name.split("/")[-1].split("_")[0][4:]
            break

    if example_epoch is not None:
        plot_epochs(example_epoch, axes[0], tmin=-0.5, tmax=2.0)
        axes[0, 0].set_ylabel(subject)

    # All participants
    tmin, tmax = -0.5, 2.0
    ax_row = axes[1]

    plr_sigs = []
    erp_sigs = []
    evoked_lst = []
    corr_series = []
    subjects = []

    for path in tqdm(epoch_paths):
        subject = path.name.split("/")[-1].split("_")[0][4:]
        epochs_et = mne.read_epochs(path)["plro_d"]
        if len(epochs_et) == 0:
            continue
        plr_sigs.append(epochs_et.get_data("pupil_left").squeeze().T.mean(axis=1))
        ax_row[0].plot(
            epochs_et.times, plr_sigs[-1] * 1e3, alpha=0.1, color="k",
        )
        ax_row[0].set_xlim(tmin, tmax)

        epochs_eeg = mne.read_epochs(path)["plro_d"]
        ch_names = [ch for ch in epochs_et.ch_names if ch[0] == "E"]
        epochs_eeg = epochs_eeg.pick(ch_names)

        erp_sigs.append(epochs_eeg.get_data(test_ch).squeeze().T.mean(axis=1))
        evoked_lst.append(epochs_eeg.average())

        corrs = [
            np.corrcoef(sig, plr_sigs[-1])[0, 1]
            for sig in epochs_eeg.average().pick(ch_names).data
        ]
        corr_series.append(pd.Series(corrs, index=ch_names))
        subjects.append(subject)

    if not plr_sigs:
        print("No valid PLR epochs found.")
        return ()

    ax_row[0].plot(
        epochs_eeg.times, np.stack(plr_sigs).mean(axis=0) * 1e3, color="r",
    )

    color = sns.color_palette()[0]
    dat = pd.DataFrame(
        np.stack(erp_sigs) * 1e6, columns=epochs_eeg.times,
    ).melt(var_name="time", value_name=f"ERP-{test_ch}")
    sns.lineplot(dat, x="time", y=f"ERP-{test_ch}", ax=ax_row[1], n_boot=100, color=color)
    ax_row[1].tick_params(axis="y", labelcolor=color)
    ax_row[1].set_ylabel(f"ERP {test_ch} (\u00b5V)", color=color)

    axes[0, 0].axvline(x=0, color="r", linestyle="dashed", alpha=0.2)
    axes[0, 1].axvline(x=0, color="r", linestyle="dashed", alpha=0.2)
    axes[1, 0].set_ylabel("All participants")
    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 1].set_xlabel("Time (s)")

    # Topomaps at selected times
    tmax_peak = epochs_eeg.times[np.argmax(np.stack(erp_sigs).mean(0))]
    t_topos = [-0.2, tmax_peak, tmax_peak + 0.1, 0.5, 1.9]
    for t in t_topos:
        axes[1, 1].axvline(x=t, linestyle="dashed", color="k", alpha=0.2)
        axes[1, 0].axvline(x=t, linestyle="dashed", color="k", alpha=0.2)

    montage = mne.channels.make_standard_montage("GSN-HydroCel-128")
    evoked = mne.combine_evoked(evoked_lst, weights="equal")
    evoked.set_montage(montage)
    evoked.plot_topomap(times=t_topos, axes=axes_topo, colorbar=False)

    # Correlation topomap with permutation cluster test
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        cor_data = pd.concat(corr_series, axis=1)
        raw_info = epochs_et.copy().pick(ch_names)
        raw_info.set_montage(montage)

        adjacency, ch_names_adj = mne.channels.find_ch_adjacency(
            raw_info.info, "eeg",
        )
        t_obs, clusters, cluster_pv, H0 = mne.stats.permutation_cluster_1samp_test(
            cor_data.T[ch_names_adj], adjacency=adjacency, seed=0,
        )

        mask_params = dict(markersize=3, markerfacecolor="y")
        mask = cor_data.copy().mean(1).astype(bool)
        mask[:] = False
        for p, cluster in zip(cluster_pv, clusters):
            if p < 0.01:
                mask[np.array(ch_names_adj)[cluster[0]]] = True

        mne.viz.plot_topomap(
            cor_data.mean(1), raw_info.info, axes=ax_corr_topo,
            mask=mask, mask_params=mask_params,
        )
        ax_corr_topo.set_title("ET/EEG corr.")


    return fig, subjects


@app.cell
def save_figure(fig):
    """Save the PLR figure."""
    if fig is not None:
        fig.savefig("PLR_final.png", dpi=300, bbox_inches="tight", pad_inches=0)
        print("Saved PLR_final.png")
    return ()


@app.cell
def site_summary(pd, load_participants_tsv, participants_tsv, subjects):
    """Show sample size per site."""
    if participants_tsv and subjects:
        sites_df = load_participants_tsv(participants_tsv)
        site_counts = (
            sites_df.set_index("participant_id")
            .loc[subjects]
            .groupby("site")
            .count()["sex"]
        )
        print("Sample sizes per site:")
        print(site_counts)
    return ()


if __name__ == "__main__":
    app.run()
