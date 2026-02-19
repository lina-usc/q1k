"""VEP task — ITC time-frequency analysis with between-site comparison."""

import marimo

__generated_with = "0.10.0"
app = marimo.App(width="full")


@app.cell
def imports():
    import warnings
    from pathlib import Path

    import matplotlib.pyplot as plt
    import mne
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import xarray as xr
    from scipy.stats import ks_2samp
    from tqdm import tqdm

    from q1k.analyses.tools import (
        load_participants_tsv,
        mask_significance,
        mask_significance_baseline,
    )
    from q1k.io import get_epoch_files

    warnings.filterwarnings("ignore")
    mne.set_log_level(verbose="CRITICAL")
    return (
        mne, np, pd, plt, sns, xr, Path, tqdm, ks_2samp,
        load_participants_tsv, mask_significance, mask_significance_baseline,
        get_epoch_files,
    )


@app.cell
def parameters():
    # Parameters — adjust for your environment
    participants_tsv = ""  # Path to participants.tsv
    recompute = False

    roi = ["E75"]
    decim = 1
    conditions = ["sv06_d", "sv15_d"]
    return participants_tsv, recompute, roi, decim, conditions


@app.cell
def setup(np, Path):
    freqs = np.arange(2, 50, 0.5)
    n_cycles = freqs / 2
    data_out = Path("xr_itc")
    data_out.mkdir(exist_ok=True)
    return freqs, n_cycles, data_out


@app.cell
def compute_tfr(
    mne, np, xr, tqdm,
    get_epoch_files, recompute, roi, decim, conditions,
    freqs, n_cycles, data_out,
):
    """Compute ITC/power/epochs per participant, save as netCDF."""
    if recompute:
        epoch_files = get_epoch_files(task="VEP")
        for filepath in tqdm(list(epoch_files)):
            participant = filepath.name.split("_")[0][4:]
            new_epoch = mne.read_epochs(filepath, verbose=False)
            for condition in conditions:
                path_out = data_out / f"{participant}_{condition[:-2]}_ITC.nc"
                epoch_roi = new_epoch[condition].pick(roi)
                power, itc = epoch_roi.compute_tfr(
                    method="morlet", average=True,
                    n_cycles=n_cycles, return_itc=True, freqs=freqs,
                    decim=decim, verbose=False,
                )
                itc_array = xr.DataArray(
                    [[itc.data.squeeze()]],
                    coords={
                        "participant": [participant],
                        "condition": [condition[:-2]],
                        "freqs": itc.freqs, "times": itc.times,
                    },
                )
                power_array = xr.DataArray(
                    [[itc.data.squeeze()]],
                    coords={
                        "participant": [participant],
                        "condition": [condition[:-2]],
                        "freqs": itc.freqs, "times": itc.times,
                    },
                )
                epochs_array = xr.DataArray(
                    [[new_epoch[condition].get_data()]],
                    coords={
                        "participant": [participant],
                        "condition": [condition[:-2]],
                        "trials": np.arange(len(new_epoch[condition])),
                        "channels": new_epoch.ch_names,
                        "times": itc.times,
                    },
                )
                xr.Dataset({
                    "ITC": itc_array, "power": power_array, "epochs": epochs_array,
                }).to_netcdf(path_out)
        print("ITC/power computation complete")
    else:
        print("Skipping computation (recompute=False). Using existing netCDF files.")
    return ()


@app.cell
def load_data(np, xr, pd, load_participants_tsv, participants_tsv, data_out):
    """Load the xarray dataset and add site information."""
    nc_files = list(data_out.glob("*.nc"))
    if not nc_files:
        print("No netCDF files found. Set recompute=True to generate them.")
        dataset = None
        sites_df = pd.DataFrame()
        return dataset, sites_df

    dataset = xr.open_mfdataset(str(data_out / "*.nc"), data_vars="minimal")

    if participants_tsv:
        sites_df = load_participants_tsv(participants_tsv)
        site_xr = xr.DataArray(
            sites_df["site"], coords={"participant": sites_df["participant_id"]},
        )
        dataset["sites"] = site_xr
    else:
        sites_df = pd.DataFrame()
        print("Warning: No participants.tsv path set.")

    return dataset, sites_df


@app.cell
def plot_itc(
    np, plt, mne, dataset, roi,
    mask_significance, mask_significance_baseline,
):
    """Build the 4x3 ITC time-frequency figure."""
    if dataset is None:
        print("No dataset loaded.")
        return ()

    n_permutations = 1000
    kind = "ITC"

    fig, axes = plt.subplots(4, 3, figsize=(8, 8), sharex=True, sharey=True)

    cluster_kwargs = dict(
        out_type="mask", n_permutations=n_permutations, tail=1, seed=4,
    )

    extent = [
        dataset[kind].times.values[0], dataset[kind].times.values[-1],
        dataset[kind].freqs.values[0], dataset[kind].freqs.values[-1],
    ]
    imshow_kwargs = dict(extent=extent, aspect="auto", origin="lower")

    y_labels = ["6Hz", "15Hz", "Across conditions", "Contrast 6Hz-15Hz"]
    for ax, y_label in zip(axes.T[0], y_labels):
        ax.set_ylabel(f"{y_label}\nFrequency (Hz)")

    sites = np.unique(dataset.sites.dropna(dim="participant"))
    for site, ax_col in zip(sites, axes.T):
        data_xr = dataset[kind].sel(
            participant=dataset["sites"] == site,
            times=dataset.times[
                (dataset.times * 1000).round().astype(int) % 10 == 0
            ],
        )
        data_xr.load()

        for ax, cond in zip(ax_col[:2], data_xr.condition):
            pos = ax.imshow(
                data_xr.mean("participant").sel(condition=cond),
                vmin=0.08, vmax=0.68, cmap="Reds", **imshow_kwargs,
            )
            fig.colorbar(pos, ax=ax)
            mask_significance_baseline(
                data_xr.sel(condition=cond), ax, cluster_kwargs, imshow_kwargs,
            )

        ax_col[0].set_title(f"ITC {roi} - {site.upper()}")
        ax_col[-1].set_xlabel("Time (s)")

        for ax, factor, vmin, vmax, cmap in zip(
            ax_col[2:], [1, -1], [0.08, -0.25], [0.68, 0.25], ["Reds", "RdBu_r"],
        ):
            contrast = data_xr.sel(condition="sv06") + factor * data_xr.sel(condition="sv15")
            if factor == 1:
                contrast /= 2
            pos = ax.imshow(
                contrast.mean("participant"), vmin=vmin, vmax=vmax,
                cmap=cmap, **imshow_kwargs,
            )
            fig.colorbar(pos, ax=ax)

        # Significance: 6Hz vs 15Hz
        mask_significance(
            data_xr.transpose("condition", "participant", "freqs", "times").values,
            ax_col[3], cluster_kwargs, imshow_kwargs,
        )

        for freq, color, axes_ in zip(
            [6, 15], ["k", "lime"],
            [ax_col[[0, 2, 3]], ax_col[[1, 2, 3]]],
        ):
            for y in np.arange(freq, 50, freq):
                for ax in axes_:
                    ax.plot([0, 0.7], [y] * 2, linestyle="dashed", alpha=0.5, color=color)

        # Significance: mean conditions vs baseline
        sum_cond = data_xr.mean("condition").transpose("participant", "freqs", "times")
        mask_significance_baseline(sum_cond, ax_col[2], cluster_kwargs, imshow_kwargs)

    # Between-site differences
    ax_col = axes.T[-1]
    data_sites = {
        "HSJ": dataset[kind].sel(participant=dataset["sites"] == "HSJ"),
        "MNI": dataset[kind].sel(participant=dataset["sites"] == "MNI"),
    }

    for ax, cond in zip(ax_col[:2], data_xr.condition):
        pos = ax.imshow(
            data_sites["HSJ"].mean("participant").sel(condition=cond)
            - data_sites["MNI"].mean("participant").sel(condition=cond),
            vmin=-0.15, vmax=0.15, cmap="RdBu_r", **imshow_kwargs,
        )
        fig.colorbar(pos, ax=ax)

        mask_significance(
            [
                data_sites["HSJ"].sel(condition=cond)
                .transpose("participant", "freqs", "times").values,
                data_sites["MNI"].sel(condition=cond)
                .transpose("participant", "freqs", "times").values,
            ],
            ax, cluster_kwargs, imshow_kwargs,
        )

    ax_col[0].set_title("Between-site differences")
    ax_col[-1].set_xlabel("Time (s)")

    for ax, factor, vmin, vmax in zip(
        ax_col[2:], [1, -1], [-0.15, -0.25], [0.15, 0.25],
    ):
        contrast_hsj = (
            data_sites["HSJ"].sel(condition="sv06")
            + factor * data_sites["HSJ"].sel(condition="sv15")
        )
        contrast_mni = (
            data_sites["MNI"].sel(condition="sv06")
            + factor * data_sites["MNI"].sel(condition="sv15")
        )
        if factor == 1:
            contrast_hsj /= 2
            contrast_mni /= 2

        pos = ax.imshow(
            contrast_hsj.mean("participant") - contrast_mni.mean("participant"),
            vmin=vmin, vmax=vmax, cmap="RdBu_r", **imshow_kwargs,
        )
        fig.colorbar(pos, ax=ax)

        mask_significance(
            [contrast_hsj.values, contrast_mni.values],
            ax, cluster_kwargs, imshow_kwargs,
        )

    for freq, color, axes_ in zip(
        [6, 15], ["k", "lime"],
        [ax_col[[0, 2, 3]], ax_col[[1, 2, 3]]],
    ):
        for y in np.arange(freq, 50, freq):
            for ax in axes_:
                ax.plot([0, 0.7], [y] * 2, linestyle="dashed", alpha=0.5, color=color)

    axes[0, 0].set_xlim(-0.19, 1.19)
    fig.tight_layout()

    return (fig,)


@app.cell
def save_figure(fig):
    """Save the VEP figure."""
    if fig is not None:
        fig.savefig("VEP.png", dpi=300)
        print("Saved VEP.png")
    return ()


@app.cell
def sample_sizes(np, dataset):
    """Print sample sizes per site."""
    if dataset is not None and "sites" in dataset:
        n_mni = int(np.sum(dataset["sites"] == "MNI"))
        n_hsj = int(np.sum(dataset["sites"] == "HSJ"))
        print(f"Sample sizes: MNI={n_mni}; HSJ={n_hsj}")
    return ()


if __name__ == "__main__":
    app.run()
