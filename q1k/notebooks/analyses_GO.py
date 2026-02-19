"""GO task — ITC/power/ERP xarray computation and multi-panel publication figure."""

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
    import xarray as xr
    from tqdm import tqdm

    from q1k.analyses.tools import load_participants_tsv
    from q1k.io import get_epoch_files

    warnings.filterwarnings("ignore")
    mne.set_log_level(verbose="CRITICAL")
    return mne, np, pd, plt, xr, Path, tqdm, load_participants_tsv, get_epoch_files


@app.cell
def parameters():
    # Parameters — adjust these for your environment
    participants_tsv = ""  # Path to participants.tsv
    recompute = False  # Set True to recompute ITC/power from epochs

    roi = ["E83"]
    decim = 2
    toffset = 0
    return participants_tsv, recompute, roi, decim, toffset


@app.cell
def setup(np, Path):
    freqs = np.arange(2, 50, 2)
    n_cycles = freqs / 2
    data_out = Path("xr_itc_GO")
    data_out.mkdir(exist_ok=True)
    return freqs, n_cycles, data_out


@app.cell
def compute_tfr(
    mne, np, xr, tqdm,
    get_epoch_files, recompute, roi, decim, freqs, n_cycles, data_out, toffset,
):
    """Compute ITC/power/ERP per participant and condition, save as netCDF."""
    if recompute:
        epoch_files = get_epoch_files(task="GO")
        for filepath in tqdm(list(epoch_files)):
            participant = filepath.name.split("_")[0][4:]
            new_epoch = mne.read_epochs(filepath, verbose=False)
            for condition in new_epoch.event_id.keys():
                if condition[-2:] == "_d":
                    continue
                path_out = data_out / f"{participant}_{condition}_ITC.nc"
                if len(new_epoch[condition]) == 0:
                    continue
                epoch_roi = new_epoch[condition].pick(roi)
                power, itc = epoch_roi.compute_tfr(
                    method="morlet", average=True,
                    n_cycles=n_cycles, return_itc=True, freqs=freqs,
                    decim=decim, verbose=False,
                )
                itc_array = xr.DataArray(
                    [[itc.data.squeeze()]],
                    coords={
                        "participant": [participant], "condition": [condition],
                        "freqs": itc.freqs, "times": itc.times - toffset,
                    },
                )
                power_array = xr.DataArray(
                    [[itc.data.squeeze()]],
                    coords={
                        "participant": [participant], "condition": [condition],
                        "freqs": itc.freqs, "times": itc.times - toffset,
                    },
                )
                erp = xr.DataArray(
                    [[new_epoch[condition].average(picks=["eeg", "misc"]).data]],
                    coords={
                        "participant": [participant], "condition": [condition],
                        "channels": new_epoch.average(picks=["eeg", "misc"]).ch_names,
                        "times": new_epoch.times - toffset,
                    },
                )
                xr.Dataset({"ITC": itc_array, "power": power_array, "erp": erp}).to_netcdf(
                    path_out
                )
        print("ITC/power/ERP computation complete")
    else:
        print("Skipping computation (recompute=False). Using existing netCDF files.")
    return ()


@app.cell
def load_demographics(pd, load_participants_tsv, participants_tsv):
    """Load participant demographics with site info."""
    if participants_tsv:
        sites_df = load_participants_tsv(participants_tsv)
    else:
        sites_df = pd.DataFrame(columns=["participant_id", "site", "sex"])
        print("Warning: No participants.tsv path set. Site info unavailable.")
    return (sites_df,)


@app.cell
def build_figure(plt, np, mne, xr, get_epoch_files, data_out, roi):
    """Build the multi-panel publication figure for GO task."""
    plt.rcParams.update({"font.size": 10})

    def get_axes_panel(fig_, row, col):
        row = 2 - row
        gs_type = fig_.add_gridspec(
            nrows=2, ncols=7,
            left=0.07 + 0.48 * col, right=0.07 + 0.48 * col + 0.40,
            bottom=0.02 + 0.32 * row + 0.01, top=0.02 + 0.32 * row + 0.29,
            wspace=0.1, hspace=1.2, height_ratios=[1.8, 1],
        )
        erps = fig_.add_subplot(gs_type[0, :])
        topos = [fig_.add_subplot(gs_type[1, i]) for i in range(7)]
        return erps, topos

    fig = plt.figure(figsize=(10, 7), facecolor="white")

    axes_erp = np.empty((3, 2), dtype="object")
    axes_topos = np.empty((3, 2), dtype="object")
    for row in range(3):
        for col in range(2):
            axes_erp[row, col], axes_topos[row, col] = get_axes_panel(fig, row, col)

    # Compute grand averages if evoked files don't exist
    def condition_summary(evokeds, ax_erp, ax_topo, title, times=None):
        grand_average = mne.grand_average(evokeds)
        grand_average.plot(axes=ax_erp, xlim=[-0.2, 0.8])
        if times is None:
            times = np.arange(-0.1, 0.51, 0.1)
        grand_average.plot_topomap(
            times=times, colorbar=False, axes=ax_topo,
            sensors=False, time_format="%.2f", vlim=[-4, 4],
        )

    # Conditions: o=overlap, b=baseline, g=gap; dt=stimulus onset, gc=gaze onset
    all_times = [
        [0, 0.13, 0.17, 0.28, 0.38, 0.48, 0.6],
        [-0.1, -0.05, 0.04, 0.12, 0.2, 0.3, 0.5],
    ]

    epoch_files = get_epoch_files(task="GO")
    if epoch_files:
        mne.read_epochs(epoch_files[0], verbose=False)

        for cond_type, erp_ax_col, topo_ax_col in zip(
            "obg", axes_erp, axes_topos
        ):
            for trig_type, ax_erp, ax_topo, times in zip(
                ["dt", "gc"], erp_ax_col, topo_ax_col, all_times
            ):
                condition = f"{trig_type}{cond_type}c"
                try:
                    evoked = mne.read_evokeds(f"evoked_{condition}_ave.fif")
                    condition_summary(evoked, ax_erp, ax_topo, condition, times=times)
                except FileNotFoundError:
                    ax_erp.text(
                        0.5, 0.5, f"No evoked for {condition}",
                        ha="center", va="center", transform=ax_erp.transAxes,
                    )

    axes_erp[0, 0].set_title("Synchronized on stimulus onset")
    axes_erp[0, 1].set_title("Synchronized on gaze onset")

    for i in range(1, 3):
        axes_erp[i, 0].set_title("")
        axes_erp[i, 1].set_title("")

    axes_erp[0, 0].set_ylabel(r"Overlap ($\mu V$)")
    axes_erp[1, 0].set_ylabel(r"Baseline ($\mu V$)")
    axes_erp[2, 0].set_ylabel(r"Gap ($\mu V$)")

    axes_erp[0, 1].set_ylabel("")
    axes_erp[1, 1].set_ylabel("")
    axes_erp[2, 1].set_ylabel("")

    # Remove N count text
    for ax in axes_erp.ravel():
        for text in ax.texts:
            text_str = text.get_text()
            if text_str[:4] == "N$_{":
                text.set_text("")

    # Add dashed lines at topomap times
    for ax_col, times in zip(axes_erp.T, all_times):
        for ax in ax_col:
            for time in times:
                ax.axvline(x=time, alpha=0.3, color="k", linestyle="dashed")

    return (fig,)


@app.cell
def save_figure(fig):
    """Save the GO figure."""
    fig.savefig("GO.png", dpi=300)
    print("Saved GO.png")
    return ()


@app.cell
def site_summary(xr, sites_df, data_out):
    """Show sample size per site."""
    nc_files = list(data_out.glob("*_ITC.nc"))
    if nc_files:
        dataset = xr.open_mfdataset(str(data_out / "*_ITC.nc"), data_vars="minimal")
        if not sites_df.empty:
            site_counts = (
                sites_df.set_index("participant_id")
                .loc[dataset.participant]
                .groupby("site")
                .count()["sex"]
            )
            print("Sample sizes per site:")
            print(site_counts)
        else:
            print(f"Total participants: {len(dataset.participant)}")
    else:
        print("No netCDF files found. Set recompute=True to generate them.")
    return ()


if __name__ == "__main__":
    app.run()
