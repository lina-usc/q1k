"""Shared analysis utilities for Q1K white paper analyses.

Provides functions for:
- Computing ITC/power/ERP xarray datasets from epoch files
- Loading participant demographics with site information
- Permutation cluster significance masking for time-frequency plots
"""

from pathlib import Path

import mne
import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm


def compute_itc_power_erp(
    epoch_files,
    conditions,
    roi,
    freqs,
    n_cycles=None,
    decim=2,
    data_out=None,
    recompute=False,
    toffset=0,
    include_epochs=False,
):
    """Compute ITC, power, and ERP xarray datasets from epoch files.

    For each epoch file and condition, computes Morlet TFR and stores
    results as netCDF files. If ``recompute`` is False and output files
    already exist, skips computation.

    Parameters
    ----------
    epoch_files : iterable of Path
        Paths to epoch .fif files.
    conditions : list of str
        Condition labels to extract from epochs.
    roi : list of str
        Channel names for TFR ROI computation.
    freqs : np.ndarray
        Frequencies for Morlet TFR.
    n_cycles : np.ndarray, optional
        Number of cycles per frequency. Defaults to ``freqs / 2``.
    decim : int
        Decimation factor for TFR computation.
    data_out : Path, optional
        Directory for output netCDF files.
    recompute : bool
        If True, recompute even if files exist.
    toffset : float
        Time offset to apply to times coordinates.
    include_epochs : bool
        If True, include raw epoch data in the dataset.

    Returns
    -------
    Path
        Path to the output directory containing netCDF files.
    """
    if n_cycles is None:
        n_cycles = freqs / 2

    if data_out is None:
        data_out = Path("xr_itc")
    data_out = Path(data_out)
    data_out.mkdir(exist_ok=True)

    if not recompute:
        return data_out

    for filepath in tqdm(list(epoch_files)):
        participant = filepath.name.split("_")[0][4:]
        new_epoch = mne.read_epochs(filepath, verbose=False)

        for condition in conditions:
            if condition not in new_epoch.event_id:
                continue
            if len(new_epoch[condition]) == 0:
                continue

            path_out = data_out / f"{participant}_{condition}_ITC.nc"
            epoch_roi = new_epoch[condition].pick(roi)
            power, itc = epoch_roi.compute_tfr(
                method="morlet",
                average=True,
                n_cycles=n_cycles,
                return_itc=True,
                freqs=freqs,
                decim=decim,
                verbose=False,
            )

            itc_array = xr.DataArray(
                [[itc.data.squeeze()]],
                coords={
                    "participant": [participant],
                    "condition": [condition],
                    "freqs": itc.freqs,
                    "times": itc.times - toffset,
                },
            )
            power_array = xr.DataArray(
                [[power.data.squeeze()]],
                coords={
                    "participant": [participant],
                    "condition": [condition],
                    "freqs": power.freqs,
                    "times": power.times - toffset,
                },
            )

            ds_vars = {"ITC": itc_array, "power": power_array}

            # ERP from all channels
            erp = xr.DataArray(
                [[new_epoch[condition].average(picks=["eeg", "misc"]).data]],
                coords={
                    "participant": [participant],
                    "condition": [condition],
                    "channels": new_epoch.average(picks=["eeg", "misc"]).ch_names,
                    "times": new_epoch.times - toffset,
                },
            )
            ds_vars["erp"] = erp

            if include_epochs:
                epochs_array = xr.DataArray(
                    [[new_epoch[condition].get_data()]],
                    coords={
                        "participant": [participant],
                        "condition": [condition],
                        "trials": np.arange(len(new_epoch[condition])),
                        "channels": new_epoch.ch_names,
                        "times": itc.times - toffset,
                    },
                )
                ds_vars["epochs"] = epochs_array

            xr.Dataset(ds_vars).to_netcdf(path_out)

    return data_out


def load_participants_tsv(participants_path):
    """Load participants TSV with site information.

    Parameters
    ----------
    participants_path : str or Path
        Path to a ``participants.tsv`` file with demographic info.

    Returns
    -------
    pd.DataFrame
        DataFrame with ``participant_id`` stripped of ``sub-`` prefix.
    """
    df = pd.read_csv(participants_path, sep="\t")
    df["participant_id"] = df["participant_id"].str.replace("sub-", "")
    return df


def mask_significance(data, ax, cluster_kwargs, imshow_kwargs, alpha=0.3):
    """Overlay significance mask from permutation cluster test on a plot.

    Parameters
    ----------
    data : list of np.ndarray
        Data arrays for permutation_cluster_test (groups to compare).
    ax : matplotlib.axes.Axes
        Axes to overlay the mask on.
    cluster_kwargs : dict
        Keyword arguments for ``mne.stats.permutation_cluster_test``.
    imshow_kwargs : dict
        Keyword arguments for ``ax.imshow`` (extent, aspect, origin).
    alpha : float
        Alpha value for the mask overlay.
    """
    res = mne.stats.permutation_cluster_test(data, **cluster_kwargs)
    F_obs, clusters, cluster_p_values, H0 = res
    mask = np.zeros_like(F_obs).astype(bool)
    for c, p_val in zip(clusters, cluster_p_values):
        if p_val <= 0.05:
            mask = mask | c

    mask = mask.astype(float)
    mask[mask.astype(bool)] = np.nan
    ax.imshow(mask, cmap="grey", alpha=alpha, **imshow_kwargs)


def mask_significance_baseline(
    data_xr, ax, cluster_kwargs, imshow_kwargs, alpha=0.15
):
    """Overlay significance mask comparing active vs baseline periods.

    Splits the xarray data into baseline periods
    (``times < -0.3`` or ``times >= 1.3``) and active period
    (``-0.2 <= times < 1.2``), then runs a cluster permutation test.

    Parameters
    ----------
    data_xr : xr.DataArray
        Time-frequency data with ``times`` and ``freqs`` coordinates.
    ax : matplotlib.axes.Axes
        Axes to overlay the mask on.
    cluster_kwargs : dict
        Keyword arguments for ``mne.stats.permutation_cluster_test``.
    imshow_kwargs : dict
        Keyword arguments for ``ax.imshow``.
    alpha : float
        Alpha value for the mask overlay.
    """
    data = [
        data_xr.sel(
            times=((data_xr.times >= -1) & (data_xr.times < -0.3))
            | ((data_xr.times >= 1.3) & (data_xr.times < 2.0))
        ).values,
        data_xr.sel(
            times=(data_xr.times >= -0.2) & (data_xr.times < 1.2)
        ).values,
    ]

    res = mne.stats.permutation_cluster_test(data, **cluster_kwargs)
    F_obs, clusters, cluster_p_values, H0 = res

    sub_mask = np.zeros_like(F_obs).astype(bool)
    for c, p_val in zip(clusters, cluster_p_values):
        if p_val <= 0.05:
            sub_mask = sub_mask | c

    mask = np.zeros(data_xr.shape[-2:])
    mask[:, (data_xr.times >= -0.2) & (data_xr.times < 1.2)] = sub_mask
    mask = mask.astype(float)
    mask[mask.astype(bool)] = np.nan

    ax.imshow(mask, cmap="grey", alpha=alpha, **imshow_kwargs)
