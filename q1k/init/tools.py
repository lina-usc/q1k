"""EEG/ET initialization tools for BIDS conversion.

Handles event extraction, task-specific event processing, eye-tracking
synchronization, and EEG/ET data combination.
"""

import mne
import numpy as np
import plotly.express as px

VALID_TASKS = ["rest", "as", "ssvep", "vs", "ap",
               "go", "plr", "mn", "nsp", "fsp", "RSRio"]


def get_event_dict(raw, events, offset):
    """Extract event dictionary from raw EEG stimulus channels.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data with stimulus channels.
    events : np.ndarray
        MNE events array.
    offset : int
        Offset for event IDs.

    Returns
    -------
    dict
        Mapping of stimulus channel names to event IDs.
    """
    stim_names = raw.copy().pick("stim").info["ch_names"]
    event_dict = {
        event: int(i) + offset
        for i, event in enumerate(stim_names)
        if event != "STI 014"
    }
    return event_dict


def eeg_event_test(eeg_events, eeg_event_dict, din_str, task_name=None):
    """Process EEG events for a specific task.

    Identifies stimulus onset DIN events by finding task-specific event
    sequences and creating new derived event labels (e.g., ``"ae06_d"``).

    Parameters
    ----------
    eeg_events : np.ndarray
        MNE events array (n_events, 3).
    eeg_event_dict : dict
        Event ID mapping.
    din_str : tuple of str
        DIN channel names to look for (e.g., ``("DIN2", "DIN3")``).
    task_name : str
        One of the valid task names (``"ap"``, ``"go"``, ``"vp"``,
        ``"plr"``, ``"as"``, ``"mn"``, ``"rest"``).

    Returns
    -------
    eeg_events : np.ndarray
        Updated events array with new derived events.
    eeg_stims : np.ndarray
        Stimulus onset events only.
    eeg_iti : np.ndarray
        Inter-trial intervals between stimulus onsets.
    din_offset : list
        DIN offset values.
    eeg_event_dict : dict
        Updated event dictionary with new labels.
    new_events : np.ndarray
        The newly created events.

    Raises
    ------
    ValueError
        If ``task_name`` is not provided or not recognized.
    NotImplementedError
        If ``task_name`` is ``"vs"``, ``"fsp"``, or ``"nsp"``
        (not yet implemented).
    """
    din_offset = []

    if not task_name:
        raise ValueError(
            f"please pass one of {VALID_TASKS}"
            " to the task_name keyword argument."
        )

    if task_name.lower() in ("ap", "aep"):
        eeg_events, eeg_stims, eeg_iti, din_offset, eeg_event_dict, new_events = (
            _process_aep(eeg_events, eeg_event_dict, din_offset)
        )

    elif task_name.lower() == "go":
        eeg_events, eeg_stims, eeg_iti, din_offset, eeg_event_dict, new_events = (
            _process_go(eeg_events, eeg_event_dict, din_str, din_offset)
        )

    elif task_name.lower() in ("vp", "vep"):
        eeg_events, eeg_stims, eeg_iti, din_offset, eeg_event_dict, new_events = (
            _process_vep(eeg_events, eeg_event_dict, din_str, din_offset)
        )

    elif task_name.lower() == "plr":
        mask = np.isin(eeg_events[:, 2], [eeg_event_dict["DIN2"]])
        eeg_stims = eeg_events[mask]
        print(f"Number of stimulus onset DIN events: {len(eeg_stims)}")
        eeg_iti = np.diff(eeg_stims[:, 0])
        new_events = np.empty((0, 3))

    elif task_name.lower() == "as":
        eeg_events, eeg_stims, eeg_iti, din_offset, eeg_event_dict, new_events = (
            _process_as(eeg_events, eeg_event_dict, din_str, din_offset)
        )

    elif task_name.lower() == "mn":
        eeg_events, eeg_stims, eeg_iti, din_offset, eeg_event_dict, new_events = (
            _process_mmn(eeg_events, eeg_event_dict, din_offset)
        )

    elif task_name.lower() in ("rest", "rs"):
        mask = np.isin(eeg_events[:, 2], [eeg_event_dict["DIN2"]])
        eeg_stims = eeg_events[mask]
        print(f"Number of stimulus onset DIN events: {len(eeg_stims)}")
        eeg_iti = np.diff(eeg_stims[:, 0])
        new_events = np.empty((0, 3))

    elif task_name == "RSRio":
        # RS Rio only has "Eye open" and "Comment" events;
        # no DIN offset procedure needed.
        eeg_stims = np.empty((0, 3))
        eeg_iti = np.array([])
        new_events = np.empty((0, 3))
        print("RSRio: skipping DIN offset procedure")

    elif task_name in ("vs", "fsp", "nsp"):
        raise NotImplementedError(f"Task {task_name} is not yet implemented.")

    else:
        raise ValueError(
            f"Could not determine task name."
            f" Expected one of {VALID_TASKS} but got {task_name}"
        )

    return eeg_events, eeg_stims, eeg_iti, din_offset, eeg_event_dict, new_events


def _remove_tsyn(eeg_events, eeg_event_dict):
    """Remove TSYN events from the events array."""
    print("Removing TSYN events...")
    mask = ~np.isin(eeg_events[:, 2], [eeg_event_dict["TSYN"]])
    return eeg_events[mask]


def _find_din_following(eeg_events, eeg_event_dict, trigger_labels,
                        din_labels, din_offset):
    """Find DIN events following specific trigger events.

    Returns new events and updated din_offset list.
    """
    new_events = np.empty((0, 3))
    base_id = len(eeg_event_dict) + 1

    for label_idx, trigger_label in enumerate(trigger_labels):
        trigger_id = eeg_event_dict[trigger_label]
        for i, e in np.ndenumerate(eeg_events[:, 2]):
          if e == trigger_id:
                if i[0] + 1 < len(eeg_events[:, 2]):
                    next_event = eeg_events[i[0] + 1, 2]
                    if any(next_event == eeg_event_dict.get(d) for d in din_labels if d in eeg_event_dict):
                        new_row = np.array([
                            [eeg_events[i[0] + 1, 0], 0, base_id + label_idx]
                        ])
                        new_events = np.append(new_events, new_row, axis=0)
                        din_offset.append(
                            eeg_events[i[0] + 1, 0] - eeg_events[i[0], 0]
                        )

    return new_events, din_offset


def _process_aep(eeg_events, eeg_event_dict, din_offset):
    """Process AEP (auditory evoked potential) task events."""
    eeg_events = _remove_tsyn(eeg_events, eeg_event_dict)
    new_events, din_offset = _find_din_following(
        eeg_events, eeg_event_dict,
        ["ae06", "ae40"], ["DIN4"], din_offset
    )

    eeg_events = np.concatenate((eeg_events, new_events))
    eeg_events = eeg_events[eeg_events[:, 0].argsort()]
    eeg_event_dict["ae06_d"] = len(eeg_event_dict) + 1
    eeg_event_dict["ae40_d"] = len(eeg_event_dict) + 1

    mask = np.isin(eeg_events[:, 2],
                   [eeg_event_dict["ae06_d"], eeg_event_dict["ae40_d"]])
    eeg_stims = eeg_events[mask]
    print(f"Number of stimulus onset DIN events: {len(eeg_stims)}")
    eeg_iti = np.diff(eeg_stims[:, 0])

    return eeg_events, eeg_stims, eeg_iti, din_offset, eeg_event_dict, new_events


def _process_go(eeg_events, eeg_event_dict, din_str, din_offset):
    """Process GO (gap-overlap) task events."""
    # Auto-detect DIN channels: HSJ uses DIN4/DIN5, MHC uses DIN2/DIN3
    din_str = next(
        (d for d in [("DIN2", "DIN3"), ("DIN4", "DIN5"), ("DIN4", "DIN3")]
         if d[0] in eeg_event_dict),
        din_str
    )
    eeg_events = _remove_tsyn(eeg_events, eeg_event_dict)
    new_events = np.empty((0, 3))

    for i, e in np.ndenumerate(eeg_events[:, 2]):
        for label_idx, trigger in enumerate(["dtoc", "dtbc", "dtgc"]):
             if e == eeg_event_dict[trigger]:
               if i[0] + 1 < len(eeg_events[:, 2]):
                    next_event = eeg_events[i[0] + 1, 2]
                    if (next_event == eeg_event_dict[din_str[0]] or
                            next_event == eeg_event_dict[din_str[1]]):
                        new_row = np.array([
                            [eeg_events[i[0] + 1, 0], 0,
                             len(eeg_event_dict) + label_idx + 1]
                        ])
                        new_events = np.append(new_events, new_row, axis=0)
                        din_offset.append(
                            eeg_events[i[0] + 1, 0] - eeg_events[i[0], 0]
                        )

    eeg_events = np.concatenate((eeg_events, new_events))
    eeg_events = eeg_events[eeg_events[:, 0].argsort()]
    eeg_event_dict["dtoc_d"] = len(eeg_event_dict) + 1
    eeg_event_dict["dtbc_d"] = len(eeg_event_dict) + 1
    eeg_event_dict["dtgc_d"] = len(eeg_event_dict) + 1

    mask = np.isin(eeg_events[:, 2], [
        eeg_event_dict["dtoc_d"],
        eeg_event_dict["dtbc_d"],
        eeg_event_dict["dtgc_d"],
    ])
    eeg_stims = eeg_events[mask]
    print(f"Number of stimulus onset DIN events: {len(eeg_stims)}")
    eeg_iti = np.diff(eeg_stims[:, 0])

    return eeg_events, eeg_stims, eeg_iti, din_offset, eeg_event_dict, new_events


def _process_vep(eeg_events, eeg_event_dict, din_str, din_offset):
    """Process VEP (visual evoked potential) task events."""
    eeg_events = _remove_tsyn(eeg_events, eeg_event_dict)
    new_events, din_offset = _find_din_following(
        eeg_events, eeg_event_dict,
        ["sv06", "sv15"], list(din_str), din_offset
    )

    eeg_events = np.concatenate((eeg_events, new_events))
    eeg_events = eeg_events[eeg_events[:, 0].argsort()]
    eeg_event_dict["sv06_d"] = len(eeg_event_dict) + 1
    eeg_event_dict["sv15_d"] = len(eeg_event_dict) + 1

    mask = np.isin(eeg_events[:, 2],
                   [eeg_event_dict["sv06_d"], eeg_event_dict["sv15_d"]])
    eeg_stims = eeg_events[mask]
    print(f"Number of stimulus onset DIN events: {len(eeg_stims)}")
    eeg_iti = np.diff(eeg_stims[:, 0])

    return eeg_events, eeg_stims, eeg_iti, din_offset, eeg_event_dict, new_events


def _process_as(eeg_events, eeg_event_dict, din_str, din_offset):
    """Process AS (anti-saccade) task events."""
    eeg_events = _remove_tsyn(eeg_events, eeg_event_dict)
    new_events = []

    new_devents = {
        eeg_event_dict["ddtr"]: 1,
        eeg_event_dict["ddtl"]: 2,
    }

    for i, e in enumerate(eeg_events[:, 2]):
        if e not in new_devents:
            continue
        if eeg_events[i + 1, 2] in [eeg_event_dict[din_str[0]],
                                     eeg_event_dict[din_str[1]]]:
            new_row = np.array([
                eeg_events[i + 1, 0], 0,
                len(eeg_event_dict) + new_devents[e]
            ])
            new_events.append(new_row)
            din_offset.append(eeg_events[i + 1, 0] - eeg_events[i, 0])

    new_events = np.stack(new_events)
    eeg_events = np.concatenate((eeg_events, new_events))
    eeg_events = eeg_events[eeg_events[:, 0].argsort()]
    eeg_event_dict["ddtr_d"] = len(eeg_event_dict) + 1
    eeg_event_dict["ddtl_d"] = len(eeg_event_dict) + 1

    mask = np.isin(eeg_events[:, 2],
                   [eeg_event_dict["ddtr_d"], eeg_event_dict["ddtl_d"]])
    eeg_stims = eeg_events[mask]
    print(f"Number of stimulus onset DIN events: {len(eeg_stims)}")
    eeg_iti = np.diff(eeg_stims[:, 0])

    return eeg_events, eeg_stims, eeg_iti, din_offset, eeg_event_dict, new_events


def _process_mmn(eeg_events, eeg_event_dict, din_offset):
    """Process MMN (mismatch negativity) task events."""
    eeg_events = _remove_tsyn(eeg_events, eeg_event_dict)
    new_events, din_offset = _find_din_following(
        eeg_events, eeg_event_dict,
        ["mmns", "mmnt"], ["DIN4"], din_offset
    )

    eeg_events = np.concatenate((eeg_events, new_events))
    eeg_events = eeg_events[eeg_events[:, 0].argsort()]
    eeg_event_dict["mmns_d"] = len(eeg_event_dict) + 1
    eeg_event_dict["mmnt_d"] = len(eeg_event_dict) + 1

    mask = np.isin(eeg_events[:, 2],
                   [eeg_event_dict["mmns_d"], eeg_event_dict["mmnt_d"]])
    eeg_stims = eeg_events[mask]
    print(f"Number of stimulus onset DIN events: {len(eeg_stims)}")
    eeg_iti = np.diff(eeg_stims[:, 0])

    return eeg_events, eeg_stims, eeg_iti, din_offset, eeg_event_dict, new_events


def et_event_test(et_raw_df, task_name=""):
    """Process eye-tracking events for a specific task.

    Parameters
    ----------
    et_raw_df : pd.DataFrame
        Eye-tracking data as a DataFrame with a ``"DIN"`` column.
    task_name : str
        Task name (``"vp"``, ``"ssaep"``, ``"plr"``, ``"as"``,
        ``"go"``, ``"mmn"``, ``"rest"``).

    Returns
    -------
    et_raw_df : pd.DataFrame
        Updated DataFrame.
    et_events : pd.DataFrame
        Filtered event rows.
    et_stims : pd.DataFrame
        Stimulus onset events.
    et_iti : pd.Series
        Inter-trial intervals.
    """
    # Fill NaNs in DIN channel
    et_raw_df["DIN"] = et_raw_df["DIN"].fillna(0)

    # Correct single-sample blips while DIN8 is on
    for ind in range(1, len(et_raw_df) - 1):
        if np.all(et_raw_df["DIN"][ind - 1:ind + 2] == [8, 0, 8]):
            et_raw_df.loc[ind, "DIN"] = 8

    # Find DIN value changes
    et_raw_df["DIN_diff"] = et_raw_df["DIN"].diff()
    et_events = et_raw_df.loc[et_raw_df["DIN_diff"] > 0]

    # Handle anomalous DIN values
    et_events = et_events.copy()
    et_events.loc[et_events["DIN"].isin([2, 18, 26]), "DIN"] = 2
    et_events.loc[et_events["DIN"].isin([4, 20, 28]), "DIN"] = 4

    if task_name == "vp":
        et_events, et_stims, et_iti = _et_process_vp(et_raw_df, et_events)
    elif task_name == "ssaep":
        et_events, et_stims, et_iti = _et_process_ssaep(et_events)
    elif task_name == "plr":
        et_events, et_stims, et_iti = _et_process_plr(et_raw_df, et_events)
    elif task_name == "as":
        et_events, et_stims, et_iti = _et_process_as(et_events)
    elif task_name == "go":
        et_events, et_stims, et_iti = _et_process_go(et_raw_df, et_events)
    elif task_name == "mmn":
        et_events, et_stims, et_iti = _et_process_mmn(et_events)
    elif task_name == "rest":
        et_events, et_stims, et_iti = _et_process_rest(et_events)
    else:
        raise ValueError(f"Unknown ET task: {task_name}")

    return et_raw_df, et_events, et_stims, et_iti


def _et_process_vp(et_raw_df, et_events):
    et_events = et_events.copy()
    et_events = et_events.loc[et_raw_df["DIN"].isin([2, 4])]
    et_events = et_events.reset_index()

    for ind in range(len(et_events)):
        if et_events["DIN"][ind] == 4:
            if ind < len(et_events) - 1:
                if et_events["DIN"][ind + 1] == 2:
                    diff = et_events["index"][ind + 1] - et_events["index"][ind]
                    if 180 < diff < 3000:
                        et_events.loc[ind + 1, "DIN_diff"] = 5

    et_stims = et_events.loc[et_events["DIN_diff"].isin([5])]
    print(f"Number of eye-tracking stimulus onset DIN events: {len(et_stims)}")
    et_iti = et_stims["index"].diff()
    return et_events, et_stims, et_iti


def _et_process_ssaep(et_events):
    et_events = et_events.copy()
    et_stims = et_events.loc[et_events["DIN_diff"].isin([8])]
    et_events = et_events.reset_index()

    for ind in range(len(et_events)):
        if ind == 0:
            et_events.loc[ind, "DIN_diff"] = 9
        elif ind < len(et_events) - 1:
            if et_events["index"][ind] - et_events["index"][ind - 1] > 300:
                et_events.loc[ind, "DIN_diff"] = 9

    et_stims = et_events.loc[et_events["DIN_diff"].isin([9])]
    print(f"Number of eye-tracking stimulus onset DIN events: {len(et_stims)}")
    et_iti = et_stims["index"].diff()
    return et_events, et_stims, et_iti


def _et_process_plr(et_raw_df, et_events):
    et_events = et_events.loc[et_raw_df["DIN_diff"].isin([2])]
    et_events = et_events.reset_index()
    et_stims = et_events.loc[et_events["DIN_diff"].isin([2])]
    print(f"Number of eye-tracking stimulus onset DIN events: {len(et_stims)}")
    et_iti = et_stims["index"].diff()
    return et_events, et_stims, et_iti


def _et_process_as(et_events):
    et_events = et_events.reset_index()

    for ind in range(len(et_events) - 2):
        if np.all(et_events["DIN_diff"][ind:ind + 3] == [4, 8, 2]):
            et_events.loc[ind + 2, "DIN_diff"] = 9

    et_stims = et_events.loc[et_events["DIN_diff"].isin([9])]
    print(f"Number of eye-tracking stimulus onset DIN events: {len(et_stims)}")
    et_iti = et_stims["index"].diff()
    return et_events, et_stims, et_iti


def _et_process_go(et_raw_df, et_events):
    for ind in et_events.index:
        if et_events["DIN_diff"][ind] == 12:
            et_events.loc[ind, "DIN_diff"] = 4

    et_events = et_events.copy()
    et_events = et_events.loc[et_raw_df["DIN_diff"].isin([2, 4])]
    et_events = et_events.reset_index()

    for ind in range(len(et_events)):
        if et_events["DIN_diff"][ind] == 4:
            if ind > 0 and et_events["DIN_diff"][ind - 1] == 2:
                if ind < len(et_events) - 1 and et_events["DIN_diff"][ind + 1] == 2:
                    et_events.loc[ind + 1, "DIN_diff"] = 3

    et_stims = et_events.loc[et_events["DIN_diff"].isin([3])]
    print(f"Number of eye-tracking stimulus onset DIN events: {len(et_stims)}")
    et_iti = et_stims["index"].diff()
    return et_events, et_stims, et_iti


def _et_process_mmn(et_events):
    et_events = et_events.copy()
    et_events = et_events.reset_index()
    et_stims = et_events.loc[et_events["DIN_diff"].isin([8])]
    print(f"Number of eye-tracking stimulus onset DIN events: {len(et_stims)}")
    et_iti = et_stims["index"].diff()
    return et_events, et_stims, et_iti


def _et_process_rest(et_events):
    et_events = et_events.copy()
    et_events = et_events.reset_index()

    for ind in range(len(et_events)):
        if (ind % 2) != 0:
            et_events.loc[ind, "DIN_diff"] = 3

    et_stims = et_events.loc[et_events["DIN_diff"].isin([3])]
    print(f"Number of eye-tracking stimulus onset DIN events: {len(et_stims)}")
    et_iti = et_stims["index"].diff()
    return et_events, et_stims, et_iti


def show_sync_offsets(eeg_stims, et_stims):
    """Plot the time offset between matched EEG and ET events.

    Parameters
    ----------
    eeg_stims : np.ndarray
        EEG stimulus events.
    et_stims : pd.DataFrame
        ET stimulus events with ``"index"`` column.
    """
    eeg_et_offset = eeg_stims[:, 0] - et_stims["index"][:]
    fig = px.scatter(y=eeg_et_offset)
    fig.show()


def eeg_et_combine(eeg_raw, et_raw, eeg_stims, et_stims):
    """Combine EEG and eye-tracking data after alignment.

    Uses ``mne.preprocessing.realign_raw`` to temporally align ET to
    EEG, then combines all channels into a single Raw object.

    Parameters
    ----------
    eeg_raw : mne.io.Raw
        Raw EEG data.
    et_raw : mne.io.Raw
        Raw eye-tracking data.
    eeg_stims : np.ndarray
        EEG stimulus events.
    et_stims : pd.DataFrame
        ET stimulus events.

    Returns
    -------
    mne.io.RawArray
        Combined EEG + ET raw data.
    """
    eeg_times = eeg_stims[:, 0] / 1000
    et_times = et_stims["time"].reset_index(drop=True).to_numpy()

    mne.preprocessing.realign_raw(et_raw, eeg_raw, et_times, eeg_times,
                                  verbose=None)

    eeg_only = eeg_raw.copy().pick_types(eeg=True)
    eeg_stim_raw = eeg_raw.copy().pick_types(stim=True)

    eeg_et_array = np.vstack((
        eeg_only.get_data(),
        et_raw.copy().get_data(),
        eeg_stim_raw.get_data(),
    ))

    info = mne.create_info(
        ch_names=(eeg_only.info["ch_names"] +
                  et_raw.info["ch_names"] +
                  eeg_stim_raw.info["ch_names"]),
        sfreq=1000,
        ch_types=(eeg_only.get_channel_types() +
                  et_raw.get_channel_types() +
                  eeg_stim_raw.get_channel_types()),
    )

    return mne.io.RawArray(eeg_et_array, info)


# =============================================================================
# ET FUNCTIONS (ported from old q1k_init_tools.py)
# =============================================================================

def et_read(path, blink_interp, fill_nans, resamp):
    """Read eye-tracking .asc file and return raw + dataframe."""
    import re as _re
    et_raw = mne.io.read_raw_eyelink(path)
    et_raw.load_data()
    et_annot_events, et_annot_event_dict = mne.events_from_annotations(et_raw)

    if blink_interp:
        print("Interpolating blinks.")
        et_raw = mne.io.read_raw_eyelink(path, create_annotations=["blinks"])
        et_raw.load_data()
        mne.preprocessing.eyetracking.interpolate_blinks(
            et_raw, buffer=(0.05, 0.2), interpolate_gaze=True
        )

    if fill_nans:
        print("Filling NaNs with zeros.")
        data = et_raw.get_data()
        data[np.isnan(data)] = 0
        et_raw._data = data

    if resamp:
        print("Resampling the data.")
        et_raw.resample(1000, npad="auto")

    et_raw_df = et_raw.to_data_frame()
    return et_raw, et_raw_df, et_annot_events, et_annot_event_dict


def et_clean_events(et_annot_event_dict, et_annot_events):
    """Clean ET annotation events: remove TRACKER_TIME/SYNC FRAME, normalise indices."""
    import re as _re

    filtered_dict = {
        k: v for k, v in et_annot_event_dict.items()
        if "TRACKER_TIME" not in k and "SYNC FRAME" not in k
    }
    updated_dict = {k: i + 1 for i, (k, _) in enumerate(filtered_dict.items())}
    valid_values = set(filtered_dict.values())
    filtered_events = np.array([row for row in et_annot_events if row[2] in valid_values])
    value_map = {v: updated_dict[k] for k, v in filtered_dict.items()}
    updated_events = np.array(
        [[row[0], row[1], value_map[row[2]]] for row in filtered_events]
    )
    et_annot_event_dict = updated_dict
    et_annot_events = updated_events

    cleaned_dict = {}
    index_map = {}
    for key, value in et_annot_event_dict.items():
        clean_key = _re.sub(r'^[-\d\s]+', '', key)
        if clean_key not in cleaned_dict:
            cleaned_dict[clean_key] = value
        else:
            cleaned_dict[clean_key] = min(cleaned_dict[clean_key], value)
        index_map[value] = clean_key

    new_dict = {key: idx + 1 for idx, key in enumerate(sorted(cleaned_dict.keys()))}
    old_to_new = {
        old: new_dict[cleaned_key]
        for old, cleaned_key in index_map.items()
    }
    updated_col = [old_to_new.get(v, -1) for v in et_annot_events[:, 2]]
    if -1 in updated_col:
        unmatched = [v for v in et_annot_events[:, 2] if v not in old_to_new]
        raise ValueError(f"Unmatched values in et_annot_events[:, 2]: {unmatched}")
    et_annot_events[:, 2] = updated_col
    return new_dict, et_annot_events


def _times_align(a_times, b_times):
    """Align a_times to b_times by nearest match."""
    return np.array([a_times[np.argmin(np.abs(a_times - b))] for b in b_times])


def eeg_et_align(eeg_event_dict, et_event_dict, eeg_events, et_events,
                 eeg_stims, et_stims, eeg_sfreq, et_sfreq, task_id):
    """Align EEG and ET stimulus times and insert eeg_sync_time / et_sync_time events."""
    eeg_times = eeg_stims[:, 0] / eeg_sfreq
    et_times  = et_stims[:, 0]  / et_sfreq

    n_eeg, n_et = len(eeg_times), len(et_times)

    if n_eeg > n_et:
        if task_id == 'GO':
            ds_idx = {v for k, v in eeg_event_dict.items() if k.startswith('ds')}
            ds_ev  = eeg_events[np.isin(eeg_events[:, 2], list(ds_idx))]
            first_eeg = np.min(ds_ev[:, 0]) / eeg_sfreq if ds_ev.size else 0

            cs_idx = {v for k, v in et_event_dict.items() if k.startswith('CS_SPIN')}
            cs_ev  = et_events[np.isin(et_events[:, 2], list(cs_idx))]
            first_et = np.min(cs_ev[:, 0]) / et_sfreq if cs_ev.size else 0

            offset = first_eeg - first_et
            adj    = eeg_times - offset
            closest = np.array([np.argmin(np.abs(adj - b)) for b in et_times])
            eeg_times = eeg_times[closest]
        else:
            print("More EEG times than ET — attempting align.")
            eeg_times = _times_align(eeg_times, et_times)
    elif n_eeg < n_et:
        print("More ET times than EEG — attempting align.")
        et_times = _times_align(et_times, eeg_times)
    else:
        print("EEG and ET times match — continuing.")

    if len(eeg_times) != len(et_times):
        print("Alignment failed — abandoning sync.")
        return eeg_event_dict, et_event_dict, eeg_events, et_events, eeg_times, et_times

    eeg_event_dict['eeg_sync_time'] = len(eeg_event_dict) + 1
    et_event_dict['et_sync_time']   = len(et_event_dict)  + 1

    eeg_sync = [[int(s), 0, eeg_event_dict['eeg_sync_time']] for s in eeg_times * eeg_sfreq]
    et_sync  = [[int(s), 0, et_event_dict['et_sync_time']]   for s in et_times  * et_sfreq]

    eeg_events = np.vstack([eeg_events, eeg_sync])
    eeg_events = eeg_events[eeg_events[:, 0].argsort()]
    et_events  = np.vstack([et_events,  et_sync])
    et_events  = et_events[et_events[:, 0].argsort()]

    return eeg_event_dict, et_event_dict, eeg_events, et_events, eeg_times, et_times


def et_events_to_annot(et_raw, et_event_dict, et_events):
    """Convert ET events array to MNE annotations and attach to et_raw."""
    et_event_dict_r = {v: k for k, v in et_event_dict.items()}
    event_annotations = mne.annotations_from_events(
        events=et_events,
        event_desc=et_event_dict_r,
        sfreq=et_raw.info['sfreq'],
        orig_time=et_raw.info['meas_date'],
    )

    existing = et_raw.annotations
    keep = ('BAD_blink', 'BAD_ACQ_SKIP')
    selected = mne.Annotations(
        onset     =[existing.onset[i]    for i, d in enumerate(existing.description) if d in keep],
        duration  =[existing.duration[i] for i, d in enumerate(existing.description) if d in keep],
        description=[d for d in existing.description if d in keep],
        orig_time =existing.orig_time,
    )

    combined = event_annotations + selected
    ch_types = et_raw.get_channel_types()
    ch_names = et_raw.ch_names
    eye_ch   = tuple(n for n, t in zip(ch_names, ch_types) if t in ('eyegaze', 'pupil'))

    combined.ch_names = np.array([
        eye_ch if d in ('fixation', 'saccade', 'BAD_blink', 'BAD_ACQ_SKIP') else ()
        for d in combined.description
    ], dtype=object)

    et_raw.set_annotations(combined)
    return et_raw


def write_et(et_raw, eeg_bids_path):
    """Save ET raw data as _et.fif alongside the EEG BIDS file.

    Parameters
    ----------
    et_raw : mne.io.Raw
    eeg_bids_path : str or Path
        Path to the written EEG BIDS file (used to derive ET output path).

    Returns
    -------
    str : path to the saved .fif file
    """
    import os as _os
    et_out_path = str(eeg_bids_path)
    et_out_path = et_out_path.replace("/eeg/", "/et/")
    et_out_path = et_out_path.replace("_eeg.edf", "_et.fif")
    _os.makedirs(_os.path.dirname(et_out_path), exist_ok=True)
    et_raw.save(et_out_path, overwrite=True)
    print(f"ET .fif saved: {et_out_path}")
    return et_out_path
