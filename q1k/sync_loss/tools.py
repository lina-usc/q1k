"""EEG/ET synchronization and PyLossless cleaning tools.

Handles applying PyLossless artifact annotations, combining EEG and
eye-tracking data, and writing BIDS output.
"""

import mne
import pandas as pd


def apply_ll(bids_path, ll_state, eeg_ll_raw):
    """Apply PyLossless flagging to raw EEG data.

    Merges flagged channels to bads, reads ICLabel classifications,
    excludes artifact ICA components, applies ICA, interpolates bad
    channels, and re-references to average.

    Parameters
    ----------
    bids_path : mne_bids.BIDSPath
        Path to the pylossless derivative.
    ll_state : pylossless.LosslessPipeline
        Loaded pylossless pipeline state.
    eeg_ll_raw : mne.io.Raw
        Raw EEG data to clean.

    Returns
    -------
    mne.io.Raw
        Cleaned raw EEG data.
    """
    bids_path_str = str(bids_path)

    # Merge flagged channels to bads
    manual = []
    for flag_type in ll_state.flags["ch"]:
        manual.extend(ll_state.flags["ch"][flag_type])
    print(ll_state.flags["ch"])
    eeg_ll_raw.info["bads"].extend(manual)
    eeg_ll_raw.info["bads"] = list(set(eeg_ll_raw.info["bads"]))

    eeg_ll_raw.plot_sensors(show_names=True)

    # Read ICLabel info and exclude artifact components
    df = pd.read_csv(
        bids_path_str.replace("_eeg.edf", "_iclabels.tsv"), sep="\t"
    )
    ll_state.ica2.exclude = list(
        df[df["ic_type"].str.match("eog|muscle|ch_noise|ecg")].index
    )

    # Load data and apply ICA
    eeg_ll_raw.load_data()
    ll_state.ica2.apply(eeg_ll_raw)
    eeg_ll_raw = eeg_ll_raw.interpolate_bads()
    eeg_ll_raw = eeg_ll_raw.set_eeg_reference(ref_channels="average")

    return eeg_ll_raw


def eeg_et_combine(eeg_raw, et_raw, eeg_times, et_times,
                    eeg_events, eeg_event_dict,
                    et_events, et_event_dict):
    """Synchronize and combine EEG and eye-tracking data.

    Uses ``mne.preprocessing.realign_raw`` to align ET data to EEG
    timing, then merges channels and annotations.

    Parameters
    ----------
    eeg_raw : mne.io.Raw
        Raw EEG data.
    et_raw : mne.io.Raw
        Raw eye-tracking data.
    eeg_times : np.ndarray
        EEG sync event times in seconds.
    et_times : np.ndarray
        ET sync event times in seconds.
    eeg_events, eeg_event_dict : np.ndarray, dict
        EEG events and event dictionary.
    et_events, et_event_dict : np.ndarray, dict
        ET events and event dictionary.

    Returns
    -------
    eeg_raw : mne.io.Raw
        Combined raw data with EEG + ET channels.
    et_raw : mne.io.Raw
        Realigned ET raw data.
    """
    eeg_raw.load_data()
    et_raw.load_data()

    # Align the data
    mne.preprocessing.realign_raw(
        eeg_raw, et_raw, eeg_times, et_times, verbose="error"
    )

    # Add ET channels to EEG raw
    eeg_raw.add_channels([et_raw], force_update_info=True)

    # Combine annotations
    eeg_annot = mne.Annotations(
        onset=eeg_raw.annotations.onset - eeg_raw.first_samp / 1000,
        duration=eeg_raw.annotations.duration,
        description=eeg_raw.annotations.description,
        orig_time=None,
    )

    et_annot = mne.Annotations(
        onset=et_raw.annotations.onset,
        duration=et_raw.annotations.duration,
        description=et_raw.annotations.description,
        orig_time=None,
    )

    combined_annotations = eeg_annot + et_annot
    eeg_raw.set_annotations(combined_annotations)

    return eeg_raw, et_raw
