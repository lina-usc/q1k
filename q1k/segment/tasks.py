"""Task-specific epoch segmentation logic.

Each function takes a raw EEG object and returns epochs, event
dictionaries, and condition labels for the corresponding task.
"""

from collections import OrderedDict

import mne
import numpy as np

from q1k.config import FRONTAL_ROI

# ── Task parameters ──────────────────────────────────────────────────

TASK_PARAMS = {
    "RS": {"tmin": -0.2, "tmax": 0.8},
    "RSRio": {"tmin": -0.2, "tmax": 0.8},
    "VEP": {"tmin": -1.0, "tmax": 2.0},
    "AEP": {"tmin": -1.0, "tmax": 2.0},
    "GO": {"tmin": -1.0, "tmax": 1.0},
    "PLR": {"tmin": -2.0, "tmax": 4.0},
    "VS": {"tmin": -1.0, "tmax": 1.0},
    "TO": {"tmin": -1.0, "tmax": 1.0},
}

# Channels used for ERP overlay plots per task
TASK_ERP_CHANNELS = {
    "RS": FRONTAL_ROI,
    "RSRio": FRONTAL_ROI,
    "VEP": ["E70"],
    "AEP": ["E6"],
    "GO": ["E6"],
    "PLR": ["E70"],
    "VS": ["E6"],
    "TO": ["E62"],
}


# ── Resting state ────────────────────────────────────────────────────

def segment_resting_state(eeg_raw, eeg_events=None, eeg_event_dict=None):
    """Segment resting state data into 1-second epochs.

    Creates events from annotation boundaries (``dbrk``, ``vs01``-``vs06``)
    with 1-second spacing.

    Parameters
    ----------
    eeg_raw : mne.io.Raw
        Preprocessed raw EEG data.
    eeg_events : np.ndarray, optional
        MNE events array. If None, extracted from annotations.
    eeg_event_dict : dict, optional
        Event ID mapping. If None, extracted from annotations.

    Returns
    -------
    epochs : mne.Epochs
        Segmented epochs.
    event_id : dict
        Event ID mapping used for epochs.
    conditions : list[str]
        Condition labels for analysis.
    """
    if eeg_events is None or eeg_event_dict is None:
        eeg_events, eeg_event_dict = mne.events_from_annotations(eeg_raw)

    freq = eeg_raw.info["sfreq"]
    annot_sample = []
    annot_id = []

    # Select rest-relevant annotations
    rest_labels = {"dbrk", "vs01", "vs02", "vs03", "vs04", "vs05", "vs06"}
    annots = [
        a for a in eeg_raw.annotations
        if a["description"] in rest_labels
    ]

    # Add an end annotation 60s after the last one
    annots.append(OrderedDict((
        ("onset", annots[-1]["onset"] + 60.0),
        ("duration", 0),
        ("description", "end"),
        ("orig_time", None),
    )))

    for annot, next_annot in zip(annots[:-1], annots[1:]):
        if annot["description"] in ("dbrk", "end"):
            continue
        samples = np.arange(
            int(annot["onset"] * freq),
            int(next_annot["onset"] * freq),
            int(1 * freq),
        )
        annot_sample.append(samples)
        id_ = eeg_event_dict[annot["description"]]
        annot_id.extend(id_ * np.ones(len(samples)))

    annot_sample = np.concatenate(annot_sample)
    rs_events = np.array(
        [annot_sample, [0] * len(annot_sample), annot_id], dtype=int
    ).T
    rs_events = mne.merge_events(rs_events, np.unique(rs_events[:, -1]), 1)

    event_id = {"rest": 1}
    params = TASK_PARAMS["RS"]
    epochs = mne.Epochs(
        eeg_raw, rs_events,
        tmin=params["tmin"], tmax=params["tmax"],
        on_missing="warn", event_id=event_id,
    )

    return epochs, event_id, ["rest"]


def _detect_rsrio(eeg_event_dict):
    """Check whether an RS recording is actually RSRio.

    RSRio files contain ``eyeo`` (eye-open) and/or ``comm`` (comment)
    annotations instead of the standard ``vs01``--``vs06`` markers.
    """
    return "eyeo" in eeg_event_dict or "comm" in eeg_event_dict


def segment_rsrio(eeg_raw, eeg_events=None, eeg_event_dict=None):
    """Segment RS Rio (resting state with movie) into 1-second epochs.

    Uses ``eyeo`` (eye-open) as the start marker and ``comm`` (comment)
    as the end marker.  Falls back to ``DIN4``/``DIN5`` for the end
    marker, or to the end of the recording if neither is found.

    Parameters
    ----------
    eeg_raw : mne.io.Raw
        Preprocessed raw EEG data.
    eeg_events : np.ndarray, optional
        MNE events array.  Extracted from annotations if *None*.
    eeg_event_dict : dict, optional
        Event ID mapping.  Extracted from annotations if *None*.

    Returns
    -------
    epochs : mne.Epochs
    event_id : dict
    conditions : list[str]
    """
    if eeg_events is None or eeg_event_dict is None:
        eeg_events, eeg_event_dict = mne.events_from_annotations(
            eeg_raw
        )

    freq = eeg_raw.info["sfreq"]

    # Gather candidate annotations
    marker_labels = {"eyeo", "comm", "DIN4", "DIN5"}
    annots = [
        a for a in eeg_raw.annotations
        if a["description"] in marker_labels
    ]
    if not annots:
        raise ValueError(
            "No annotations found for RSRio "
            "(expected eyeo, comm, DIN4, or DIN5)"
        )

    # --- Find start marker: prefer 'eyeo', fallback to first ---
    start_annot = None
    for a in annots:
        if a["description"] == "eyeo":
            start_annot = a
            break
    if start_annot is None:
        start_annot = annots[0]
        print(
            f"Warning: 'eyeo' not found, using first annotation "
            f"'{start_annot['description']}' as start marker"
        )

    # --- Find end marker: prefer 'comm', fallback to DIN4/5 ---
    end_annot = None
    for a in annots:
        if a["description"] == "comm":
            end_annot = a
            break

    if end_annot is None:
        candidates = [
            a for a in annots
            if a["description"] in ("DIN4", "DIN5")
            and a["onset"] > start_annot["onset"]
        ]
        if candidates:
            end_annot = candidates[-1]
            print(
                f"Warning: 'comm' not found, using last "
                f"'{end_annot['description']}' as end marker"
            )
        else:
            end_time = (eeg_raw.n_times - 1) / freq
            print(
                f"Warning: No end marker found. Using end of "
                f"recording ({end_time:.2f}s) as end marker"
            )
            end_annot = OrderedDict((
                ("onset", end_time),
                ("duration", 0),
                ("description", "recording_end"),
                ("orig_time", None),
            ))

    if end_annot["onset"] <= start_annot["onset"]:
        print(
            f"Warning: end marker ({end_annot['description']}) "
            f"onset {end_annot['onset']:.2f} is not after start "
            f"marker ({start_annot['description']}) onset "
            f"{start_annot['onset']:.2f}"
        )

    # Create 1-second event samples from start to end
    start_sample = int(start_annot["onset"] * freq)
    end_sample = int(end_annot["onset"] * freq)
    annot_sample = np.arange(start_sample, end_sample, int(freq))

    if len(annot_sample) == 0:
        print(
            f"Warning: No 1-second epochs fit between start "
            f"({start_annot['onset']:.2f}s) and end "
            f"({end_annot['onset']:.2f}s)"
        )
        rs_events = np.array(
            [[start_sample, 0, 1]], dtype=int
        )
    else:
        if start_annot["description"] in eeg_event_dict:
            id_ = eeg_event_dict[start_annot["description"]]
        else:
            id_ = 1
        annot_id = [id_] * len(annot_sample)
        rs_events = np.array(
            [annot_sample, [0] * len(annot_sample), annot_id],
            dtype=int,
        ).T
        rs_events = mne.merge_events(
            rs_events, np.unique(rs_events[:, -1]), 1
        )

    event_id = {"rest": 1}
    params = TASK_PARAMS["RSRio"]
    epochs = mne.Epochs(
        eeg_raw, rs_events,
        tmin=params["tmin"], tmax=params["tmax"],
        on_missing="warn", event_id=event_id,
    )

    return epochs, event_id, ["rest"]


# ── VEP ──────────────────────────────────────────────────────────────

def segment_vep(eeg_raw, eeg_events=None, eeg_event_dict=None):
    """Segment VEP (visual evoked potential) data.

    Conditions: ``sv06_d`` (6Hz) and ``sv15_d`` (15Hz).
    """
    if eeg_events is None or eeg_event_dict is None:
        eeg_events, eeg_event_dict = mne.events_from_annotations(eeg_raw)

    # Build epoch event dict from annotations containing sv06/sv15
    epoch_event_dict = {
        k: v for k, v in eeg_event_dict.items()
        if "sv06_d" in k or "sv15_d" in k
    }

    params = TASK_PARAMS["VEP"]
    epochs = mne.Epochs(
        eeg_raw, eeg_events,
        tmin=params["tmin"], tmax=params["tmax"],
        on_missing="warn", event_id=epoch_event_dict,
    )

    conditions = sorted(epoch_event_dict.keys())
    return epochs, epoch_event_dict, conditions


# ── AEP ──────────────────────────────────────────────────────────────

def segment_aep(eeg_raw, eeg_events=None, eeg_event_dict=None):
    """Segment AEP (auditory evoked potential) data.

    Conditions: ``ae06_d`` (6Hz) and ``ae40_d`` (40Hz).
    """
    if eeg_events is None or eeg_event_dict is None:
        eeg_events, eeg_event_dict = mne.events_from_annotations(eeg_raw)

    epoch_event_dict = {
        k: v for k, v in eeg_event_dict.items()
        if "ae06_d" in k or "ae40_d" in k
    }

    params = TASK_PARAMS["AEP"]
    epochs = mne.Epochs(
        eeg_raw, eeg_events,
        tmin=params["tmin"], tmax=params["tmax"],
        on_missing="warn", event_id=epoch_event_dict,
    )

    conditions = sorted(epoch_event_dict.keys())
    return epochs, epoch_event_dict, conditions


# ── GO ───────────────────────────────────────────────────────────────

def segment_go(eeg_raw, eeg_events=None, eeg_event_dict=None):
    """Segment GO (gap-overlap) data.

    Conditions: ``dtbc_d`` (baseline), ``dtoc_d`` (overlap),
    ``dtgc_d`` (gap).
    """
    if eeg_events is None or eeg_event_dict is None:
        eeg_events, eeg_event_dict = mne.events_from_annotations(eeg_raw)

    # Filter for gap-overlap events (starting with d or g)
    epoch_event_dict = {
        k: v for k, v in eeg_event_dict.items()
        if k.startswith(("d", "g"))
    }

    params = TASK_PARAMS["GO"]
    epochs = mne.Epochs(
        eeg_raw, eeg_events,
        tmin=params["tmin"], tmax=params["tmax"],
        on_missing="warn", event_id=epoch_event_dict,
    )

    conditions = [k for k in epoch_event_dict if k.endswith("_d")]
    return epochs, epoch_event_dict, conditions


# ── PLR ──────────────────────────────────────────────────────────────

def segment_plr(eeg_raw, eeg_events=None, eeg_event_dict=None):
    """Segment PLR (pupillary light reflex) data.

    Single condition: ``plro_d``.
    """
    if eeg_events is None or eeg_event_dict is None:
        eeg_events, eeg_event_dict = mne.events_from_annotations(eeg_raw)

    epoch_event_dict = {
        k: v for k, v in eeg_event_dict.items()
        if k == "plro_d"
    }

    params = TASK_PARAMS["PLR"]
    epochs = mne.Epochs(
        eeg_raw, eeg_events,
        tmin=params["tmin"], tmax=params["tmax"],
        on_missing="warn", event_id=epoch_event_dict,
    )

    return epochs, epoch_event_dict, ["plro_d"]


# ── VS ───────────────────────────────────────────────────────────────

def segment_vs(eeg_raw, eeg_events=None, eeg_event_dict=None):
    """Segment VS (visual search) data.

    Conditions grouped by array size: 5-item, 9-item, 13-item.
    Includes derived eye-tracking metrics (velocity, distance).
    """
    if eeg_events is None or eeg_event_dict is None:
        eeg_events, eeg_event_dict = mne.events_from_annotations(eeg_raw)

    # Filter for visual search events
    epoch_event_dict = {
        k: v for k, v in eeg_event_dict.items()
        if k.startswith(("d", "g", "v"))
    }

    params = TASK_PARAMS["VS"]
    epochs = mne.Epochs(
        eeg_raw, eeg_events,
        tmin=params["tmin"], tmax=params["tmax"],
        on_missing="warn", event_id=epoch_event_dict,
    )

    conditions = sorted(set(k for k in epoch_event_dict if "ds" in k))
    return epochs, epoch_event_dict, conditions


# ── TO ───────────────────────────────────────────────────────────────

def segment_to(eeg_raw, eeg_events=None, eeg_event_dict=None):
    """Segment TO (tone oddball) data.

    Conditions: standard items, deviant items.
    """
    if eeg_events is None or eeg_event_dict is None:
        eeg_events, eeg_event_dict = mne.events_from_annotations(eeg_raw)

    # Filter for tone oddball events
    epoch_event_dict = {
        k: v for k, v in eeg_event_dict.items()
        if k.startswith("to/") or k.startswith("SO") or k.startswith("Dev")
    }

    params = TASK_PARAMS["TO"]
    epochs = mne.Epochs(
        eeg_raw, eeg_events,
        tmin=params["tmin"], tmax=params["tmax"],
        on_missing="warn", event_id=epoch_event_dict,
    )

    conditions = sorted(epoch_event_dict.keys())
    return epochs, epoch_event_dict, conditions


# ── Dispatcher ───────────────────────────────────────────────────────

SEGMENT_FUNCTIONS = {
    "RS": segment_resting_state,
    "RSRio": segment_rsrio,
    "VEP": segment_vep,
    "AEP": segment_aep,
    "GO": segment_go,
    "PLR": segment_plr,
    "VS": segment_vs,
    "TO": segment_to,
}


def segment_task(task, eeg_raw, eeg_events=None, eeg_event_dict=None):
    """Dispatch to the appropriate segmentation function.

    Parameters
    ----------
    task : str
        Task code (e.g., ``"RS"``, ``"VEP"``).
    eeg_raw : mne.io.Raw
        Preprocessed raw EEG data.
    eeg_events : np.ndarray, optional
        MNE events array.
    eeg_event_dict : dict, optional
        Event ID mapping.

    Returns
    -------
    epochs : mne.Epochs
    event_id : dict
    conditions : list[str]

    Raises
    ------
    ValueError
        If the task is not supported.
    """
    func = SEGMENT_FUNCTIONS.get(task)
    if func is None:
        raise ValueError(
            f"Unknown task: {task}. "
            f"Supported tasks: {list(SEGMENT_FUNCTIONS.keys())}"
        )
    return func(eeg_raw, eeg_events, eeg_event_dict)
