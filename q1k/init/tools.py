"""EEG/ET initialization tools for BIDS conversion.

Handles event extraction, task-specific event processing, eye-tracking
synchronization, and EEG/ET data combination.
"""
import glob
import os

import mne
import mne_bids
import numpy as np
import plotly.express as px

#VALID_TASKS = ["rest", "as", "ssvep", "vs", "ap",
#               "go", "plr", "mn", "nsp", "fsp", "RSRio"]

VALID_TASKS = ['rest', 'RS', 'as', 'AS', 'ssvep', 'vp', 'VEP', 'vs', 'VS', 'ap', 'AEP',
               'go', 'GO', 'plr', 'mn', 'TO', 'nsp', 'fsp', 'PLR']


def generate_session_ids(dataset_group, project_path, site_code, task_id_in, subject_id_in, run_id):
    if dataset_group == "control":
        session_path_eeg = project_path + '/sourcefiles/' + subject_id_in + '/' + subject_id_in + '_eeg/'
        session_file_name_eeg = glob.glob(session_path_eeg + '*_' + task_id_in + '_*.mff')

    elif dataset_group == "experimental":
        session_path_eeg = project_path + '/source_prime/'  + site_code + '/'+  subject_id_in + '/eeg/' + task_id_in.upper() + '/'
        session_file_name_eeg = glob.glob(session_path_eeg + subject_id_in + '_*.mff')
        print(session_path_eeg)
        print(session_file_name_eeg)

        session_path_et = project_path + '/source_prime/' + site_code + '/' + subject_id_in + '/et/' + task_id_in.upper() + '/'
        session_file_name_et = glob.glob(session_path_et + subject_id_in + '_' + task_id_in.upper() + '.asc')
        print(session_path_et)
        print(session_file_name_et)

    return session_file_name_eeg, session_file_name_et


def set_family_code(session_code_in):
    if session_code_in.startswith(("1025-", "1525-")):
        family_code_out = session_code_in[5:9]
    elif session_code_in.startswith(("100", "200")):
        family_code_out = session_code_in[3:].zfill(4)
    else:
        family_code_out = session_code_in
    return family_code_out


def set_din_str(task_id_out):
    task_id_out = task_id_out.upper()
    if task_id_out == 'AEP':
        event_dict_offset = 1
        din_str = ('DIN4','DIN5')
    if task_id_out == 'AS':
        event_dict_offset = 1
        din_str = ('DIN2','DIN2')
    if task_id_out == 'GO':
        event_dict_offset = 1
        din_str = ('DIN4','DIN5')
    if task_id_out == 'TO':
        event_dict_offset = 1
        din_str = ('DIN4','DIN5')
    if task_id_out == 'VEP':
        event_dict_offset = 1
        din_str = ('DIN2','DIN3')
    if task_id_out == 'PLR':
        event_dict_offset = 1
        din_str = ('DIN2','DIN3')
    if task_id_out == 'RS':
        event_dict_offset = 1
        din_str = ('DIN2','DIN3')
    if task_id_out == 'VS':
        event_dict_offset = 1
        din_str = ('DIN2','DIN3')
    if task_id_out == 'NSP':
        event_dict_offset = 1
        din_str = ('DIN2','DIN3')
    return din_str, event_dict_offset







def get_din_diff(events, event_dict, din_str):
    din_diffs = []
    din_diffs_time = []
    last_din = 0
    for row in events:
        if row[2] == event_dict[din_str[0]] or row[2] == event_dict[din_str[0]]:
            if last_din > 0:
                din_diffs.append(row[0] - last_din)
                din_diffs_time.append(row[0])
                last_din = row[0]
            else:
                last_din = row[0]
    return din_diffs, din_diffs_time



def din_check(event_dict, din_str):
    exists_in_dict = [din in event_dict for din in din_str]
    print(din_str)
    print(exists_in_dict)
    if all(exists_in_dict):
        print("Both strings exist in eeg_event_dict.")
    elif any(exists_in_dict):
        existing_string = din_str[exists_in_dict.index(True)]
        din_str = (existing_string, existing_string)
        print(f"Only one string exists. din_str updated to: {din_str}")
    else:
        din_str=()
        print("Neither DIN string exists in eeg_event_dict.")
    print(din_str)
    return din_str






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
    import numpy as np

    din_offset = []
    din_str = next(
        (d for d in [("DIN2", "DIN3"), ("DIN4", "DIN5"), ("DIN4", "DIN3")]
         if d[0] in eeg_event_dict),
        din_str,
    )

    print("Removing TSYN events...")
    if "TSYN" in eeg_event_dict:
        eeg_events = eeg_events[~np.isin(eeg_events[:, 2], [eeg_event_dict["TSYN"]])]

    new_events = np.empty((0, 3), dtype=int)
    base_id = len(eeg_event_dict) + 1

    for i, e in np.ndenumerate(eeg_events[:, 2]):
        for label_idx, trigger in enumerate(["dtoc", "dtbc", "dtgc"]):
            if e == eeg_event_dict[trigger] and i[0] + 1 < len(eeg_events[:, 2]):
                next_event = eeg_events[i[0] + 1, 2]
                if next_event == eeg_event_dict[din_str[0]] or next_event == eeg_event_dict[din_str[1]]:
                    new_row = np.array([[eeg_events[i[0] + 1, 0], 0, base_id + label_idx]])
                    new_events = np.append(new_events, new_row, axis=0)
                    din_offset.append(eeg_events[i[0] + 1, 0] - eeg_events[i[0], 0])

    eeg_events = np.concatenate((eeg_events, new_events))
    eeg_events = eeg_events[eeg_events[:, 0].argsort()]

    eeg_event_dict["dtoc_d"] = len(eeg_event_dict) + 1
    eeg_event_dict["dtbc_d"] = len(eeg_event_dict) + 1
    eeg_event_dict["dtgc_d"] = len(eeg_event_dict) + 1

    eeg_stims = eeg_events[np.isin(eeg_events[:, 2], [
        eeg_event_dict["dtoc_d"],
        eeg_event_dict["dtbc_d"],
        eeg_event_dict["dtgc_d"],
    ])]

    print("Number of stimulus onset DIN events: " + str(len(eeg_stims)))
    eeg_iti = np.diff(eeg_stims[:, 0])

    return eeg_events, eeg_stims, eeg_iti, din_offset, eeg_event_dict, new_events



def eeg_clean_events(eeg_events, eeg_event_dict, din_str):
    print('Removing TSYN events...')
    mask = np.isin(eeg_events[:,2],[eeg_event_dict['TSYN']])
    eeg_events = eeg_events[~mask]
    if 'TSYN' in eeg_event_dict:
        del eeg_event_dict['TSYN']

    filtered_dict = {k: v for k, v in eeg_event_dict.items() if not k.startswith('DIN') or k in din_str}
    filtered_dict = {key: i + 1 for i, (key, _) in enumerate(filtered_dict.items())}
    updated_events = np.array([
        [row[0], row[1], filtered_dict[key]]
        for row in eeg_events if (key := next((k for k, v in eeg_event_dict.items() if v == row[2]), None)) in filtered_dict
    ])

    eeg_event_dict = filtered_dict
    eeg_events = updated_events
    print("Updated Dictionary:", eeg_event_dict)
    return eeg_events, eeg_event_dict













'''
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
                    if any(next_event == eeg_event_dict.get(d)
                        for d in din_labels if d in eeg_event_dict):
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
'''
def eeg_task_events(eeg_events, eeg_event_dict, din_str, task_name=None):

    din_offset = []
    new_events = np.empty((0, 3))

    if not task_name:
        raise ValueError(f'please pass one of {VALID_TASKS}'
                         ' to the task_name keyword argument.')

    if task_name == 'PLR':
        #find the first din_str event following plro events and add new *d events
        for i, e in np.ndenumerate(eeg_events[:,2]):
            if e == eeg_event_dict['plro']:
                if i[0]+1 < len(eeg_events[:,2]):
                    if eeg_events[i[0]+1, 2] == eeg_event_dict[din_str[0]] or eeg_events[i[0]+1, 2] == eeg_event_dict[din_str[1]]:
                        new_row = np.array([[eeg_events[i[0] + 1, 0], 0, len(eeg_event_dict) + 1]])
                        new_events = np.append(new_events,new_row, axis=0)
                        din_offset.append(eeg_events[i[0]+1, 0] - eeg_events[i[0], 0])

        #append new events to eeg_events
        eeg_events = np.concatenate((eeg_events,new_events))
        eeg_events = eeg_events[eeg_events[:,0].argsort()]
        # add the new stimulus onset DIN labels to the event_dict..
        eeg_event_dict['plro_d'] = len(eeg_event_dict) + 1

        #select all of the newly categorized stimulus DIN events
        mask = np.isin(eeg_events[:,2],[eeg_event_dict['plro_d']])
        eeg_stims = eeg_events[mask]
        print('Number of stimulus onset DIN events: ' + str(len(eeg_stims))) #the length of this array should equal the number of stimulus trials in the task

        #calculate the inter trial interval between stimulus onset DIN events
        eeg_iti = np.diff(eeg_stims[:,0])



    elif task_name == 'VEP':

        # find the first DIN3 event following either sv06 or sv15 events and add new *d events
        # DIN3 is used here even though it is the second DIN in the visual stimulus inversion animations...
        # DIN3 is used because because it is more reliable that DIN2 at the HSJ site and its offset from the initial DIN2 event is exactly determined by stimulus condition
        # This is handled differently it the et_task_event function.. but because the stimulus sequence is exactly deterministic these always line up in practice
        # I would continue to keep an eye on this...
        for i, e in np.ndenumerate(eeg_events[:,2]):
            if e == eeg_event_dict['sv06']:
                if i[0]+1 < len(eeg_events[:,2]):
                    if eeg_events[i[0]+1, 2] == eeg_event_dict[din_str[0]] or eeg_events[i[0]+1, 2] == eeg_event_dict[din_str[1]]:
                        if eeg_events[i[0]+1, 2] == eeg_event_dict[din_str[1]] and din_str[1] == 'DIN3':
                            cor_val = 166
                        else:
                            cor_val = 0
                        new_row = np.array([[eeg_events[i[0] + 1, 0] - cor_val, 0, len(eeg_event_dict) + 1]])
                        new_events = np.append(new_events,new_row, axis=0)
                        din_offset.append(eeg_events[i[0]+1, 0] - eeg_events[i[0], 0])
            if e == eeg_event_dict['sv15']:
                if i[0]+1 < len(eeg_events[:,2]):
                    if eeg_events[i[0]+1, 2] == eeg_event_dict[din_str[0]] or eeg_events[i[0]+1, 2] == eeg_event_dict[din_str[1]]:
                        if eeg_events[i[0]+1, 2] == eeg_event_dict[din_str[1]] and din_str[1] == 'DIN3':
                            cor_val = 66
                        else:
                            cor_val = 0
                        new_row = np.array([[eeg_events[i[0] + 1, 0] - cor_val, 0, len(eeg_event_dict) + 2]])
                        new_events = np.append(new_events,new_row, axis=0)
                        din_offset.append(eeg_events[i[0]+1, 0] - eeg_events[i[0], 0])

        # append new events to eeg_events
        eeg_events = np.concatenate((eeg_events,new_events))
        eeg_events = eeg_events[eeg_events[:,0].argsort()]
        # add the new stimulus onset DIN labels to the event_dict..
        eeg_event_dict['sv06_d'] = len(eeg_event_dict) + 1
        eeg_event_dict['sv15_d'] = len(eeg_event_dict) + 1

        #select all of the newly categorized stimulus DIN events
        mask = np.isin(eeg_events[:,2],[eeg_event_dict['sv06_d'],eeg_event_dict['sv15_d']])
        eeg_stims = eeg_events[mask]
        print('Number of stimulus onset DIN events: ' + str(len(eeg_stims))) #the length of this array should equal the number of stimulus trials in the task

        #calculate the inter trial interval between stimulus onset DIN events
        eeg_iti = np.diff(eeg_stims[:,0])




    elif task_name == 'ap' or task_name == 'AEP':

        # find the first DIN4 event following either mmns or mmnt events and add new *d events
        for i, e in np.ndenumerate(eeg_events[:,2]):
            if e == eeg_event_dict['ae06']:
                if i[0]+1 < len(eeg_events[:,2]):
                    if eeg_events[i[0]+1, 2] == eeg_event_dict['DIN4']:
                        new_row = np.array([[eeg_events[i[0] + 1, 0], 0, len(eeg_event_dict) + 1]])
                        new_events = np.append(new_events,new_row, axis=0)
                        din_offset.append(eeg_events[i[0]+1, 0] - eeg_events[i[0], 0])
            if e == eeg_event_dict['ae40']:
                if i[0]+1 < len(eeg_events[:,2]):
                    if eeg_events[i[0]+1, 2] == eeg_event_dict['DIN4']:
                        new_row = np.array([[eeg_events[i[0] + 1, 0], 0, len(eeg_event_dict) + 2]])
                        new_events = np.append(new_events,new_row, axis=0)
                        din_offset.append(eeg_events[i[0]+1, 0] - eeg_events[i[0], 0])

        # append new events to eeg_events
        eeg_events = np.concatenate((eeg_events,new_events))
        eeg_events = eeg_events[eeg_events[:,0].argsort()]
        # add the new stimulus onset DIN labels to the event_dict..
        eeg_event_dict['ae06_d'] = len(eeg_event_dict) + 1
        eeg_event_dict['ae40_d'] = len(eeg_event_dict) + 1

        #select all of the newly categorized stimulus DIN events
        mask = np.isin(eeg_events[:,2],[eeg_event_dict['ae06_d'],eeg_event_dict['ae40_d']])
        eeg_stims = eeg_events[mask]
        print('Number of stimulus onset DIN events: ' + str(len(eeg_stims))) #the length of this array should equal the number of stimulus trials in the task

        #calculate the inter trial interval between stimulus onset DIN events
        eeg_iti = np.diff(eeg_stims[:,0])


    #elif task_name == 'go':

    elif task_name=='go'or task_name == 'GO':

        # find the first DIN4 event following either mmns or mmnt events and add new *d events
        for i, e in np.ndenumerate(eeg_events[:,2]):

            if 'DIN2' in din_str:
                eeg_event_list = ['dfoc','dfbc','dfgc']
                eeg_d_event_list = ['dfoc_d','dfbc_d','dfgc_d']
            else:
                eeg_event_list = ['dsoc','dsbc','dsgc']
                eeg_d_event_list = ['dsoc_d','dsbc_d','dsgc_d']

            if e == eeg_event_dict[eeg_event_list[0]]:
                if i[0]+1 < len(eeg_events[:,2]):
                    if eeg_events[i[0]+1, 2] == eeg_event_dict[din_str[0]] or eeg_events[i[0]+1, 2] == eeg_event_dict[din_str[0]]:
                        new_row = np.array([[eeg_events[i[0] + 1, 0], 0, len(eeg_event_dict) + 1]])
                        new_events = np.append(new_events,new_row, axis=0)
                        din_offset.append(eeg_events[i[0]+1, 0] - eeg_events[i[0], 0])
            if e == eeg_event_dict[eeg_event_list[1]]:
                if i[0]+1 < len(eeg_events[:,2]):
                    if eeg_events[i[0]+1, 2] == eeg_event_dict[din_str[0]] or eeg_events[i[0]+1, 2] == eeg_event_dict[din_str[0]]:
                        new_row = np.array([[eeg_events[i[0] + 1, 0], 0, len(eeg_event_dict) + 2]])
                        new_events = np.append(new_events,new_row, axis=0)
                        din_offset.append(eeg_events[i[0]+1, 0] - eeg_events[i[0], 0])
            if e == eeg_event_dict[eeg_event_list[2]]:
                if i[0]+1 < len(eeg_events[:,2]):
                    if eeg_events[i[0]+1, 2] == eeg_event_dict[din_str[0]] or eeg_events[i[0]+1, 2] == eeg_event_dict[din_str[0]]:
                        new_row = np.array([[eeg_events[i[0] + 1, 0], 0, len(eeg_event_dict) + 3]])
                        new_events = np.append(new_events,new_row, axis=0)
                        din_offset.append(eeg_events[i[0]+1, 0] - eeg_events[i[0], 0])

        # append new events to eeg_events
        eeg_events = np.concatenate((eeg_events,new_events))
        eeg_events = eeg_events[eeg_events[:,0].argsort()]
        # add the new stimulus onset DIN labels to the event_dict..
        eeg_event_dict[eeg_d_event_list[0]] = len(eeg_event_dict) + 1
        eeg_event_dict[eeg_d_event_list[1]] = len(eeg_event_dict) + 1
        eeg_event_dict[eeg_d_event_list[2]] = len(eeg_event_dict) + 1

        #select all of the newly categorized stimulus DIN events
        mask = np.isin(eeg_events[:,2],[eeg_event_dict[eeg_d_event_list[0]],eeg_event_dict[eeg_d_event_list[1]],eeg_event_dict[eeg_d_event_list[2]]])
        eeg_stims = eeg_events[mask]
        print('Number of stimulus onset DIN events: ' + str(len(eeg_stims))) #the length of this array should equal the number of stimulus trials in the task

        #calculate the inter trial interval between stimulus onset DIN events
        eeg_iti = np.diff(eeg_stims[:,0])


    elif task_name=='vs'or task_name == 'VS':

        # find the first DIN4 event following either mmns or mmnt events and add new *d events
        for i, e in np.ndenumerate(eeg_events[:,2]):

            #eeg_event_list = ['df', 'ds']

            #if e in {value for key, value in eeg_event_dict.items() if key.startswith(('df','ds'))}:
            if e in {value for key, value in eeg_event_dict.items() if key.startswith(('da'))}:
                if i[0]+1 < len(eeg_events[:,2]):
                    #if eeg_events[i[0]+1, 2] == eeg_event_dict[din_str[0]] or eeg_events[i[0]+1, 2] == eeg_event_dict[din_str[1]]:
                    if eeg_events[i[0]+1, 2] == eeg_event_dict['DIN3']: #or eeg_events[i[0]+1, 2] == eeg_event_dict[din_str[1]]:
                        new_row = np.array([[eeg_events[i[0] + 1, 0], 0, len(eeg_event_dict) + 1]])
                        new_events = np.append(new_events,new_row, axis=0)
                        din_offset.append(eeg_events[i[0]+1, 0] - eeg_events[i[0], 0])

        # append new events to eeg_events
        eeg_events = np.concatenate((eeg_events,new_events))
        eeg_events = eeg_events[eeg_events[:,0].argsort()]
        # add the new stimulus onset DIN labels to the event_dict..
        eeg_event_dict['da_d'] = len(eeg_event_dict) + 1

        #select all of the newly categorized stimulus DIN events
        mask = np.isin(eeg_events[:,2],[eeg_event_dict['da_d']])
        eeg_stims = eeg_events[mask]
        print('Number of stimulus onset DIN events: ' + str(len(eeg_stims))) #the length of this array should equal the number of stimulus trials in the task

        #calculate the inter trial interval between stimulus onset DIN events
        eeg_iti = np.diff(eeg_stims[:,0])


    elif task_name=='nsp'or task_name == 'NSP':

        # find the first DIN4 event following either mmns or mmnt events and add new *d events
        for i, e in np.ndenumerate(eeg_events[:,2]):
            if e in {value for key, value in eeg_event_dict.items() if key.startswith(('dfns'))}:
                if i[0]+1 < len(eeg_events[:,2]):
                    if eeg_events[i[0]+1, 2] == eeg_event_dict['DIN3']: #or eeg_events[i[0]+1, 2] == eeg_event_dict[din_str[1]]:
                        new_row = np.array([[eeg_events[i[0] + 1, 0], 0, len(eeg_event_dict) + 1]])
                        new_events = np.append(new_events,new_row, axis=0)
                        din_offset.append(eeg_events[i[0]+1, 0] - eeg_events[i[0], 0])

        # append new events to eeg_events
        eeg_events = np.concatenate((eeg_events,new_events))
        eeg_events = eeg_events[eeg_events[:,0].argsort()]
        # add the new stimulus onset DIN labels to the event_dict..
        eeg_event_dict['dfns_d'] = len(eeg_event_dict) + 1

        #select all of the newly categorized stimulus DIN events
        mask = np.isin(eeg_events[:,2],[eeg_event_dict['dfns_d']])
        eeg_stims = eeg_events[mask]
        print('Number of stimulus onset DIN events: ' + str(len(eeg_stims))) #the length of this array should equal the number of stimulus trials in the task

        #calculate the inter trial interval between stimulus onset DIN events
        eeg_iti = np.diff(eeg_stims[:,0])


    elif task_name=='mn' or task_name=='TO':

        s_ind = [value for key, value in eeg_event_dict.items() if key.startswith('SO')]
        t_ind = [value for key, value in eeg_event_dict.items() if key.startswith('Dev')]

        # find the first DIN4 event following either mmns or mmnt events and add new *d events
        for i, e in np.ndenumerate(eeg_events[:,2]):
            if e in s_ind:
                if i[0]+1 < len(eeg_events[:,2]):
                    if eeg_events[i[0]+1, 2] == eeg_event_dict['DIN4']:
                        new_row = np.array([[eeg_events[i[0] + 1, 0], 0, len(eeg_event_dict) + 1]])
                        new_events = np.append(new_events,new_row, axis=0)
                        din_offset.append(eeg_events[i[0]+1, 0] - eeg_events[i[0], 0])
            if e in t_ind:
                if i[0]+1 < len(eeg_events[:,2]):
                    if eeg_events[i[0]+1, 2] == eeg_event_dict['DIN4']:
                        new_row = np.array([[eeg_events[i[0] + 1, 0], 0, len(eeg_event_dict) + 2]])
                        new_events = np.append(new_events,new_row, axis=0)
                        din_offset.append(eeg_events[i[0]+1, 0] - eeg_events[i[0], 0])

        # append new events to eeg_events
        eeg_events = np.concatenate((eeg_events,new_events))
        eeg_events = eeg_events[eeg_events[:,0].argsort()]
        # add the new stimulus onset DIN labels to the event_dict..
        eeg_event_dict['to_s_d'] = len(eeg_event_dict) + 1
        eeg_event_dict['to_t_d'] = len(eeg_event_dict) + 1

        #select all of the newly categorized stimulus DIN events
        mask = np.isin(eeg_events[:,2],[eeg_event_dict['to_s_d'],eeg_event_dict['to_t_d']])
        eeg_stims = eeg_events[mask]
        print('Number of stimulus onset DIN events: ' + str(len(eeg_stims))) #the length of this array should equal the number of stimulus trials in the task

        #calculate the inter trial interval between stimulus onset DIN events
        eeg_iti = np.diff(eeg_stims[:,0])



    elif task_name=='rest' or task_name=='RS':

        v_ind = [value for key, value in eeg_event_dict.items() if key.startswith('vs')]
        b_ind = [value for key, value in eeg_event_dict.items() if key.startswith('dbrk')]

        # find the first DIN4 event following either mmns or mmnt events and add new *d events
        for i, e in np.ndenumerate(eeg_events[:,2]):
            if e in v_ind:
                if i[0]+1 < len(eeg_events[:,2]):
                    if eeg_events[i[0]+1, 2] == eeg_event_dict['DIN2']:
                        new_row = np.array([[eeg_events[i[0] + 1, 0], 0, len(eeg_event_dict) + 1]])
                        new_events = np.append(new_events,new_row, axis=0)
                        din_offset.append(eeg_events[i[0]+1, 0] - eeg_events[i[0], 0])
            if e in b_ind:
                if i[0]+1 < len(eeg_events[:,2]):
                    if eeg_events[i[0]+1, 2] == eeg_event_dict['DIN2']:
                        new_row = np.array([[eeg_events[i[0] + 1, 0], 0, len(eeg_event_dict) + 2]])
                        new_events = np.append(new_events,new_row, axis=0)
                        din_offset.append(eeg_events[i[0]+1, 0] - eeg_events[i[0], 0])
                        #eeg_events[i[0]+1, 2] = len(eeg_event_dict) + 2 #mmnt DIN onset
                        #new_events.append([eeg_events[i[0], 0], 0 , len(eeg_event_dict) + 2])
                        #new_events = np.append(new_events,[eeg_events[i[0], 0], 0, len(eeg_event_dict) + 2], axis=0)
                        #din_offset.append(eeg_events[i[0]+1, 0] - eeg_events[i[0], 0])

        # append new events to eeg_events
        eeg_events = np.concatenate((eeg_events,new_events))
        eeg_events = eeg_events[eeg_events[:,0].argsort()]
        # add the new stimulus onset DIN labels to the event_dict..
        eeg_event_dict['vs_d'] = len(eeg_event_dict) + 1
        eeg_event_dict['brk_d'] = len(eeg_event_dict) + 1

        #select all of the newly categorized stimulus DIN events
        mask = np.isin(eeg_events[:,2],[eeg_event_dict['vs_d'],eeg_event_dict['brk_d']])
        eeg_stims = eeg_events[mask]
        print('Number of stimulus onset DIN events: ' + str(len(eeg_stims))) #the length of this array should equal the number of stimulus trials in the task

        #calculate the inter trial interval between stimulus onset DIN events
        eeg_iti = np.diff(eeg_stims[:,0])


    elif task_name == 'as' or task_name == 'AS':

        d_ind = [value for key, value in eeg_event_dict.items() if key.startswith('dd')]
        t_ind = [value for key, value in eeg_event_dict.items() if key.startswith('dt')]

        # find the first DIN3 or DIN4 event following either mmns or mmnt events and add new *d events
        for i, e in np.ndenumerate(eeg_events[:,2]):
            if e in d_ind:
                if i[0]+1 < len(eeg_events[:,2]):
                    if eeg_events[i[0]+1, 2] == eeg_event_dict[din_str[0]]:# or eeg_events[i[0]+1, 2] == eeg_event_dict[din_str[1]]:
                        new_row = np.array([[eeg_events[i[0] + 1, 0], 0, len(eeg_event_dict) + 1]])
                        new_events = np.append(new_events,new_row, axis=0)
                        din_offset.append(eeg_events[i[0]+1, 0] - eeg_events[i[0], 0])
            if e in t_ind:
                if i[0]+1 < len(eeg_events[:,2]):
                    if eeg_events[i[0]+1, 2] == eeg_event_dict[din_str[0]]:# or eeg_events[i[0]+1, 2] == eeg_event_dict[din_str[1]]:
                        new_row = np.array([[eeg_events[i[0] + 1, 0], 0, len(eeg_event_dict) + 2]])
                        new_events = np.append(new_events,new_row, axis=0)
                        din_offset.append(eeg_events[i[0]+1, 0] - eeg_events[i[0], 0])

        # append new events to eeg_events
        eeg_events = np.concatenate((eeg_events,new_events))
        eeg_events = eeg_events[eeg_events[:,0].argsort()]
        # add the new stimulus onset DIN labels to the event_dict..
        eeg_event_dict['dd_d'] = len(eeg_event_dict) + 1
        eeg_event_dict['dt_d'] = len(eeg_event_dict) + 1

        #select all of the newly categorized stimulus DIN events
        mask = np.isin(eeg_events[:,2],[eeg_event_dict['dd_d'],eeg_event_dict['dt_d']])
        eeg_stims = eeg_events[mask]
        print('Number of stimulus onset DIN events: ' + str(len(eeg_stims))) #the length of this array should equal the number of stimulus trials in the task

        #calculate the inter trial interval between stimulus onset DIN events
        eeg_iti = np.diff(eeg_stims[:,0])


    elif task_name in ['vs', 'fsp', 'nsp']:
        raise NotImplementedError
    else:
        raise ValueError('Could not determine task name.'
                         f' Expected one of {VALID_TASKS} but got {task_name}')


    #rename DIN* events to eeg_DIN*
    '''
    renamed_dict = {
        (f"eeg_{key}" if key.startswith('DIN') else key): value
        for key, value in eeg_event_dict.items()
    }
    eeg_event_dict = renamed_dict'''

    # print result
    print("EEG event Dictionary:", eeg_event_dict)

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
    et_events = et_events.loc[et_raw_df["DIN_diff"].isin([2, 4])]
    et_events = et_events.reset_index()
    et_stims = et_events.loc[et_events["DIN_diff"].isin([2, 4])]
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



def et_task_events(et_raw_df, et_annot_event_dict, et_annot_events, task_id, din_str):
    # fill NaNs in DIN channel with zeros
    et_raw_df['DIN']=et_raw_df['DIN'].fillna(0)

    # Correct blips to zero for a single sample while DIN8 is on.
    for ind, row in et_raw_df.iterrows():
        if ind < len(et_raw_df)-1:
            if ind > 0:
                if et_raw_df['DIN'][ind] == 0:
                    if et_raw_df['DIN'][ind-1] == 8:
                        if et_raw_df['DIN'][ind+1] == 8:
                            et_raw_df['DIN'].loc[ind] = 8

    # convert the ET DIN channel into ET events
    # find when the DIN channel changes values
    et_raw_df['DIN_diff']=et_raw_df['DIN'].diff()
    # select all non-zero DIN changes
    et_din_events=et_raw_df.loc[et_raw_df['DIN_diff']>0]

    if task_id == 'VEP' or task_id == 'PLR' or task_id == 'GO' or task_id == 'VS' or task_id == 'NSP':
        # there should only be DIN 2 and 4 in the Q1K visual tasks.. however there are frequently binary values greater than 4 indicating that there are anomalous pin4 and pin5 pulses
        # bin2=pin2, bin4=pin3, bin8=pin4, bin16=pin5, bin18=pin2+pin5, bin20=pin3+pin5, bin24=pin4+pin5, bin26=pin2+pin4+pin5, bin28=pin3+pin4+pin5
        # given these anomalous pin4 and pin5 pulses the conversion at pin change time is: binary 2,18,26 = 2, and binary 4,20,28 = 4

        # perform the anomalous DIN conversion
        et_din_events = et_din_events.copy()
        et_din_events['DIN'].loc[et_din_events['DIN'].isin([2,18,26])] = 2
        et_din_events['DIN'].loc[et_din_events['DIN'].isin([4,20,28])] = 4
        #et_din_events['DIN'].loc[et_din_events['DIN'].isin([4,20,24,28])] = 4


        et_din_events = et_din_events.copy()
        et_din_events=et_din_events.loc[et_raw_df['DIN'].isin([2,4])]
        et_din_events = et_din_events.reset_index()
        et_din_events['DIN_diff'] = et_din_events['DIN_diff'].astype(int)
        et_din_events


    #convert DIN_diff to integers
    et_din_events['DIN_diff'] = et_din_events['DIN_diff'].astype(int)

    #add DIN events to et_annot_event_dict with the next available small integer
    existing_indices = set(et_annot_event_dict.values())
    next_index = max(existing_indices) + 1

    for din_diff in et_din_events['DIN_diff']:
        din_key = f'DIN{din_diff}'
        if din_key not in et_annot_event_dict:
            et_annot_event_dict[din_key] = next_index
            next_index += 1

    #create new rows for et_annot_events based on et_din_events
    #map DIN_diff to the new dictionary indices
    et_din_events['mapped_value'] = et_din_events['DIN_diff'].map(lambda x: et_annot_event_dict[f'DIN{x}'])

    #add new rows to et_annot_events
    new_events = np.array([[row['index'], 0, row['mapped_value']] for _, row in et_din_events.iterrows()])
    et_annot_events = np.vstack((et_annot_events, new_events))

    #sort the updated et_annot_events array by the first column (timestamps)
    et_annot_events = et_annot_events[np.argsort(et_annot_events[:, 0])]
    et_annot_events = et_annot_events.astype(int)


    if task_id == 'VEP':
        target_values = {et_annot_event_dict['STIM'], et_annot_event_dict['CS_SPIN']}
        #initialize results and tracking for pruning
        result_events = []
        pruned_indices = set()
        #iterate through rows and apply pruning for 'STIM' and 'CS_SPIN'
        for i, row in enumerate(et_annot_events):
            if i in pruned_indices:
                continue  #skip rows already excluded
            if row[2] in target_values:
                #add the first occurrence of 'STIM' or 'CS_SPIN'
                result_events.append(row)
                #exclude rows of the same type within +500 range
                pruned_indices.update(
                    j for j, other_row in enumerate(et_annot_events)
                    #if abs(other_row[0] - row[0]) <= 500 and other_row[2] == row[2]
                    if other_row[0] - row[0] <= 1000 and other_row[2] == row[2]
                )
            else:
                #retain rows unrelated to 'STIM' or 'CS_SPIN'
                result_events.append(row)
        #convert results back to a numpy array
        result_events = np.array(result_events)
        et_annot_events=result_events

        # add a new key for 'STIM_d' in the dictionary
        stim_d_value = max(et_annot_event_dict.values()) + 1
        et_annot_event_dict['STIM_d'] = stim_d_value

        #process rows to handle 'DIN2' and 'DIN4' for each 'STIM'
        new_rows = []
        used_indices = set()  # To ensure only the first 'DIN2' or 'DIN4' is used

        for stim_index, stim_row in enumerate(et_annot_events):
            if stim_row[2] == et_annot_event_dict['STIM']:
                stim_time = stim_row[0]  # First column of the 'STIM' row
                stim_d_time = None

                # Look for the first 'DIN2' within 1000 ms after this 'STIM'
                for i in range(stim_index + 1, len(et_annot_events)):
                    din2_row = et_annot_events[i]
                    if 'DIN2' in et_annot_event_dict:
                        if (
                            din2_row[2] == et_annot_event_dict['DIN2'] and
                            i not in used_indices and
                            0 <= din2_row[0] - stim_time <= 1000
                        ):
                            stim_d_time = din2_row[0]  # Use 'DIN2' time directly
                            new_rows.append([stim_d_time, 0, stim_d_value])
                            used_indices.add(i)
                            break

                # If no 'DIN2' is found, look for the first 'DIN4' and calculate midpoint if necessary
                if stim_d_time is None:
                    first_din4_time = None
                    second_din4_time = None

                    for i in range(stim_index + 1, len(et_annot_events)):
                        din4_row = et_annot_events[i]
                        if (
                            din4_row[2] == et_annot_event_dict['DIN4'] and
                            i not in used_indices and
                            0 <= din4_row[0] - stim_time <= 1000
                        ):
                            if first_din4_time is None:
                                first_din4_time = din4_row[0]
                                used_indices.add(i)
                            elif second_din4_time is None:
                                second_din4_time = din4_row[0]
                                break

                    # If two DIN4s are found, calculate the midpoint
                    if first_din4_time is not None and second_din4_time is not None:
                        stim_d_time = first_din4_time - (second_din4_time - first_din4_time) // 2
                        new_rows.append([stim_d_time, 0, stim_d_value])

        #add the new rows to the existing events
        et_annot_events = np.vstack([et_annot_events, new_rows])

        #sort the array by the first column for clarity (optional)
        et_annot_events = et_annot_events[et_annot_events[:, 0].argsort()]

    if task_id == 'PLR':
        target_values = {et_annot_event_dict['STIM'], et_annot_event_dict['FIX'], et_annot_event_dict['ISI']}
        #initialize results and tracking for pruning
        result_events = []
        pruned_indices = set()
        #iterate through rows and apply pruning for target)values
        for i, row in enumerate(et_annot_events):
            if i in pruned_indices:
                continue  #skip rows already excluded
            if row[2] in target_values:
                #add the first occurrence of target_values
                result_events.append(row)
                #exclude rows of the same type within +500 range
                pruned_indices.update(
                    j for j, other_row in enumerate(et_annot_events)
                    #if abs(other_row[0] - row[0]) <= 500 and other_row[2] == row[2]
                    if other_row[0] - row[0] <= 1000 and other_row[2] == row[2]
                )
            else:
                #retain rows unrelated to target_values
                result_events.append(row)
        #convert results back to a numpy array
        result_events = np.array(result_events)
        et_annot_events=result_events

        # add a new key for 'STIM_d' in the dictionary
        stim_d_value = max(et_annot_event_dict.values()) + 1
        et_annot_event_dict['STIM_d'] = stim_d_value

        #process rows to handle 'DIN2' and 'DIN4' for each 'STIM'
        new_rows = []
        used_indices = set()  # To ensure only the first 'DIN2' or 'DIN4' is used

        for stim_index, stim_row in enumerate(et_annot_events):
            if stim_row[2] == et_annot_event_dict['STIM']:
                stim_time = stim_row[0]  # First column of the 'STIM' row
                stim_d_time = None

                # Look for the first 'DIN2' within 1000 ms after this 'STIM'
                if 'DIN2' in et_annot_event_dict:
                    for i in range(stim_index + 1, len(et_annot_events)):
                        din2_row = et_annot_events[i]
                        if (
                            din2_row[2] == et_annot_event_dict['DIN2'] and
                            i not in used_indices and
                            0 <= din2_row[0] - stim_time <= 1000
                        ):
                            stim_d_time = din2_row[0]  # Use 'DIN2' time directly
                            new_rows.append([stim_d_time, 0, stim_d_value])
                            used_indices.add(i)
                            break

                # If no 'DIN2' is found, look for the first 'DIN4' and calculate midpoint if necessary
                if stim_d_time is None:
                    if 'DIN4' in et_annot_event_dict:
                        first_din4_time = None
                        second_din4_time = None

                        for i in range(stim_index + 1, len(et_annot_events)):
                            din4_row = et_annot_events[i]
                            if (
                                din4_row[2] == et_annot_event_dict['DIN4'] and
                                i not in used_indices and
                                0 <= din4_row[0] - stim_time <= 1000
                            ):
                                stim_d_time = din4_row[0]  # Use 'DIN2' time directly
                                new_rows.append([stim_d_time, 0, stim_d_value])
                                used_indices.add(i)
                                break

        #add the new rows to the existing events
        et_annot_events = np.vstack([et_annot_events, new_rows])

        #sort the array by the first column for clarity (optional)
        et_annot_events = et_annot_events[et_annot_events[:, 0].argsort()]


    if task_id == 'GO':
        target_values = {et_annot_event_dict['CS_ONSET'], et_annot_event_dict['CS_SPIN'], et_annot_event_dict['ONSET_200MS'], et_annot_event_dict['ONSET_PS'], et_annot_event_dict['REWARD_ONSET']}
        #target_values = {et_annot_event_dict['CS_SPIN']}
        #initialize results and tracking for pruning
        result_events = []
        pruned_indices = set()
        #iterate through rows and apply pruning for target_values
        for i, row in enumerate(et_annot_events):
            if i in pruned_indices:
                continue  #skip rows already excluded
            if row[2] in target_values:
                #add the first occurrence of target_values
                result_events.append(row)
                #exclude rows of the same type within +500 range
                pruned_indices.update(
                    j for j, other_row in enumerate(et_annot_events)
                    #if abs(other_row[0] - row[0]) <= 500 and other_row[2] == row[2]
                    if other_row[0] - row[0] <= 1000 and other_row[2] == row[2]
                )
            else:
                #retain rows unrelated to target_values
                result_events.append(row)
        #convert results back to a numpy array
        result_events = np.array(result_events)
        et_annot_events=result_events

        # add a new key for 'STIM_d' in the dictionary
        stim_d_value = max(et_annot_event_dict.values()) + 1
        et_annot_event_dict['STIM_d'] = stim_d_value

        #process rows to handle 'DIN2' and 'DIN4' for each 'STIM'
        new_rows = []
        used_indices = set()  # To ensure only the first 'DIN2' or 'DIN4' is used

        if 'DIN2' in din_str:
            et_event_list = ['CS_ONSET']
            din_name = 'DIN2'
            #eeg_d_event_list = ['dfoc_d','dfbc_d','dfgc_d']
        else:
            et_event_list = ['CS_SPIN']
            din_name = 'DIN4'
            #eeg_d_event_list = ['dsoc_d','dsbc_d','dsgc_d']

        for stim_index, stim_row in enumerate(et_annot_events):
            if stim_row[2] == et_annot_event_dict[et_event_list[0]]:
                stim_time = stim_row[0]  # First column of the 'STIM' row
                stim_d_time = None

                # Look for the first 'DIN2' within 1000 ms after this 'STIM'
                if din_name in et_annot_event_dict:
                    for i in range(stim_index + 1, len(et_annot_events)):
                        din2_row = et_annot_events[i]
                        if (
                            din2_row[2] == et_annot_event_dict[din_name] and
                            i not in used_indices and
                            0 <= din2_row[0] - stim_time <= 500
                        ):
                            stim_d_time = din2_row[0]  # Use 'DIN2' time directly
                            new_rows.append([stim_d_time, 0, stim_d_value])
                            used_indices.add(i)
                            break

        #add the new rows to the existing events
        et_annot_events = np.vstack([et_annot_events, new_rows])

        #sort the array by the first column for clarity (optional)
        et_annot_events = et_annot_events[et_annot_events[:, 0].argsort()]



    if task_id == 'VS':
        target_values = {et_annot_event_dict['APPLE_FLY_IN'], et_annot_event_dict['DISPLAY_REWARD']}
        #target_values = {et_annot_event_dict['CS_SPIN']}
        #initialize results and tracking for pruning
        result_events = []
        pruned_indices = set()
        #iterate through rows and apply pruning for target_values
        for i, row in enumerate(et_annot_events):
            if i in pruned_indices:
                continue  #skip rows already excluded
            if row[2] in target_values:
                #add the first occurrence of target_values
                result_events.append(row)
                #exclude rows of the same type within +500 range
                pruned_indices.update(
                    j for j, other_row in enumerate(et_annot_events)
                    #if abs(other_row[0] - row[0]) <= 500 and other_row[2] == row[2]
                    if other_row[0] - row[0] <= 1200 and other_row[2] == row[2]
                )
            else:
                #retain rows unrelated to target_values
                result_events.append(row)
        #convert results back to a numpy array
        result_events = np.array(result_events)
        et_annot_events=result_events

        # add a new key for 'STIM_d' in the dictionary
        stim_d_value = max(et_annot_event_dict.values()) + 1
        et_annot_event_dict['STIM_d'] = stim_d_value

        #process rows to handle 'DIN2' and 'DIN4' for each 'STIM'
        new_rows = []
        used_indices = set()  # To ensure only the first 'DIN2' or 'DIN4' is used

        #et_event_list = ['DISPLAY_FIXATION','DISPLAY_SEARCH']
        et_event_list = ['APPLE_FLY_IN']
        din_name = 'DIN4'

        for stim_index, stim_row in enumerate(et_annot_events):
            if stim_row[2] == et_annot_event_dict[et_event_list[0]]: #or stim_row[2] == et_annot_event_dict[et_event_list[1]]:
                stim_time = stim_row[0]  # First column of the 'STIM' row
                stim_d_time = None

                # Look for the first 'DIN2' within 1000 ms after this 'STIM'
                if din_name in et_annot_event_dict:
                    for i in range(stim_index + 1, len(et_annot_events)):
                        din2_row = et_annot_events[i]
                        if (
                            din2_row[2] == et_annot_event_dict[din_name] and
                            i not in used_indices and
                            0 <= din2_row[0] - stim_time <= 500
                        ):
                            stim_d_time = din2_row[0]  # Use 'DIN2' time directly
                            new_rows.append([stim_d_time, 0, stim_d_value])
                            used_indices.add(i)
                            break

        #add the new rows to the existing events
        et_annot_events = np.vstack([et_annot_events, new_rows])

        #sort the array by the first column for clarity (optional)
        et_annot_events = et_annot_events[et_annot_events[:, 0].argsort()]



    if task_id == 'NSP':
        target_values = {et_annot_event_dict['CALIB_ANIMATION_ONSET']}
        #target_values = {et_annot_event_dict['CS_SPIN']}
        #initialize results and tracking for pruning
        result_events = []
        pruned_indices = set()
        #iterate through rows and apply pruning for target_values
        for i, row in enumerate(et_annot_events):
            if i in pruned_indices:
                continue  #skip rows already excluded
            if row[2] in target_values:
                #add the first occurrence of target_values
                result_events.append(row)
                #exclude rows of the same type within +500 range
                pruned_indices.update(
                    j for j, other_row in enumerate(et_annot_events)
                    #if abs(other_row[0] - row[0]) <= 500 and other_row[2] == row[2]
                    if other_row[0] - row[0] <= 1200 and other_row[2] == row[2]
                )
            else:
                #retain rows unrelated to target_values
                result_events.append(row)
        #convert results back to a numpy array
        result_events = np.array(result_events)
        et_annot_events=result_events

        # add a new key for 'STIM_d' in the dictionary
        stim_d_value = max(et_annot_event_dict.values()) + 1
        et_annot_event_dict['STIM_d'] = stim_d_value

        #process rows to handle 'DIN2' and 'DIN4' for each 'STIM'
        new_rows = []
        used_indices = set()  # To ensure only the first 'DIN2' or 'DIN4' is used

        et_event_list = ['CALIB_ANIMATION_ONSET']
        din_name = 'DIN4'

        for stim_index, stim_row in enumerate(et_annot_events):
            if stim_row[2] == et_annot_event_dict[et_event_list[0]]: #or stim_row[2] == et_annot_event_dict[et_event_list[1]]:
                stim_time = stim_row[0]  # First column of the 'STIM' row
                stim_d_time = None

                # Look for the first 'DIN2' within 1000 ms after this 'STIM'
                if din_name in et_annot_event_dict:
                    for i in range(stim_index + 1, len(et_annot_events)):
                        din2_row = et_annot_events[i]
                        if (
                            din2_row[2] == et_annot_event_dict[din_name] and
                            i not in used_indices and
                            0 <= din2_row[0] - stim_time <= 500
                        ):
                            stim_d_time = din2_row[0]  # Use 'DIN2' time directly
                            new_rows.append([stim_d_time, 0, stim_d_value])
                            used_indices.add(i)
                            break

        #add the new rows to the existing events
        et_annot_events = np.vstack([et_annot_events, new_rows])

        #sort the array by the first column for clarity (optional)
        et_annot_events = et_annot_events[et_annot_events[:, 0].argsort()]




    #rename DIN* events to eeg_DIN*
    renamed_dict = {
        (f"et_{key}" if key.startswith('DIN') else key): value
        for key, value in et_annot_event_dict.items()
    }
    et_annot_event_dict = renamed_dict

    # print result
    print("Updated Dictionary:", et_annot_event_dict)


    return et_annot_event_dict, et_annot_events, et_raw_df


















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

'''
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
'''

# =============================================================================
# ET FUNCTIONS (ported from old q1k_init_tools.py)
# =============================================================================

def et_read(path, blink_interp, fill_nans, resamp):
    """Read eye-tracking .asc file and return raw + dataframe."""
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


#def _times_align(a_times, b_times):
#    """Align a_times to b_times by nearest match."""
#    return np.array([a_times[np.argmin(np.abs(a_times - b))] for b in b_times])

'''
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
        elif task_id == 'PLR':
            print(f"PLR: trimming to {n_et} points")
            eeg_times = eeg_times[:n_et]
            # FIX: ET has 2 sync points per stimulus, keeping only every other
            #et_times = et_times[::2]
            #et_stims_np = et_stims_np[::2]
            #print(f"PLR: deduplicated ET sync points: {len(et_times)}")
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
'''


def eeg_et_align(eeg_event_dict, et_event_dict, eeg_events, et_events, eeg_stims, et_stims, eeg_sfreq, et_sfreq, task_id):
    eeg_times = eeg_stims[:, 0] / eeg_sfreq
    et_times = et_stims[:, 0] / et_sfreq

    n_eeg_times = len(eeg_times)
    n_et_times = len(et_times)

    if n_eeg_times > n_et_times:
        if task_id == 'GO':
            # Get indices for events starting with 'ds'
            ds_event_indices = {v for k, v in eeg_event_dict.items() if k.startswith('ds')}
            # Filter eeg_events for these indices
            ds_events = eeg_events[np.isin(eeg_events[:, 2], list(ds_event_indices))]
            # Find the earliest time
            first_ds_time = np.min(ds_events[:, 0]) if ds_events.size > 0 else None
            first_eeg_ds_sec = first_ds_time / eeg_sfreq

            # Get indices for events starting with 'ds'
            ds_event_indices = {v for k, v in et_event_dict.items() if k.startswith('CS_SPIN')}
            # Filter eeg_events for these indices
            ds_events = et_events[np.isin(et_events[:, 2], list(ds_event_indices))]
            # Find the earliest time
            first_ds_time = np.min(ds_events[:, 0]) if ds_events.size > 0 else None
            first_et_ds_sec = first_ds_time / et_sfreq

            first_ds_offset_sec = first_eeg_ds_sec - first_et_ds_sec


            eeg_times = eeg_stims[:, 0] / eeg_sfreq
            eeg_times = eeg_times - first_ds_offset_sec
            et_times = et_stims[:, 0] / et_sfreq

            a_times = eeg_times
            b_times = et_times

            # Find closest matches
            closest_indices = np.array([np.argmin(np.abs(a_times - b)) for b in b_times])
            closest_a_times = a_times[closest_indices] + first_ds_offset_sec

            # Print result
            #print(closest_a_times)
            eeg_times = closest_a_times
        else:
            print("there are more eeg_times than there are et_times.. attempting align")
            eeg_times = times_align(eeg_times,et_times)
            #if eeg_times:
            #    "..resolved eeg/et times discrepancy..."
            #else:
            #    "..could not resolved eeg/et times discrepancy..."

    elif n_eeg_times < n_et_times:
        print("there are more et_times than there are eeg_times.. attempting align")
        et_times = times_align(et_times,eeg_times)
    else:
        print("there are the same number of eeg_times and et_times.. continuing")

    #check if alignment was successful..
    n_eeg_times = len(eeg_times)
    n_et_times = len(et_times)
    if n_eeg_times != n_et_times:
        print("EEG and ET times alignment was not successful... abandoning sync procedures...")
    else:
        #create the sync_time events for the EEG and ET data.
        #convert times to samples..
        #eeg_samps = eeg_stims[:, 0]
        #et_samps = et_stims[:, 0]
        eeg_samps = eeg_times * eeg_sfreq
        et_samps = et_times * et_sfreq
        #add "*_sync_time" to the dictionary
        eeg_event_dict['eeg_sync_time'] = len(eeg_event_dict) + 1
        et_event_dict['et_sync_time'] = len(et_event_dict) + 1
        #add rows to the events array for "*_sync_time"
        eeg_sync_time_rows = [[samp, 0, eeg_event_dict['eeg_sync_time']] for samp in eeg_samps]
        et_sync_time_rows = [[samp, 0, et_event_dict['et_sync_time']] for samp in et_samps]
        #combine the new rows with the existing events
        eeg_events = np.vstack([eeg_events, eeg_sync_time_rows])
        eeg_events = eeg_events[eeg_events[:, 0].argsort()]  # Sort by the first column (time)
        et_events = np.vstack([et_events, et_sync_time_rows])
        et_events = et_events[et_events[:, 0].argsort()]  # Sort by the first column (time)

        print("Updated EEG event dictionary:")
        eeg_event_dict

    return eeg_event_dict, et_event_dict, eeg_events, et_events, eeg_times, et_times




def times_align(a_times, b_times):
    b_intervals = np.diff(b_times)

    def find_best_index_to_remove(a_times, target_intervals):
        min_deviation = float('inf')
        best_index = None
        for i in range(1, len(a_times) - 1):
            modified_a_times = np.delete(a_times, i)
            modified_intervals = np.diff(modified_a_times)
            deviation = np.sum(np.abs(modified_intervals - target_intervals))
            if deviation < min_deviation:
                min_deviation = deviation
                best_index = i
        return best_index

    index_to_remove = find_best_index_to_remove(a_times, b_intervals)
    print("Index to remove:", index_to_remove)
    a_times = np.delete(a_times, index_to_remove)
    return a_times



def eeg_et_combine(eeg_raw, et_raw, eeg_times, et_times, eeg_events, eeg_event_dict, et_events, et_event_dict):
    eeg_raw.load_data()
    et_raw.load_data()
    mne.preprocessing.realign_raw(eeg_raw, et_raw, eeg_times, et_times, verbose="error")
    eeg_raw.add_channels([et_raw], force_update_info=True)
    eeg_event_dict_r = {value: key for key, value in eeg_event_dict.items()}
    eeg_annots = mne.annotations_from_events(events=eeg_events, event_desc=eeg_event_dict_r, sfreq=eeg_raw.info["sfreq"], orig_time=eeg_raw.info["meas_date"])
    eeg_raw.set_annotations(eeg_annots)
    return eeg_raw




def write_eeg(eeg_raw, eeg_event_dict, eeg_events, subject_id_out, session_id, task_id_out, project_path, device_info):
    from pathlib import Path
    eeg_raw.info["line_freq"] = 60
    eeg_raw.info['device_info']=device_info
    eeg_raw.info['device_info']['type'] = eeg_raw.info['device_info']['type'].replace(' ', '-')
    '''
    def fillna(raw, fill_val=0):
        return mne.io.RawArray(np.nan_to_num(raw.get_data(), nan=fill_val), raw.info)
    eeg_raw=fillna(eeg_raw,fill_val=0)'''


    #deriv_root = Path(project_path) / "derivatives" / "init" / task_id_out / f"sub-{subject_id_out}" / f"ses-{session_id}"


    root_path = Path(project_path) / "derivatives" / "init" / task_id_out
    root_path.mkdir(parents=True, exist_ok=True)
    eeg_bids_path = mne_bids.BIDSPath(subject=subject_id_out, session=session_id, task=task_id_out, run="1", datatype="eeg", root=str(root_path))


    print(f"Writing EEG to: {eeg_bids_path}")
    eeg_raw.set_annotations(None)

    mne_bids.write_raw_bids(raw=eeg_raw, bids_path=eeg_bids_path, events=eeg_events, event_id=eeg_event_dict, format="EDF", overwrite=True, allow_preload=True)
    return eeg_bids_path









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

    et_out_path = str(eeg_bids_path)
    et_out_path = et_out_path.replace("/eeg/", "/et/")
    et_out_path = et_out_path.replace("_eeg.edf", "_et.fif")
    os.makedirs(os.path.dirname(et_out_path), exist_ok=True)
    et_raw.save(et_out_path, overwrite=True)
    print(f"ET .fif saved: {et_out_path}")
    return et_out_path
