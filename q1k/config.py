"""Central configuration constants for the Q1K pipeline."""

# Supported experimental tasks
VALID_TASKS = ["RS", "RSRio", "VEP", "AEP", "GO", "PLR", "VS", "NSP", "TO"]

# Tasks that skip the DIN offset procedure (only have simple events)
NO_DIN_OFFSET_TASKS = {"RSRio"}

# Task name aliases (lowercase → canonical uppercase)
TASK_ALIASES = {
    "rest": "RS",
    "rs": "RS",
    "rsrio": "RSRio",
    "vp": "VEP",
    "vep": "VEP",
    "ap": "AEP",
    "aep": "AEP",
    "go": "GO",
    "plr": "PLR",
    "vs": "VS",
    "nsp": "NSP",
    "as": "TO",
    "to": "TO",
    "mn": "AEP",
    "fsp": "NSP",
}

# EOG channels to mark as bad before pylossless processing
EOG_CHANNELS = ["E125", "E126", "E127", "E128"]

# Site codes
SITE_CODES = ["HSJ", "MHC", "NIM"]

# Default BIDS parameters
DEFAULT_SESSION_ID = "01"
DEFAULT_RUN_ID = "1"

# Default HPC project account
DEFAULT_SLURM_ACCOUNT = "def-emayada"

# Derivative directory names
DERIV_PYLOSSLESS = "derivatives/pylossless"
DERIV_SYNC_LOSS = "derivatives/sync_loss"
DERIV_SEGMENT = "derivatives/segment"
DERIV_AUTOREJ = "derivatives/autorej"

# Frequency bands for spectral analysis
FREQ_BANDS = {
    "delta": (0, 4),
    "theta": (4, 7),
    "alpha": (8, 12),
    "beta": (13, 30),
    "gamma": (30, 45),
}

# Frontal ROI channels used in resting state analysis
FRONTAL_ROI = [
    "E18", "E19", "E23", "E24", "E27",
    "E3", "E4", "E10", "E118", "E123", "E124",
]

# ── Pipeline tracking constants ─────────────────────────────────────

# Ordered pipeline stages for tracking data flow
PIPELINE_STAGES = [
    "EEG Raw Files",
    "BIDS",
    "Pylossless",
    "ET_sync_loss",
    "Segmentation",
    "Autoreject",
]

# REDCap column rename mappings for task completion CSV
REDCAP_TASK_COLUMNS = {
    "eeg_rs_done": "RS",
    "eeg_to_done": "TO",
    "eeg_go_done": "GO",
    "eeg_vep_done": "VEP",
    "eeg_aep_done": "AEP",
    "eeg_nsp_done": "NSP",
    "eeg_pl_done": "PLR",
    "eeg_vs_done": "VS",
    "eeg_mmn_done": "MMN",
}

# REDCap task failure reason column mappings
REDCAP_FAIL_COLUMNS = {
    "eeg_rs_notdone": "RS_fail_reason",
    "eeg_to_notdone": "TO_fail_reason",
    "eeg_go_notdone": "GO_fail_reason",
    "eeg_vep_notdone": "VEP_fail_reason",
    "eeg_aep_notdone": "AEP_fail_reason",
    "eeg_nsp_notdone": "NSP_fail_reason",
    "eeg_pl_notdone": "PLR_fail_reason",
    "eeg_vs_notdone": "VS_fail_reason",
}

# REDCap column rename mappings for demographics CSV
REDCAP_DEMO_COLUMNS = {
    "enr2_pro_sex": "sex",
    "q1k_disorderdiag_1": "ndd",
    "cfq_diag_asd": "asd",
    "cfq_diag_adhd": "adhd",
    "ghf_asd": "asd_healthform",
    "ev_status": "affected_status",
    "reg_diag_asd": "registry_confirmed_asd",
    "icf_form_phase_3_complete": "phase_3_consented",
}

# REDCap session log column rename mappings
REDCAP_SESSION_COLUMNS = {
    "eeg_code_software": "et_id",
    "eeget_date_v2_v2": "visit_date",
    "eeg_age_years_testdate": "eeg_age",
    "eeg_sex_birth": "eeg_sex",
    "eeg_diagnosis": "eeg_diagnosis",
    "eeg_attempted": "eeg_attempt",
    "eeg_participant_code": "participant_code",
    "eeg_attempted_reasons": "EEG_failed_attempt_reason",
}

# Site ID prefixes used in Q1K IDs (for q1k_to_bids conversion)
SITE_ID_PREFIXES = {
    "HSJ": "100",
    "MHC": "200",
    "GA": "600",
    "SHR": "5526",
    "OIM": "4529",
    "NIM": "3530",
}

# Dash-separated family ID prefixes
FAMILY_ID_PREFIXES = ["1025", "1525", "2524"]
