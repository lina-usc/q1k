"""Central configuration constants for the Q1K pipeline."""

# Supported experimental tasks
VALID_TASKS = ["RS", "VEP", "AEP", "GO", "PLR", "VS", "NSP", "TO"]

# Task name aliases (lowercase → canonical uppercase)
TASK_ALIASES = {
    "rest": "RS",
    "rs": "RS",
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
