#!/usr/bin/env python3
"""
MHC GO ET Conversion Script
Converts GO_raw .edf files to .asc and places them in
sourcedata/MHC/et/Q1K_MHC_200XXX_YY/ folders.

Q265_P -> Q1K_MHC_200265_P -> sourcedata/MHC/et/Q1K_MHC_200265_P/265P_GO.asc
"""

import re
from pathlib import Path
from eyelinkio.edf.to_asc import to_asc

WD         = Path("/lustre07/scratch/rsweety/white_paper/wd")
GO_RAW     = WD / "sourcedata" / "MHC" / "et" / "GO_raw"
MHC_ET     = WD / "sourcedata" / "MHC" / "et"
MHC_EEG    = WD / "sourcedata" / "MHC" / "eeg"

# Build EEG lookup: family_id -> Q1K_MHC_200XXX_YY
eeg_lookup = {}
for d in MHC_EEG.iterdir():
    if not d.is_dir(): continue
    m = re.match(r'Q1K_MHC_200(\d+)_(\w+)', d.name)
    if m:
        fam_id = m.group(1)  # e.g. "265"
        suffix = m.group(2)  # e.g. "P"
        key = f"{fam_id}_{suffix}"  # "265_P"
        eeg_lookup[key] = d.name   # "Q1K_MHC_200265_P"

print(f"EEG lookup built: {len(eeg_lookup)} MHC subjects")

# Process GO_raw
converted = 0
skipped   = 0
missing   = 0
failed    = 0

for subj_dir in sorted(GO_RAW.iterdir()):
    if not subj_dir.is_dir(): continue
    et_id = subj_dir.name  # e.g. "Q265_P"

    # Extract family_id and suffix from Q265_P
    m = re.match(r'Q(\d+)_(\w+)', et_id)
    if not m:
        print(f"  SKIP (no match): {et_id}")
        skipped += 1
        continue

    key = f"{m.group(1)}_{m.group(2)}"  # "265_P"
    q1k_id = eeg_lookup.get(key)

    if not q1k_id:
        print(f"  MISSING EEG match for {et_id} (key={key})")
        missing += 1
        continue

    # Find the .edf file
    edf_files = list(subj_dir.glob("*.edf"))
    if not edf_files:
        print(f"  NO EDF: {et_id}")
        skipped += 1
        continue

    edf_path = edf_files[0]

    # Output path: sourcedata/MHC/et/Q1K_MHC_200265_P/265P_GO.asc
    out_dir = MHC_ET / q1k_id
    out_dir.mkdir(exist_ok=True)

    # BIDS id without underscore e.g. "265P"
    bids_id = key.replace("_", "")
    asc_path = out_dir / f"{bids_id}_GO.asc"

    if asc_path.exists():
        print(f"  EXISTS: {asc_path.name}")
        skipped += 1
        continue

    print(f"  Converting {et_id} -> {q1k_id} -> {asc_path.name}")
    try:
        to_asc(str(edf_path), str(asc_path))
        converted += 1
        print(f"    OK: {asc_path}")
    except Exception as e:
        print(f"    FAILED: {e}")
        failed += 1

print(f"\n=== DONE ===")
print(f"Converted : {converted}")
print(f"Skipped   : {skipped}")
print(f"Missing   : {missing}")
print(f"Failed    : {failed}")
