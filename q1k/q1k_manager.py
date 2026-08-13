#!/usr/bin/env python3
"""Q1K pipeline manager for Narval.

This manager is intentionally for the five pipeline stages:

    init -> pylossless -> sync_loss -> segment -> autoreject

It matches the directory layout currently used in the white paper workspace:

    source_prime/<SITE>/<SUBJECT>/eeg/<TASK>/*.mff
    source_prime/<SITE>/<SUBJECT>/et/<TASK>/*.{asc,edf}
    derivatives/init/<TASK>/sub-<SUBJECT>/ses-01/eeg/*_eeg.edf
    derivatives/init/<TASK>/sub-<SUBJECT>/ses-01/et/*_et.fif
    derivatives/pylossless/<TASK>/sub-<SUBJECT>/ses-01/eeg/*_eeg.edf
    derivatives/sync_loss/<TASK>/sub-<SUBJECT>/ses-01/eeg/*_eeg.edf
    derivatives/segment/epoch_fif_files/<TASK>/*_epo.fif
    derivatives/autoreject/epoch_fif_files/<TASK>/*_epo.fif

Common commands
---------------
Scan and write CSVs:

    python3 q1k_manager.py scan

Print summary:

    python3 q1k_manager.py summary

Build pending lists for the next stage:

    python3 q1k_manager.py lists --stage autoreject

Submit one pending list:

    python3 q1k_manager.py submit --stage segment --site MHC --task VEP --max-jobs 5

Open lightweight text dashboard:

    python3 q1k_manager.py dashboard
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

DEFAULT_WD = Path(os.environ.get("Q1K_WD", "/lustre07/scratch/rsweety/white_paper/wd"))
DEFAULT_SITES = ("HSJ", "MHC", "NIM")
DEFAULT_TASKS = ("GO", "PLR", "VEP")

STAGES = ("init", "pylossless", "sync_loss", "segment", "autoreject")
STAGE_LABEL = {
    "init": "INIT",
    "pylossless": "PYLL",
    "sync_loss": "SYNC",
    "segment": "SEG",
    "autoreject": "AR",
}
STAGE_CLI = {
    "init": "q1k-init",
    "pylossless": "q1k-pylossless",
    "sync_loss": "q1k-sync-loss",
    "segment": "q1k-segment",
    "autoreject": "q1k-autorej",
}
STAGE_RESOURCES = {
    "init": ("12:00:00", "16G", 2),
    "pylossless": ("1-00:00:00", "32G", 2),
    "sync_loss": ("08:00:00", "24G", 2),
    "segment": ("06:00:00", "24G", 2),
    "autoreject": ("08:00:00", "32G", 2),
}
STRICT_ET_TASKS = {"GO", "PLR"}
ERROR_WORDS = (
    "Traceback",
    "MarimoExceptionRaisedError",
    "Error:",
    "ERROR",
    "Exception",
    "FileNotFoundError",
    "ValueError",
    "KeyError",
    "No such file",
    "Timed out",
    "TIMEOUT",
    "Killed",
    "OutOfMemory",
)


def norm_subject(value: str) -> str:
    value = value.strip()
    if value.startswith("sub-"):
        value = value[4:]
    return value.strip()


def site_from_subject(subject: str) -> str:
    for site in DEFAULT_SITES:
        if subject.startswith(site):
            return site
    if subject.startswith("2"):
        return "MHC"
    return "UNKNOWN"


def task_dir(base: Path, task: str) -> Path:
    """Return a task directory, accepting case drift if present."""
    direct = base / task
    if direct.exists():
        return direct
    if base.exists():
        for child in base.iterdir():
            if child.is_dir() and child.name.upper() == task.upper():
                return child
    return direct


def first_existing(paths: Iterable[Path]) -> Optional[Path]:
    for path in paths:
        if path.exists():
            return path
    return None


def glob_count(path: Path, pattern: str, recursive: bool = False) -> int:
    if not path.exists():
        return 0
    if recursive:
        return sum(1 for _ in path.rglob(pattern))
    return sum(1 for _ in path.glob(pattern))


def extract_subject_from_epoch_name(path: Path) -> Optional[str]:
    match = re.search(r"sub-([^_/]+)", path.name)
    if match:
        return norm_subject(match.group(1))
    match = re.search(r"sub-([^/]+)", str(path))
    if match:
        return norm_subject(match.group(1))
    return None


def is_error_text(text: str) -> bool:
    return any(word in text for word in ERROR_WORDS)


def classify_log(text: str) -> str:
    """Classify a Slurm/notebook log into one compact reason."""
    low = text.lower()
    if not text:
        return "no_log_found"
    if "pupil_left" in low and ("could not be interpreted" in low or "picks" in low):
        return "pupil_left_bug_old_log"
    if "et .fif not found" in low or "_et.fif" in low and "no such file" in low:
        return "missing_et_fif"
    if "et_sync_time" in low and is_error_text(text):
        return "missing_or_bad_et_sync_time"
    if "eeg_sync_time" in low and is_error_text(text):
        return "missing_or_bad_eeg_sync_time"
    if "sync count" in low or "same number of sync" in low or "sync points" in low and "valueerror" in low:
        return "sync_count_mismatch"
    if "no stim channels found" in low:
        return "no_stim_channels"
    if "missing_go_labels" in low:
        return "missing_GO_labels"
    if "info1.xml" in low and ("no such file" in low or "no element found" in low):
        return "bad_mff_info_xml"
    if "expaterror" in low or "no element found" in low:
        return "bad_xml_or_empty_mff"
    if "could not open file" in low and ".edf" in low:
        return "bad_edf"
    if "dtoc" in low or "dtbc" in low or "dtgc" in low:
        if "keyerror" in low or "marimoexception" in low:
            return "missing_task_event_label"
    if "sample_fields" in low:
        return "sample_fields_error"
    if "outofmemory" in low or "out of memory" in low or "oom" in low or "killed" in low:
        return "memory_or_killed"
    if "timed out" in low or "time limit" in low or "timeout" in low:
        return "timeout"
    if "filenotfounderror" in low or "no such file or directory" in low:
        return "file_not_found"
    if "keyerror" in low:
        return "key_error"
    if "valueerror" in low:
        return "value_error"
    if "marimoexceptionraisederror" in low:
        return "marimo_exception"
    if "traceback" in low or "error:" in low or "exception" in low:
        return "other_error"
    if "status: 0" in low or "report saved" in low or "saved cleaned epochs" in low:
        return "log_says_done_but_output_missing"
    return "unknown_log_reason"


@dataclass
class PathsFound:
    source_mff_count: int = 0
    source_et_count: int = 0
    init_eeg_count: int = 0
    init_et_count: int = 0
    pylossless_eeg_count: int = 0
    sync_eeg_count: int = 0
    segment_epo_count: int = 0
    autoreject_epo_count: int = 0


@dataclass
class StageResult:
    status: str
    reason: str
    log: str = ""


@dataclass
class InventoryRow:
    site: str
    task: str
    subject: str
    source_expected: str
    source_mff_count: int
    source_et_count: int
    init_eeg_count: int
    init_et_count: int
    pylossless_eeg_count: int
    sync_eeg_count: int
    segment_epo_count: int
    autoreject_epo_count: int
    init_status: str
    init_reason: str
    init_log: str
    pylossless_status: str
    pylossless_reason: str
    pylossless_log: str
    sync_loss_status: str
    sync_loss_reason: str
    sync_loss_log: str
    segment_status: str
    segment_reason: str
    segment_log: str
    autoreject_status: str
    autoreject_reason: str
    autoreject_log: str
    next_stage: str
    next_action: str
    remark: str


class Q1KManager:
    def __init__(
        self,
        wd: Path = DEFAULT_WD,
        sites: Sequence[str] = DEFAULT_SITES,
        tasks: Sequence[str] = DEFAULT_TASKS,
        out_dir: Optional[Path] = None,
    ) -> None:
        self.wd = wd
        self.sites = tuple(sites)
        self.tasks = tuple(tasks)
        self.source_prime = wd / "source_prime"
        self.derivatives = wd / "derivatives"
        self.slurm_output = wd / "slurm_output"
        self.venv_activate = wd / "q1k_venv_scratch" / "bin" / "activate"
        self.out_dir = out_dir or (wd / "q1k_manager_inventory")
        self.log_cache: Dict[Path, str] = {}
        self.log_files: List[Path] = []

    def discover_subjects(self) -> Dict[Tuple[str, str, str], None]:
        subjects: Dict[Tuple[str, str, str], None] = {}

        for site in self.sites:
            site_dir = self.source_prime / site
            if not site_dir.exists():
                continue
            for subj_dir in sorted(site_dir.iterdir()):
                if not subj_dir.is_dir() or subj_dir.name.startswith("."):
                    continue
                subject = norm_subject(subj_dir.name)
                for task in self.tasks:
                    eeg_dir = task_dir(subj_dir / "eeg", task)
                    et_dir = task_dir(subj_dir / "et", task)
                    if eeg_dir.exists() or et_dir.exists():
                        subjects[(site, task, subject)] = None

        for task in self.tasks:
            for stage_base in (
                self.derivatives / "init" / task,
                self.derivatives / "pylossless" / task,
                self.derivatives / "sync_loss" / task,
            ):
                for sub_dir in stage_base.glob("sub-*"):
                    if sub_dir.is_dir():
                        subject = norm_subject(sub_dir.name)
                        subjects[(site_from_subject(subject), task, subject)] = None

            for epoch_base in (
                self.derivatives / "segment" / "epoch_fif_files" / task,
                self.derivatives / "autoreject" / "epoch_fif_files" / task,
                self.derivatives / "autorej" / "epoch_fif_files" / task,
            ):
                for epoch_file in epoch_base.glob("*_epo.fif"):
                    subject = extract_subject_from_epoch_name(epoch_file)
                    if subject:
                        subjects[(site_from_subject(subject), task, subject)] = None

        return {key: None for key in sorted(subjects.keys())}

    def paths_for(self, site: str, task: str, subject: str) -> PathsFound:
        subj_src = self.source_prime / site / subject
        eeg_dir = task_dir(subj_src / "eeg", task)
        et_dir = task_dir(subj_src / "et", task)

        init_eeg = self.derivatives / "init" / task / f"sub-{subject}" / "ses-01" / "eeg"
        init_et = self.derivatives / "init" / task / f"sub-{subject}" / "ses-01" / "et"
        pyll_eeg = self.derivatives / "pylossless" / task / f"sub-{subject}" / "ses-01" / "eeg"
        sync_eeg = self.derivatives / "sync_loss" / task / f"sub-{subject}" / "ses-01" / "eeg"
        segment_dir = self.derivatives / "segment" / "epoch_fif_files" / task
        ar_dir = self.derivatives / "autoreject" / "epoch_fif_files" / task
        ar_old_dir = self.derivatives / "autorej" / "epoch_fif_files" / task

        return PathsFound(
            source_mff_count=glob_count(eeg_dir, "*.mff", recursive=True),
            source_et_count=glob_count(et_dir, "*.asc", recursive=True) + glob_count(et_dir, "*.edf", recursive=True),
            init_eeg_count=glob_count(init_eeg, "*_eeg.edf"),
            init_et_count=glob_count(init_et, "*_et.fif"),
            pylossless_eeg_count=glob_count(pyll_eeg, "*_eeg.edf"),
            sync_eeg_count=glob_count(sync_eeg, "*_eeg.edf"),
            segment_epo_count=glob_count(segment_dir, f"sub-{subject}_*_task-{task}_*_epo.fif"),
            autoreject_epo_count=(
                glob_count(ar_dir, f"sub-{subject}_*_task-{task}_*_epo.fif")
                + glob_count(ar_old_dir, f"sub-{subject}_*_task-{task}_*_epo.fif")
            ),
        )

    def load_logs(self) -> None:
        if not self.slurm_output.exists():
            self.log_files = []
            return
        self.log_files = sorted(self.slurm_output.glob("*.out"), key=lambda p: p.stat().st_mtime, reverse=True)

    def _read_log(self, path: Path) -> str:
        if path not in self.log_cache:
            try:
                self.log_cache[path] = path.read_text(errors="replace")
            except Exception:
                self.log_cache[path] = ""
        return self.log_cache[path]

    def find_log(self, site: str, task: str, subject: str, stage: str) -> Tuple[str, str]:
        if not self.log_files:
            self.load_logs()

        stage_patterns = {
            "init": ("INIT",),
            "pylossless": ("PYLL", "PYLOSS", "PYL"),
            "sync_loss": ("SYNC",),
            "segment": ("SEG", "SEGMENT"),
            "autoreject": ("_AR", "AUTOREJ", "AR"),
        }[stage]

        candidates: List[Path] = []
        for lf in self.log_files:
            name = lf.name.upper()
            if task.upper() not in name:
                continue
            if not any(pat in name for pat in stage_patterns):
                continue
            if site.upper() in name or subject.upper() in name:
                candidates.append(lf)

        # Some rerun logs do not include the site but do include task/stage.
        if not candidates:
            for lf in self.log_files:
                name = lf.name.upper()
                if task.upper() in name and any(pat in name for pat in stage_patterns):
                    candidates.append(lf)

        for lf in candidates:
            text = self._read_log(lf)
            if subject in text or subject in lf.name:
                return classify_log(text), str(lf)

        return "no_log_found", ""

    def stage_result(self, site: str, task: str, subject: str, paths: PathsFound, stage: str) -> StageResult:
        if stage == "init":
            if paths.init_eeg_count:
                return StageResult("completed", "init_eeg_exists")
            if paths.source_mff_count == 0:
                return StageResult("blocked", "source_mff_missing")
            reason, log = self.find_log(site, task, subject, stage)
            return StageResult("failed_or_pending", reason, log)

        if stage == "pylossless":
            if paths.pylossless_eeg_count:
                return StageResult("completed", "pylossless_eeg_exists")
            if paths.init_eeg_count == 0:
                return StageResult("blocked", "init_eeg_missing")
            reason, log = self.find_log(site, task, subject, stage)
            return StageResult("failed_or_pending", reason, log)

        if stage == "sync_loss":
            if paths.sync_eeg_count:
                return StageResult("completed", "sync_eeg_exists")
            if paths.pylossless_eeg_count == 0:
                return StageResult("blocked", "pylossless_eeg_missing")
            if task in STRICT_ET_TASKS and paths.init_et_count == 0:
                reason, log = self.find_log(site, task, subject, stage)
                if reason == "no_log_found":
                    reason = "missing_et_fif"
                return StageResult("blocked", reason, log)
            reason, log = self.find_log(site, task, subject, stage)
            return StageResult("failed_or_pending", reason, log)

        if stage == "segment":
            if paths.segment_epo_count:
                return StageResult("completed", "segment_epoch_exists")
            if paths.sync_eeg_count == 0:
                return StageResult("blocked", "sync_eeg_missing")
            reason, log = self.find_log(site, task, subject, stage)
            return StageResult("failed_or_pending", reason, log)

        if stage == "autoreject":
            if paths.autoreject_epo_count:
                return StageResult("completed", "autoreject_epoch_exists")
            if paths.segment_epo_count == 0:
                return StageResult("blocked", "segment_epoch_missing")
            reason, log = self.find_log(site, task, subject, stage)
            return StageResult("failed_or_pending", reason, log)

        return StageResult("unknown", "unknown_stage")

    def build_row(self, site: str, task: str, subject: str) -> InventoryRow:
        paths = self.paths_for(site, task, subject)
        results = {stage: self.stage_result(site, task, subject, paths, stage) for stage in STAGES}
        next_stage = ""
        next_action = "complete"
        remark_parts: List[str] = []

        for stage in STAGES:
            result = results[stage]
            if result.status != "completed":
                next_stage = stage
                if result.status == "blocked":
                    next_action = f"blocked: {result.reason}"
                else:
                    next_action = f"rerun_or_review: {result.reason}"
                break

        for stage in STAGES:
            result = results[stage]
            if result.status != "completed":
                remark_parts.append(f"{stage}:{result.status}:{result.reason}")

        return InventoryRow(
            site=site,
            task=task,
            subject=subject,
            source_expected="YES" if paths.source_mff_count else "NO",
            source_mff_count=paths.source_mff_count,
            source_et_count=paths.source_et_count,
            init_eeg_count=paths.init_eeg_count,
            init_et_count=paths.init_et_count,
            pylossless_eeg_count=paths.pylossless_eeg_count,
            sync_eeg_count=paths.sync_eeg_count,
            segment_epo_count=paths.segment_epo_count,
            autoreject_epo_count=paths.autoreject_epo_count,
            init_status=results["init"].status,
            init_reason=results["init"].reason,
            init_log=results["init"].log,
            pylossless_status=results["pylossless"].status,
            pylossless_reason=results["pylossless"].reason,
            pylossless_log=results["pylossless"].log,
            sync_loss_status=results["sync_loss"].status,
            sync_loss_reason=results["sync_loss"].reason,
            sync_loss_log=results["sync_loss"].log,
            segment_status=results["segment"].status,
            segment_reason=results["segment"].reason,
            segment_log=results["segment"].log,
            autoreject_status=results["autoreject"].status,
            autoreject_reason=results["autoreject"].reason,
            autoreject_log=results["autoreject"].log,
            next_stage=next_stage,
            next_action=next_action,
            remark=";".join(remark_parts) if remark_parts else "complete",
        )

    def scan(self) -> List[InventoryRow]:
        self.load_logs()
        subjects = self.discover_subjects()
        rows = [self.build_row(site, task, subject) for site, task, subject in subjects]
        rows.sort(key=lambda r: (r.site, r.task, r.subject))
        return rows

    def write_outputs(self, rows: Sequence[InventoryRow]) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        master_csv = self.out_dir / "q1k_master_inventory.csv"
        master_json = self.out_dir / "q1k_master_inventory.json"
        summary_csv = self.out_dir / "q1k_summary_by_site_task.csv"
        failures_csv = self.out_dir / "q1k_failures_reasonwise.csv"

        fieldnames = list(InventoryRow.__dataclass_fields__.keys())
        with master_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(asdict(row))

        master_json.write_text(json.dumps([asdict(r) for r in rows], indent=2))

        summary_rows = self.summary(rows)
        with summary_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()) if summary_rows else [])
            if summary_rows:
                writer.writeheader()
                writer.writerows(summary_rows)

        with failures_csv.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["site", "task", "subject", "stage", "status", "reason", "log"])
            for row in rows:
                for stage in STAGES:
                    status = getattr(row, f"{stage}_status")
                    reason = getattr(row, f"{stage}_reason")
                    log = getattr(row, f"{stage}_log")
                    if status != "completed":
                        writer.writerow([row.site, row.task, row.subject, stage, status, reason, log])

    def summary(self, rows: Sequence[InventoryRow]) -> List[Dict[str, object]]:
        out: List[Dict[str, object]] = []
        for site in self.sites:
            for task in self.tasks:
                selected = [r for r in rows if r.site == site and r.task == task]
                if not selected:
                    continue
                item: Dict[str, object] = {"site": site, "task": task, "subjects": len(selected)}
                for stage in STAGES:
                    item[f"{stage}_completed"] = sum(1 for r in selected if getattr(r, f"{stage}_status") == "completed")
                    item[f"{stage}_blocked"] = sum(1 for r in selected if getattr(r, f"{stage}_status") == "blocked")
                    item[f"{stage}_pending_or_failed"] = sum(1 for r in selected if getattr(r, f"{stage}_status") == "failed_or_pending")
                out.append(item)
        return out

    def print_summary(self, rows: Sequence[InventoryRow]) -> None:
        header = [
            "site", "task", "subjects",
            "init", "pyll", "sync", "seg", "ar",
            "next_init", "next_pyll", "next_sync", "next_seg", "next_ar",
        ]
        data = []
        for item in self.summary(rows):
            site = str(item["site"])
            task = str(item["task"])
            selected = [r for r in rows if r.site == site and r.task == task]
            data.append([
                site,
                task,
                str(item["subjects"]),
                str(item["init_completed"]),
                str(item["pylossless_completed"]),
                str(item["sync_loss_completed"]),
                str(item["segment_completed"]),
                str(item["autoreject_completed"]),
                str(sum(1 for r in selected if r.next_stage == "init")),
                str(sum(1 for r in selected if r.next_stage == "pylossless")),
                str(sum(1 for r in selected if r.next_stage == "sync_loss")),
                str(sum(1 for r in selected if r.next_stage == "segment")),
                str(sum(1 for r in selected if r.next_stage == "autoreject")),
            ])
        print_table(header, data)

    def write_pending_lists(self, rows: Sequence[InventoryRow], stage: Optional[str] = None) -> None:
        list_dir = self.out_dir / "pending_lists"
        list_dir.mkdir(parents=True, exist_ok=True)
        stages = [stage] if stage else list(STAGES)
        for st in stages:
            for site in self.sites:
                for task in self.tasks:
                    selected = [
                        r.subject
                        for r in rows
                        if r.site == site
                        and r.task == task
                        and r.next_stage == st
                        and not r.next_action.startswith("blocked:")
                    ]
                    selected = sorted(set(selected))
                    path = list_dir / f"{site}_{task}_{st}_ready.txt"
                    path.write_text("\n".join(selected) + ("\n" if selected else ""))

    def slurm_script_path(self, stage: str) -> Path:
        return self.wd / "slurm" / f"q1k_manager_{stage}.slurm"

    def write_slurm_scripts(self) -> None:
        slurm_dir = self.wd / "slurm"
        slurm_dir.mkdir(parents=True, exist_ok=True)
        self.slurm_output.mkdir(parents=True, exist_ok=True)

        for stage in STAGES:
            time_s, mem_s, cpus = STAGE_RESOURCES[stage]
            cli = STAGE_CLI[stage]
            label = STAGE_LABEL[stage]
            script = self.slurm_script_path(stage)
            site_arg = ' --site "$SITE"' if stage == "init" else ""
            content = f"""#!/bin/bash
#SBATCH --account=def-emayada
#SBATCH --time={time_s}
#SBATCH --mem={mem_s}
#SBATCH --cpus-per-task={cpus}
#SBATCH --job-name=Q1K_{label}
#SBATCH --output={self.slurm_output}/%x_%A_%a.out

set -u

cd {self.wd}
source {self.venv_activate}

SUBJECT=$(sed -n "${{SLURM_ARRAY_TASK_ID}}p" "$LIST")

echo "========================================"
echo "Q1K MANAGER JOB"
echo "STAGE: {stage}"
echo "SITE: $SITE"
echo "TASK: $TASK"
echo "SUBJECT: $SUBJECT"
echo "ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
echo "LIST: $LIST"
echo "HOST: $(hostname)"
echo "START: $(date)"
echo "========================================"

{cli} --project-path {self.wd} --task "$TASK" --subject "$SUBJECT"{site_arg}
status=$?

echo "========================================"
echo "END: $(date)"
echo "STATUS: $status"
echo "STAGE: {stage}"
echo "TASK: $TASK"
echo "SUBJECT: $SUBJECT"
echo "========================================"

exit $status
"""
            script.write_text(content)
            script.chmod(0o755)

    def submit(self, rows: Sequence[InventoryRow], stage: str, site: str, task: str, max_jobs: int, dry_run: bool = False) -> None:
        self.write_pending_lists(rows, stage)
        self.write_slurm_scripts()
        list_path = self.out_dir / "pending_lists" / f"{site}_{task}_{stage}_ready.txt"
        if not list_path.exists() or not list_path.read_text().strip():
            print(f"No ready subjects for {site} {task} {stage}")
            return
        n = len([line for line in list_path.read_text().splitlines() if line.strip()])
        script = self.slurm_script_path(stage)
        job_name = f"{site}_{task}_{STAGE_LABEL[stage]}"
        cmd = [
            "sbatch",
            f"--job-name={job_name}",
            f"--array=1-{n}%{max_jobs}",
            f"--export=ALL,SITE={site},TASK={task},LIST={list_path}",
            str(script),
        ]
        print(" ".join(cmd))
        if dry_run:
            return
        subprocess.run(cmd, check=True)

    def recent_errors(self, limit: int = 80) -> List[Tuple[str, str]]:
        self.load_logs()
        found: List[Tuple[str, str]] = []
        for lf in self.log_files[:limit]:
            text = self._read_log(lf)
            lines = [line.strip() for line in text.splitlines() if is_error_text(line)]
            if lines:
                found.append((str(lf), lines[0][:200]))
        return found


def print_table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> None:
    widths = [len(h) for h in header]
    for row in rows:
        for i, value in enumerate(row):
            widths[i] = max(widths[i], len(str(value)))
    fmt = "  ".join("{:<" + str(w) + "}" for w in widths)
    print(fmt.format(*header))
    print(fmt.format(*["-" * w for w in widths]))
    for row in rows:
        print(fmt.format(*row))


def load_or_scan(manager: Q1KManager, force_scan: bool = False) -> List[InventoryRow]:
    csv_path = manager.out_dir / "q1k_master_inventory.csv"
    if force_scan or not csv_path.exists():
        rows = manager.scan()
        manager.write_outputs(rows)
        manager.write_pending_lists(rows)
        manager.write_slurm_scripts()
        return rows

    rows: List[InventoryRow] = []
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for item in reader:
            converted = {}
            int_fields = {
                "source_mff_count",
                "source_et_count",
                "init_eeg_count",
                "init_et_count",
                "pylossless_eeg_count",
                "sync_eeg_count",
                "segment_epo_count",
                "autoreject_epo_count",
            }
            for field in InventoryRow.__dataclass_fields__:
                value = item.get(field, "")
                if field in int_fields:
                    converted[field] = int(value or 0)
                else:
                    converted[field] = value
            rows.append(InventoryRow(**converted))
    return rows


def run_dashboard(manager: Q1KManager) -> None:
    """Small refreshable text dashboard; safer than a complex curses TUI."""
    while True:
        rows = load_or_scan(manager, force_scan=True)
        os.system("clear")
        print("Q1K PIPELINE MANAGER")
        print(f"WD: {manager.wd}")
        print(f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        manager.print_summary(rows)
        print()
        print("Commands: [r] refresh  [e] recent errors  [l] pending lists  [q] quit")
        choice = input("> ").strip().lower()
        if choice == "q":
            break
        if choice == "e":
            print()
            errors = manager.recent_errors()
            if not errors:
                print("No recent error lines found.")
            for path, line in errors[:40]:
                print(f"{path}: {line}")
            input("\nPress Enter to continue...")
        elif choice == "l":
            manager.write_pending_lists(rows)
            print(f"Pending lists written to: {manager.out_dir / 'pending_lists'}")
            input("\nPress Enter to continue...")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Q1K pipeline manager")
    parser.add_argument("command", choices=("scan", "summary", "lists", "submit", "errors", "dashboard"))
    parser.add_argument("--wd", default=str(DEFAULT_WD), help="Q1K working directory")
    parser.add_argument("--out-dir", default=None, help="Output directory for manager CSVs")
    parser.add_argument("--sites", default=",".join(DEFAULT_SITES), help="Comma-separated sites")
    parser.add_argument("--tasks", default=",".join(DEFAULT_TASKS), help="Comma-separated tasks")
    parser.add_argument("--stage", choices=STAGES, default=None, help="Pipeline stage")
    parser.add_argument("--site", choices=DEFAULT_SITES, default=None, help="Site for submit")
    parser.add_argument("--task", choices=DEFAULT_TASKS, default=None, help="Task for submit")
    parser.add_argument("--max-jobs", type=int, default=3, help="Slurm array concurrency")
    parser.add_argument("--dry-run", action="store_true", help="Print sbatch command without submitting")
    parser.add_argument("--force", action="store_true", help="Force fresh scan")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    wd = Path(args.wd).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else None
    sites = tuple(s.strip() for s in args.sites.split(",") if s.strip())
    tasks = tuple(t.strip().upper() for t in args.tasks.split(",") if t.strip())
    manager = Q1KManager(wd=wd, sites=sites, tasks=tasks, out_dir=out_dir)

    if args.command == "dashboard":
        run_dashboard(manager)
        return 0

    rows = load_or_scan(manager, force_scan=args.force or args.command == "scan")

    if args.command == "scan":
        print(f"Wrote inventory to: {manager.out_dir}")
        manager.print_summary(rows)
        return 0

    if args.command == "summary":
        manager.print_summary(rows)
        return 0

    if args.command == "lists":
        manager.write_pending_lists(rows, args.stage)
        print(f"Pending lists written to: {manager.out_dir / 'pending_lists'}")
        return 0

    if args.command == "errors":
        errors = manager.recent_errors()
        if not errors:
            print("No recent error lines found.")
        for path, line in errors:
            print(f"{path}: {line}")
        return 0

    if args.command == "submit":
        if not (args.stage and args.site and args.task):
            print("submit requires --stage --site --task", file=sys.stderr)
            return 2
        manager.submit(rows, args.stage, args.site, args.task, args.max_jobs, args.dry_run)
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
