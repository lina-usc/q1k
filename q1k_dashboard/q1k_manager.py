# Q1K EEG/ET Pipeline Manager v5 — Narval TUI
# python3 q1k_manager.py
#Last updated March 20th, 2026, Yayyyy !! it works now
# Fixes in v5 (over v4):
#   1. Overview/Tasks progress bars now count INIT+PYLS+SYNC+SEGM+AUTO
#      correctly — a subject at SYNC stage counts as INIT+PYLS done,
#      not as 0% done.
#   2. Participants tab INIT column now shows V when BIDS .edf exists.
#   3. Pipeline tab "done" counts reflect actual filesystem state per stage.
#   4. Auto-refresh every 5 minutes (configurable via AUTO_SCAN_INTERVAL).
#   5. Submitted job IDs shown in Pipeline drill-down INFO column.
#   6. Terminal assist: "init MHC" label fixed (was "MNI").
"""
Tabs: 1=Overview 2=Tasks 3=Pipeline 4=Participants 5=Jobs 6=Terminal 7=Debug 8=Help
Keys: S=scan  Q=quit  1-8=tabs
"""
import curses
import json
import os
import queue
import re
import subprocess
import threading
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
WD           = Path("/lustre07/scratch/rsweety/white_paper/wd")
SOURCEDATA   = WD / "sourcedata"
Q1K_NEW      = WD / "q1k_new"
VENV         = WD / "q1k_venv_scratch" / "bin" / "activate"
SLURM_OUT    = WD / "slurm_output"
DERIVATIVES  = WD / "derivatives"
PYLL_DERIV   = DERIVATIVES / "pylossless" / "derivatives" / "pylossless"
SYNC_DERIV   = DERIVATIVES / "pylossless" / "derivatives" / "sync_loss"
SEG_DERIV    = SYNC_DERIV  / "derivatives" / "segment"   / "epoch_fif_files"
AUTO_DERIV   = SYNC_DERIV  / "derivatives" / "autorej"   / "epoch_fif_files"
INVENTORY    = WD / "participant_inventory.json"
PROJECT_PATH = str(WD) + "/"

HOSPITALS   = ["HSJ", "MHC"]
SITE_FLAG   = {"HSJ": "HSJ", "MHC": "MHC"}
STAGES      = ["INIT","PYLOSSLESS","SYNC_LOSS","SEGMENTATION","AUTOREJECT"]
STAGE_CLI   = {"INIT":"q1k-init","PYLOSSLESS":"q1k-pylossless",
                "SYNC_LOSS":"q1k-sync-loss","SEGMENTATION":"q1k-segment",
                "AUTOREJECT":"q1k-autorej"}
STAGE_SHORT = {"INIT":"INIT","PYLOSSLESS":"PYLS","SYNC_LOSS":"SYNC",
                "SEGMENTATION":"SEGM","AUTOREJECT":"AUTO"}
VALID_TASKS = {"RS","TO","GO","RSRio","NSP","VS","AEP","PLR","VEP","MMN"}

# Tasks that require .asc ET file at SYNC_LOSS stage
ET_TASKS = {"VEP","GO","NSP","PLR","VS"}

# Auto-scan interval in seconds (approx 5 minutes or so, but I might increase it )
AUTO_SCAN_INTERVAL = 300

# Valid subject ID pattern
VALID_PID = re.compile(r'^\d{4,6}[A-Z0-9]\d?$')

# ── Data verification ──────────────────────────────────────────────────────
def _asc_exists(pid, hosp, task):
    if task not in ET_TASKS:
        return True
    et_dir = SOURCEDATA / hosp / "et"
    if not et_dir.exists():
        return False
    for sub_dir in et_dir.iterdir():
        if not sub_dir.is_dir(): continue
        raw_pid, _ = sourcedata_to_bids(sub_dir.name)
        if raw_pid != pid: continue
        for f in sub_dir.rglob("*.asc"):
            return True
    return False

def verify_data(p, task, stage):
    pid  = p["id"]
    hosp = p["hospital"]

    if stage == "INIT":
        eeg_dir = SOURCEDATA / hosp / "eeg"
        if eeg_dir.exists():
            for raw_dir in eeg_dir.iterdir():
                rpid, _ = sourcedata_to_bids(raw_dir.name)
                if rpid == pid:
                    mffs = list(raw_dir.glob(f"*{task}*.mff"))
                    if mffs: return True, "EEG .mff found"
        return False, f"No {task} .mff in sourcedata/{hosp}/eeg/{pid}"

    if stage == "PYLOSSLESS":
        be = WD / f"sub-{pid}" / "ses-01" / "eeg"
        if be.exists() and any(be.glob(f"*_task-{task}_*_eeg.edf")):
            return True, "BIDS .edf found"
        return False, "No BIDS .edf — run INIT first"

    if stage == "SYNC_LOSS":
        pe = PYLL_DERIV / f"sub-{pid}" / "ses-01" / "eeg"
        if not (pe.exists() and any(pe.glob(f"*_task-{task}_*_ll_config.yaml"))):
            return False, "No pylossless output — run PYLOSSLESS first"
        if task in ET_TASKS and not _asc_exists(pid, hosp, task):
            return False, f"No .asc ET file in sourcedata/{hosp}/et/{pid}"
        return True, "pylossless done" + (" + .asc found" if task in ET_TASKS else "")

    if stage == "SEGMENTATION":
        se = SYNC_DERIV / f"sub-{pid}" / "ses-01" / "eeg"
        if se.exists() and any(se.glob(f"*_task-{task}_*")):
            return True, "sync_loss output found"
        return False, "No sync_loss output — run SYNC_LOSS first"

    if stage == "AUTOREJECT":
        sd = SEG_DERIV / task
        if sd.exists() and any(sd.glob(f"sub-{pid}_*_task-{task}_*_epo.fif")):
            return True, "segmentation output found"
        return False, "No segmentation output — run SEGMENTATION first"

    return True, "ok"

# ── Stage detection ────────────────────────────────────────────────────────
def detect_stage(pid, task):
    """
    FIX v5: 
    Previously PYLOSSLESS/SYNC/etc were only checked if the previous
    stage was 'done', causing subjects that skipped INIT (already had
    BIDS files from earlier runs) to show all stages as pending.
    Now each stage is checked directly so the UI reflects reality.
    """
    r = {s: "pending" for s in STAGES}

    # INIT — BIDS .edf exists
    be = WD / f"sub-{pid}" / "ses-01" / "eeg"
    if be.exists() and any(be.glob(f"*_task-{task}_*_eeg.edf")):
        r["INIT"] = "done"

    # PYLOSSLESS — ll_config.yaml exists (independent check)
    pe = PYLL_DERIV / f"sub-{pid}" / "ses-01" / "eeg"
    if pe.exists() and any(pe.glob(f"*_task-{task}_*_ll_config.yaml")):
        r["PYLOSSLESS"] = "done"
        # If PYLS done but INIT not detected, mark INIT done too
        # (BIDS files may have been written to a different path earlier)
        if r["INIT"] == "pending":
            r["INIT"] = "done"

    # SYNC_LOSS — independent check
    se = SYNC_DERIV / f"sub-{pid}" / "ses-01" / "eeg"
    if se.exists() and any(se.glob(f"*_task-{task}_*")):
        r["SYNC_LOSS"] = "done"
        if r["PYLOSSLESS"] == "pending":
            r["PYLOSSLESS"] = "done"
        if r["INIT"] == "pending":
            r["INIT"] = "done"

    # SEGMENTATION — independent check
    sd = SEG_DERIV / task
    if sd.exists() and any(sd.glob(f"sub-{pid}_*_task-{task}_*_epo.fif")):
        r["SEGMENTATION"] = "done"

    # AUTOREJECT — independent check
    ad = AUTO_DERIV / task
    if ad.exists() and any(ad.glob(f"sub-{pid}_*_task-{task}_*_epo.fif")):
        r["AUTOREJECT"] = "done"

    return r

def sourcedata_to_bids(raw_name):
    parts = raw_name.split("_")
    if len(parts) >= 4 and parts[0] == "Q1K" and parts[1] in HOSPITALS:
        return parts[2] + parts[3], parts[1]
    return raw_name, ("MHC" if raw_name.startswith("2") else "HSJ")

def _valid_pid(pid):
    if not pid or len(pid) < 4: return False
    if "-" in pid: return False
    if pid in ("Pilots", "archive", "GO_raw", "__MACOSX", "Q043_F1"): return False
    return bool(VALID_PID.match(pid))

def _extract_task_from_raw(name):
    for part in name.replace(".mff","").replace(".edf","").split("_"):
        if part in VALID_TASKS: return part
    return None

def scan_participants(progress_cb=None):
    pmap = {}
    for hosp in HOSPITALS:
        for mod in ["eeg","et"]:
            d = SOURCEDATA / hosp / mod
            if not d.exists(): continue
            for rd in sorted(d.iterdir()):
                if not rd.is_dir() or rd.name.startswith("."): continue
                if rd.name in ("Pilots","archive","GO_raw","__MACOSX"): continue
                pid, h = sourcedata_to_bids(rd.name)
                if not _valid_pid(pid): continue
                if pid not in pmap:
                    pmap[pid] = {"id":pid,"hospital":h,"has_eeg":False,"has_et":False,
                                 "raw_name":rd.name,"eeg_tasks":[],"task_stages":{},
                                 "slurm_jobs":{},"notes":"","et_asc_tasks":[]}
                p = pmap[pid]
                if mod == "eeg":
                    p["has_eeg"] = True
                    tf = set(p["eeg_tasks"])
                    for entry in rd.iterdir():
                        t = _extract_task_from_raw(entry.name)
                        if t: tf.add(t)
                    p["eeg_tasks"] = sorted(tf)
                else:
                    p["has_et"] = True
                    asc_tasks = set()
                    for f in rd.rglob("*.asc"):
                        for t in ET_TASKS:
                            if t.upper() in f.name.upper():
                                asc_tasks.add(t)
                        if not asc_tasks:
                            asc_tasks.update(ET_TASKS)
                    p["et_asc_tasks"] = sorted(asc_tasks)

    raw_name_map = {v["id"]: v["raw_name"] for v in pmap.values()}
    for bs in sorted(WD.glob("sub-*/")):
        pid = bs.name[4:]
        if not _valid_pid(pid): continue
        ed = bs / "ses-01" / "eeg"
        if not ed.exists(): continue
        hosp = "MHC" if pid.startswith("2") else "HSJ"
        raw = raw_name_map.get(pid, pid)
        if raw == pid:
            for hosp2 in HOSPITALS:
                eeg_d = SOURCEDATA / hosp2 / "eeg"
                if not eeg_d.exists(): continue
                for rd in eeg_d.iterdir():
                    rpid, _ = sourcedata_to_bids(rd.name)
                    if rpid == pid:
                        raw = rd.name
                        hosp = hosp2
                        break
        if pid not in pmap:
            pmap[pid] = {"id":pid,"hospital":hosp,"has_eeg":True,"has_et":False,
                         "raw_name":raw,"eeg_tasks":[],"task_stages":{},"slurm_jobs":{},
                         "notes":"","et_asc_tasks":[]}
        p = pmap[pid]; p["has_eeg"] = True
        tf = set(p["eeg_tasks"])
        for edf in ed.glob("*_eeg.edf"):
            m = re.search(r"_task-(\w+)_", edf.name)
            if m and m.group(1) in VALID_TASKS: tf.add(m.group(1))
        p["eeg_tasks"] = sorted(tf)

    participants = []
    total = len(pmap)
    for i, (pid, p) in enumerate(sorted(pmap.items())):
        if progress_cb: progress_cb(i, total, pid)
        p["task_stages"] = {t: detect_stage(pid, t) for t in p["eeg_tasks"]}
        participants.append(p)
    return participants

def load_inv():
    if INVENTORY.exists():
        try: return json.loads(INVENTORY.read_text())
        except: pass
    return []

def save_inv(ps):
    INVENTORY.parent.mkdir(parents=True, exist_ok=True)
    INVENTORY.write_text(json.dumps(ps, indent=2))

def stage_counts(participants, task):
    out = {s: {h: defaultdict(int) for h in HOSPITALS+["ALL"]} for s in STAGES}
    for p in participants:
        if task not in p.get("eeg_tasks",[]): continue
        h  = p["hospital"]
        ts = p.get("task_stages",{}).get(task,{})
        for s in STAGES:
            st = ts.get(s,"pending")
            out[s][h][st]     += 1
            out[s]["ALL"][st] += 1
    return out

def run_cmd(cmd, timeout=60, cwd=None):
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                           timeout=timeout, cwd=str(cwd or WD))
        return r.stdout, r.stderr, r.returncode
    except subprocess.TimeoutExpired: return "", "Timeout", 1
    except Exception as e:            return "", str(e), 1

def run_cmd_stream(cmd, out_queue, cwd=None):
    def _run():
        try:
            proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True,
                                    cwd=str(cwd or WD))
            for line in proc.stdout:
                out_queue.put(("out", line.rstrip()))
            proc.wait()
            out_queue.put(("done", str(proc.returncode)))
        except Exception as e:
            out_queue.put(("err", str(e)))
            out_queue.put(("done", "1"))
    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return t

def venv_cmd(cli_cmd):
    return f'bash -c "source {VENV} && cd {Q1K_NEW} && {cli_cmd}"'

def make_slurm_script(task, stage, list_path, site, n, job_name):
    cli  = STAGE_CLI[stage]
    sf   = f"--site {SITE_FLAG.get(site,site)}" if stage=="INIT" else ""
    slf  = "--slurm" if stage in ("PYLOSSLESS","AUTOREJECT") else ""
    return (f"#!/bin/bash\n#SBATCH --account=def-emayada\n#SBATCH --time=48:00:00\n"
            f"#SBATCH --mem=16G\n#SBATCH --cpus-per-task=2\n#SBATCH --job-name={job_name}\n"
            f"#SBATCH --output={SLURM_OUT}/{job_name}_%j_%a.out\n"
            f"#SBATCH --array=1-{n}%10\n\nsource {VENV}\ncd {Q1K_NEW}\n\n"
            f"SUBJECT=$(sed -n \"${{SLURM_ARRAY_TASK_ID}}p\" {list_path})\n"
            f"echo \"Processing: $SUBJECT\"\n"
            f"{cli} --project-path {PROJECT_PATH} --task {task} --subject $SUBJECT {sf} {slf}\n")

# ── Colors ─────────────────────────────────────────────────────────────────
def init_colors():
    curses.start_color(); curses.use_default_colors()
    curses.init_pair(1, curses.COLOR_CYAN,    -1)
    curses.init_pair(2, curses.COLOR_GREEN,   -1)
    curses.init_pair(3, curses.COLOR_YELLOW,  -1)
    curses.init_pair(4, curses.COLOR_RED,     -1)
    curses.init_pair(5, curses.COLOR_MAGENTA, -1)
    curses.init_pair(6, curses.COLOR_WHITE,   -1)
    curses.init_pair(7, curses.COLOR_BLACK,   curses.COLOR_CYAN)
    curses.init_pair(8, curses.COLOR_BLACK,   curses.COLOR_WHITE)
    curses.init_pair(9, curses.COLOR_CYAN,    curses.COLOR_BLACK)
    curses.init_pair(10,curses.COLOR_BLACK,   curses.COLOR_GREEN)
    curses.init_pair(11,curses.COLOR_BLACK,   curses.COLOR_RED)
    curses.init_pair(12,curses.COLOR_BLACK,   curses.COLOR_YELLOW)

AC  = lambda: curses.color_pair(1)|curses.A_BOLD
GR  = lambda: curses.color_pair(2)
YL  = lambda: curses.color_pair(3)
RD  = lambda: curses.color_pair(4)
MG  = lambda: curses.color_pair(5)
WH  = lambda: curses.color_pair(6)
SL  = lambda: curses.color_pair(7)
HD  = lambda: curses.color_pair(8)|curses.A_BOLD
TB  = lambda: curses.color_pair(9)|curses.A_BOLD
OKB = lambda: curses.color_pair(10)|curses.A_BOLD
ERB = lambda: curses.color_pair(11)|curses.A_BOLD
WRN = lambda: curses.color_pair(12)|curses.A_BOLD

SCOL = {"done":GR,"running":YL,"error":RD,"queued":MG,"pending":WH}
SICO = {"done":"V","running":"*","error":"X","queued":"Q","pending":"."}

# ═══════════════════════════════════════════════════════════════════════════
class App:
    TABS = ["1:Overview","2:Tasks","3:Pipeline","4:Participants",
            "5:Jobs","6:Terminal","7:Debug","8:Help"]

    def __init__(self, scr):
        self.scr = scr; self.running = True; self.tab = 0
        self.msg = "S=scan  1-8=tabs  Q=quit"; self.msg_ok = True
        self.ps = []; self.tcounts = {}; self.task_list = []; self.sel_task = None
        self.tcur = 0
        self.pipe_stage = 0; self.pipe_hosp = "ALL"
        self.pipe_drill = False
        self.pipe_dcur  = 0; self.pipe_doff = 0; self.pipe_dlist = []
        self.pcur = 0; self.poff = 0; self.filt_ps = []
        self.fh = "ALL"; self.fss = "ALL"; self.fsearch = ""
        self.det_p = None; self.det_sc = 0
        self.selected_pids = set()
        self.jobs_list   = []; self.jobs_cur = 0
        self.jobs_log    = []; self.jobs_log_sc = 0
        self.jobs_delta  = []; self.jobs_errors = []
        self.jobs_panel  = "list"
        self.cwd = WD; self.cin = ""; self.chist = []; self.chidx = 0
        self.cout = []; self._acmds = []
        self._bg_queue  = queue.Queue()
        self._bg_thread = None
        self._bg_cmd    = ""
        self._bg_done   = True
        self.dbg_lines=[]; self.dbg_sc=0; self.dbg_jobs=[]; self.dbg_cur=0
        # FIX v5: auto-scan timer
        self._last_scan_time = 0.0
        curses.curs_set(0); scr.keypad(True); init_colors()
        scr.timeout(200)
        self.ps = load_inv()
        if self.ps:
            self._rebuild()
            self._status(f"Loaded {len(self.ps)} participants — press S to rescan",True)
        else:
            self._status("No inventory — press S to scan",False)

    # ── helpers ──────────────────────────────────────────────────────────
    def _status(self,msg,ok=True): self.msg=str(msg)[:200]; self.msg_ok=ok

    def _rebuild(self):
        """
        FIX v5: Overview progress counts subjects that have completed
        each individual stage, not just fully-done subjects.
        DONE in overview = reached at least INIT stage.
        The progress bar shows INIT done / total for each task.
        """
        self.tcounts = {}
        for p in self.ps:
            for t in p.get("eeg_tasks",[]):
                if t not in VALID_TASKS: continue
                if t not in self.tcounts: self.tcounts[t] = defaultdict(int)
                self.tcounts[t]["total"] += 1
                ts = p.get("task_stages",{}).get(t,{})
                # Count per-stage completions
                for s in STAGES:
                    if ts.get(s) == "done":
                        self.tcounts[t][f"done_{s}"] += 1
                # Overall status for color coding
                sv = [ts.get(s,"pending") for s in STAGES]
                if   all(x=="done"    for x in sv): ov="done"
                elif any(x=="error"   for x in sv): ov="error"
                elif any(x=="done"    for x in sv): ov="partial"
                else:                               ov="pending"
                self.tcounts[t][ov] += 1
        self.task_list = sorted([t for t in self.tcounts if t in VALID_TASKS],
                                key=lambda t:-self.tcounts[t]["total"])
        if self.sel_task not in self.task_list:
            self.sel_task = self.task_list[0] if self.task_list else None
        self._filt()

    def _filt(self):
        if not self.sel_task: self.filt_ps=[]; return
        out=[]
        for p in self.ps:
            if self.sel_task not in p.get("eeg_tasks",[]): continue
            if self.fh!="ALL" and p["hospital"]!=self.fh: continue
            ts = p.get("task_stages",{}).get(self.sel_task,{})
            sv = [ts.get(s,"pending") for s in STAGES]
            if self.fss=="DONE"    and not all(x=="done"  for x in sv): continue
            if self.fss=="ERROR"   and "error"   not in sv:             continue
            if self.fss=="PENDING" and any(x!="pending"   for x in sv): continue
            if self.fss=="PARTIAL" and not (any(x=="done" for x in sv)
                                   and any(x=="pending"   for x in sv)): continue
            if self.fsearch and self.fsearch.lower() not in p["id"].lower(): continue
            out.append(p)
        self.filt_ps=out
        if self.pcur>=len(self.filt_ps): self.pcur=max(0,len(self.filt_ps)-1)

    def _w(self,y,x,txt,attr=0):
        h,w=self.scr.getmaxyx()
        if y<0 or y>=h or x<0: return
        txt=str(txt)[:max(0,w-x-1)]
        if not txt: return
        try: self.scr.addstr(y,x,txt,attr)
        except curses.error: pass

    def _hl(self,y,ch="─",attr=None):
        _,w=self.scr.getmaxyx(); self._w(y,0,ch*(w-1),attr or AC())

    def _bar(self,y,x,bw,done,total,sel=False):
        if total==0: self._w(y,x,"░"*bw,WH()); return
        fil=int(done/total*bw)
        self._w(y,x,"█"*fil,          GR() if not sel else SL())
        self._w(y,x+fil,"░"*(bw-fil), WH() if not sel else SL())
        self._w(y,x+bw, f" {done}/{total} {done/total*100:.0f}%", AC() if not sel else SL())

    def _next_pending(self,p,task):
        for s in STAGES:
            if p.get("task_stages",{}).get(task,{}).get(s,"pending")=="pending": return s
        return None

    def _poll_bg(self):
        changed = False
        try:
            while True:
                kind, data = self._bg_queue.get_nowait()
                if kind == "done":
                    code = data
                    self.cout.append((f"  [exit {code}]", "ok" if code=="0" else "err"))
                    self.cout.append(("","out"))
                    self._bg_done = True
                    if code != "0":
                        self._status(f"Command FAILED (exit {code}) — see Terminal",False)
                    else:
                        self._status(f"Done: {self._bg_cmd[:50]}",True)
                elif kind == "err":
                    self.cout.append((f"  ERR: {data}","err"))
                else:
                    self.cout.append((f"  {data}","out"))
                if len(self.cout)>600: self.cout=self.cout[-400:]
                changed = True
        except queue.Empty:
            pass
        return changed

    def _check_auto_scan(self):
        """FIX v5: Auto-scan every AUTO_SCAN_INTERVAL seconds."""
        now = time.time()
        if now - self._last_scan_time >= AUTO_SCAN_INTERVAL and self._bg_done:
            self._last_scan_time = now
            self._do_scan(silent=True)

    # ── draw router ──────────────────────────────────────────────────────
    def draw(self):
        self.scr.erase()
        h,w=self.scr.getmaxyx()
        if h<22 or w<80:
            self._w(0,0,f"Too small ({w}x{h}) — need 80x22+"); self.scr.refresh(); return
        self._draw_hdr(h,w); self._draw_tabs(w); self._draw_status(h,w)
        cy,ch=3,h-4
        fns=[self._draw_overview,self._draw_tasks,self._draw_pipeline,
             self._draw_participants,self._draw_jobs,self._draw_terminal,
             self._draw_debug,self._draw_help]
        fns[self.tab](cy,ch,w)
        self.scr.refresh()

    def _draw_hdr(self,h,w):
        n=len(self.ps); hsj=sum(1 for p in self.ps if p["hospital"]=="HSJ")
        now=datetime.now().strftime("%H:%M %d %b")
        task=f" [{self.sel_task}]" if self.sel_task else ""
        sel_s=f" SEL:{len(self.selected_pids)}" if self.selected_pids else ""
        bg_s=" [BG RUNNING]" if not self._bg_done else ""
        # Show time until next auto-scan
        next_scan = max(0, int(AUTO_SCAN_INTERVAL - (time.time() - self._last_scan_time)))
        scan_s = f" [scan in {next_scan}s]" if self._last_scan_time > 0 else ""
        self.scr.attron(HD())
        self._w(0,0," "*(w-1))
        self._w(0,0,f" Q1K v5{task}{sel_s}{bg_s}{scan_s}")
        self._w(0,max(0,w-38),f" {n} ptcps | HSJ:{hsj} MHC:{n-hsj} | {now} ")
        self.scr.attroff(HD())

    def _draw_tabs(self,w):
        self._hl(1); x=1
        for i,name in enumerate(self.TABS):
            lbl=f" {name} "
            self._w(1,x,lbl,TB() if i==self.tab else AC())
            x+=len(lbl)+1
        self._w(1,w-18," S=scan  Q=quit ",YL())

    def _draw_status(self,h,w):
        attr=GR() if self.msg_ok else RD()
        try:
            self.scr.attron(attr); self._w(h-1,0," "*(w-1)); self._w(h-1,1,self.msg)
            self.scr.attroff(attr)
        except: pass

    # ── 1 Overview ────────────────────────────────────────────────────────
    def _draw_overview(self,y,h,w):
        if not self.ps: self._w(y+2,4,"No data — press S to scan",YL()); return
        n=len(self.ps); hsj=sum(1 for p in self.ps if p["hospital"]=="HSJ")
        cards=[("TOTAL",n,WH()),("HSJ",hsj,AC()),("MHC",n-hsj,MG()),
               ("EEG",sum(1 for p in self.ps if p["has_eeg"]),GR()),
               ("ET", sum(1 for p in self.ps if p["has_et"]),YL())]
        cw=max(12,(w-2)//len(cards))
        self._w(y,1,"DATASET SUMMARY",AC())
        for i,(lbl,val,attr) in enumerate(cards):
            self._w(y+1,1+i*cw,str(val),attr|curses.A_BOLD)
            self._w(y+2,1+i*cw,lbl,WH())
        self._w(y+4,1,"TASK OVERVIEW — progress = INIT done  ↑↓=select  Enter=Pipeline",AC())
        bw=min(28,w-58)
        # FIX v5: show INIT done / total (most meaningful high-level metric)
        self._w(y+5,0,f"  {'TASK':<10}{'TOTAL':>6}{'INIT':>7}{'PYLS':>7}{'SYNC':>7}{'SEGM':>7}  PROGRESS (INIT)"[:w-1],HD())
        for ri,task in enumerate(self.task_list):
            if y+6+ri>=y+h-1: break
            c=self.tcounts.get(task,{}); tot=c["total"]
            init_done = c.get("done_INIT",0)
            pyls_done = c.get("done_PYLOSSLESS",0)
            sync_done = c.get("done_SYNC_LOSS",0)
            segm_done = c.get("done_SEGMENTATION",0)
            sel=task==self.sel_task; attr=SL() if sel else WH()
            line=f"  {task:<10}{tot:>6}{init_done:>7}{pyls_done:>7}{sync_done:>7}{segm_done:>7}  "
            self._w(y+6+ri,0,line,attr)
            self._bar(y+6+ri,len(line),bw,init_done,tot,sel)

    # ── 2 Tasks ───────────────────────────────────────────────────────────
    def _draw_tasks(self,y,h,w):
        self._w(y,1,f"SELECT TASK  Active:[{self.sel_task or 'none'}]  ↑↓=move  Enter=Pipeline",AC())
        bw=min(24,w-58)
        self._w(y+1,0,f"  {'#':<4}{'TASK':<12}{'TOTAL':>6}{'INIT':>7}{'PYLS':>7}{'SYNC':>7}{'SEGM':>7}  PROGRESS"[:w-1],HD())
        vis=h-4; off=max(0,self.tcur-vis//2)
        for i,task in enumerate(self.task_list[off:off+vis]):
            row=y+2+i; idx=off+i
            c=self.tcounts.get(task,{}); tot=c["total"]
            init_done = c.get("done_INIT",0)
            pyls_done = c.get("done_PYLOSSLESS",0)
            sync_done = c.get("done_SYNC_LOSS",0)
            segm_done = c.get("done_SEGMENTATION",0)
            sel=idx==self.tcur; attr=SL() if sel else WH()
            line=f"  {idx+1:<4}{task:<12}{tot:>6}{init_done:>7}{pyls_done:>7}{sync_done:>7}{segm_done:>7}  "
            self._w(row,0,line[:w-bw-8],attr)
            self._bar(row,len(line),bw,init_done,tot,sel)
            if sel: self._w(row,w-12,"<- ACTIVE",TB())

    # ── 3 Pipeline ────────────────────────────────────────────────────────
    def _draw_pipeline(self,y,h,w):
        task=self.sel_task
        if not task: self._w(y+2,4,"No task — press 2",YL()); return
        if self.pipe_drill:
            self._draw_pipeline_drill(y,h,w); return

        ps_t=[p for p in self.ps if task in p.get("eeg_tasks",[])]
        sc=stage_counts(ps_t,task)
        bw=min(18,w-80)

        self._w(y,1,f"PIPELINE [{task}]  H:[{self.pipe_hosp}]"
                    f"  ↑↓=stage  H=hosp  Enter=drill  T=test  Y=submit",AC())
        hdr=(f"  {'#':<3}{'STAGE':<14}"
             f"{'HSJ done':>9}{'HSJ pend':>9}{'HSJ err':>8}"
             f"{'MHC done':>9}{'MHC pend':>9}{'MHC err':>8}"
             f"  ALL PROGRESS")
        self._w(y+1,0,hdr[:w-1],HD())

        for si,stage in enumerate(STAGES):
            sel=si==self.pipe_stage; attr=SL() if sel else WH()
            r=y+2+si*2
            ch=sc[stage]["HSJ"]; cm=sc[stage]["MHC"]; ca=sc[stage]["ALL"]
            hd=ch.get("done",0); hp=ch.get("pending",0); he=ch.get("error",0)
            md=cm.get("done",0); mp=cm.get("pending",0); me=cm.get("error",0)
            tot=sum(ca.values()); done=ca.get("done",0)
            prev=STAGES[si-1] if si>0 else None
            ready=[p for p in ps_t
                   if p.get("task_stages",{}).get(task,{}).get(stage,"pending")=="pending"
                   and (prev is None or p.get("task_stages",{}).get(task,{}).get(prev)=="done")
                   and verify_data(p,task,stage)[0]]
            blocked=[p for p in ps_t
                     if p.get("task_stages",{}).get(task,{}).get(stage,"pending")=="pending"
                     and not verify_data(p,task,stage)[0]]
            line=(f"  {si+1:<3}{stage:<14}"
                  f"{hd:>9}{hp:>9}{he:>8}"
                  f"{md:>9}{mp:>9}{me:>8}  ")
            self._w(r,0,line[:w-bw-14],attr)
            self._bar(r,len(line),bw,done,tot,sel)
            ready_s=f"  {len(ready)} ready"
            blk_s  =f"  {len(blocked)} blocked" if blocked else ""
            self._w(r+1,6,f"  {STAGE_CLI[stage]}"[:w//3],WH())
            self._w(r+1,w//3+2,ready_s,GR() if ready else WH())
            if blocked: self._w(r+1,w//3+14,blk_s,RD())

        sy=y+2+len(STAGES)*2+1
        self._w(sy,1,"Enter=drill  T=test-1-subject  Y=submit-array  H=toggle-hosp  A=submit-all-stages",YL())

    def _draw_pipeline_drill(self,y,h,w):
        task=self.sel_task; stage=STAGES[self.pipe_stage]
        ps_t=[p for p in self.ps if task in p.get("eeg_tasks",[])]
        if self.pipe_hosp!="ALL":
            ps_t=[p for p in ps_t if p["hospital"]==self.pipe_hosp]

        prev=STAGES[STAGES.index(stage)-1] if STAGES.index(stage)>0 else None

        self.pipe_dlist=[]
        for p in ps_t:
            ts=p.get("task_stages",{}).get(task,{})
            st=ts.get(stage,"pending")
            ok,reason=verify_data(p,task,stage)
            prev_done=prev is None or ts.get(prev)=="done"
            cat=("done"   if st=="done" else
                 "error"  if st=="error" else
                 "ready"  if st=="pending" and prev_done and ok else
                 "blocked"if st=="pending" and (not prev_done or not ok) else
                 "queued")
            # FIX v5: show job ID in info if available
            job_info = p.get("slurm_jobs",{}).get(f"{task}:{stage}","")
            info = job_info if job_info else reason
            self.pipe_dlist.append((p,cat,info))

        counts={c:sum(1 for _,cat,_ in self.pipe_dlist if cat==c)
                for c in ("done","ready","blocked","error","queued")}

        self._w(y,1,f"DRILL [{task}/{stage}]  H:[{self.pipe_hosp}]  "
                    f"done:{counts['done']} ready:{counts['ready']} "
                    f"blocked:{counts['blocked']} err:{counts['error']}",AC())
        self._w(y,w-40," Spc=check  B=submit-ready  A=submit-all  ESC=back",YL())

        sab=[STAGE_SHORT[s] for s in STAGES]
        hdr=f" {'CK':<3}{'ID':<26}{'H':<5}{'CAT':<8}" + "".join(f"{s:^7}" for s in sab)+"  INFO"
        self._w(y+1,0,hdr[:w-1],HD())

        vis=h-5
        if self.pipe_dcur<self.pipe_doff: self.pipe_doff=self.pipe_dcur
        if self.pipe_dcur>=self.pipe_doff+vis: self.pipe_doff=self.pipe_dcur-vis+1

        for i,(p,cat,info) in enumerate(self.pipe_dlist[self.pipe_doff:self.pipe_doff+vis]):
            row=y+2+i; idx=self.pipe_doff+i; sel=idx==self.pipe_dcur
            chk="V" if p["id"] in self.selected_pids else " "
            ts=p.get("task_stages",{}).get(task,{})
            cat_at=(GR() if cat=="done" else OKB() if cat=="ready" else
                    RD() if cat in ("error","blocked") else MG())
            if sel: cat_at=SL()
            base=SL() if sel else WH()
            self._w(row,0," "*(w-1),base)
            self._w(row,0,f" [{chk}]{p['id']:<26}{p['hospital']:<5}",base)
            self._w(row,37,f"{cat:<8}",cat_at)
            x=45
            for s in STAGES:
                st=ts.get(s,"pending"); ic=SICO.get(st,"?")
                at=SL() if sel else SCOL.get(st,WH)()
                self._w(row,x,f" {ic:^5}",at); x+=7
            info_s=info[:w-x-2] if info else ""
            self._w(row,x+1,info_s,YL() if not sel else SL())

        self._w(y+h-2,0," ↑↓=scroll  Space=check  B=submit-ready  A=submit-all  T=test-1  ESC=back",YL())

    # ── 4 Participants ────────────────────────────────────────────────────
    def _draw_participants(self,y,h,w):
        if not self.sel_task: self._w(y+2,4,"No task — press 2",YL()); return
        if self.det_p: self._draw_detail(y,h,w); return
        fbar=(f" Task:[{self.sel_task}] H:[{self.fh}] St:[{self.fss}]"
              f" [{self.fsearch or '-'}] ({len(self.filt_ps)}) CHK:{len(self.selected_pids)}")
        self._w(y,0,fbar[:w-1],AC())
        self._w(y,w-34," H=hosp F=status /=search Spc=check",YL())
        sab=[STAGE_SHORT[s] for s in STAGES]
        hdr=f" {'CK':<3}{'ID':<26}{'H':<5}" + "".join(f"{s:^7}" for s in sab)+"  NEXT"
        self._w(y+1,0,hdr[:w-1],HD())
        vis=h-5
        if self.pcur<self.poff: self.poff=self.pcur
        if self.pcur>=self.poff+vis: self.poff=self.pcur-vis+1
        for i,p in enumerate(self.filt_ps[self.poff:self.poff+vis]):
            row=y+2+i; idx=self.poff+i; sel=idx==self.pcur
            chk="V" if p["id"] in self.selected_pids else " "
            ts=p.get("task_stages",{}).get(self.sel_task,{})
            nxt="DONE"
            for s in STAGES:
                st=ts.get(s,"pending")
                if st=="pending": nxt=STAGE_SHORT[s]; break
                elif st=="error": nxt=f"E:{STAGE_SHORT[s]}"; break
            base=SL() if sel else WH()
            self._w(row,0," "*(w-1),base)
            self._w(row,0,f" [{chk}]{p['id']:<26}{p['hospital']:<5}",base)
            x=41
            for s in STAGES:
                st=ts.get(s,"pending"); ic=SICO.get(st,"?")
                at=SL() if sel else SCOL.get(st,WH)()
                self._w(row,x,f" {ic:^5}",at); x+=7
            nc=RD() if "E:" in nxt else (GR() if nxt=="DONE" else YL())
            if sel: nc=SL()
            self._w(row,x+1,nxt,nc)
        if len(self.filt_ps)>vis>0:
            sb=int(self.poff/max(1,len(self.filt_ps)-vis)*(vis-1))
            for i in range(vis): self._w(y+2+i,w-1,"#" if i==sb else "|",AC())
        self._w(y+h-2,0," ↑↓ Spc=check Enter=detail R=next-stage T=test B=batch C=clear H=hosp F=filter /=search",YL())

    def _draw_detail(self,y,h,w):
        p=self.det_p; task=self.sel_task
        self._w(y,1,f"ESC=back  {p['id']}  ({p['hospital']})",AC())
        ts=p.get("task_stages",{}).get(task,{})
        be=WD/f"sub-{p['id']}"/("ses-01")/("eeg")
        tfiles=list(be.glob(f"*_task-{task}_*_eeg.edf")) if be.exists() else []
        et_ok=p.get("has_et",False)
        asc_tasks=p.get("et_asc_tasks",[])
        self._w(y+1,1,f"Task:{task}  BIDS-files:{len(tfiles)}  ET:{et_ok}  ASC-tasks:{asc_tasks}",WH())
        for fi,fn in enumerate(tfiles[:2]): self._w(y+2+fi,4,fn.name[:w-6],WH())
        self._w(y+4,1,f"STAGES for {task}:",AC())
        self._w(y+4,32,"Status    Verify           Job",HD())
        for si,stage in enumerate(STAGES):
            st=ts.get(stage,"pending"); ic=SICO.get(st,"?")
            at=SCOL.get(st,WH)(); sel=si==self.det_sc; bg=SL() if sel else at
            ok,reason=verify_data(p,task,stage)
            ver_s=(f"OK:{reason[:18]}" if ok else f"NO:{reason[:18]}")
            ver_at=GR() if ok else RD()
            if sel: ver_at=SL(); bg=SL()
            job=p.get("slurm_jobs",{}).get(f"{task}:{stage}","—")
            self._w(y+5+si,1,f" {ic} {stage:<14}{st.upper():<10}",bg)
            self._w(y+5+si,27,ver_s[:22],ver_at if not sel else SL())
            self._w(y+5+si,51,job[:18],bg)
            if sel:
                hint={"pending":"[Enter=submit T=test]","error":"[Enter=retry D=debug]",
                      "done":"done V","running":"running..."}.get(st,"")
                self._w(y+5+si,w-22,hint[:20],YL())
        ay=y+5+len(STAGES)+1
        self._w(ay,1,"R=next-stage  A=all  T=test  D=debug  L=logs  ↑↓=select  Enter=submit  ESC=back",YL())
        ox=w//2+2
        self._w(y+1,ox,"ALL TASKS:",AC())
        for ti,t in enumerate(p.get("eeg_tasks",[])):
            if y+2+ti>=y+h: break
            tts=p.get("task_stages",{}).get(t,{})
            sv=[tts.get(s,"pending") for s in STAGES]; dn=sum(1 for x in sv if x=="done")
            if all(x=="done" for x in sv): ta,ic2=GR(),"DONE"
            elif any(x=="error" for x in sv): ta,ic2=RD(),"ERR "
            else: ta,ic2=WH(),f"{dn}/{len(STAGES)}"
            self._w(y+2+ti,ox,f" {'>' if t==task else ' '}{t:<12}{ic2}",ta)

    # ── 5 Jobs ────────────────────────────────────────────────────────────
    def _draw_jobs(self,y,h,w):
        lw=38; rw=w-lw-2
        self._w(y,1,"JOBS  R=refresh  S=rescan+delta  F=errors  L=list  G=log  D=delta  E=errors",AC())
        self._hl(y+1)
        self._w(y+2,0,f"{'JOBID':<14}{'NAME':<16}{'STATE':<10}"[:lw],HD())
        vis_j=h-6
        for i,jd in enumerate(self.jobs_list[:vis_j]):
            sel=i==self.jobs_cur
            st=jd.get("state","?")
            at=(SL() if sel else GR() if st in ("COMPLETED","RUNNING") else
                RD() if st in ("FAILED","CANCELLED","TIMEOUT") else
                MG() if st=="PENDING" else WH())
            self._w(y+3+i,0,
                    f"{jd.get('jobid',''):<14}{jd.get('name','')[:15]:<16}{st:<10}"[:lw],at)
        panel_title={"list":"JOB DETAILS","log":"LOG OUTPUT",
                     "delta":"NEW COMPLETIONS SINCE LAST SCAN",
                     "errors":"ERROR SUMMARY"}
        self._w(y+2,lw+1,panel_title.get(self.jobs_panel,""),HD())
        lines=(self.jobs_log     if self.jobs_panel=="log"    else
               self.jobs_delta   if self.jobs_panel=="delta"  else
               [f"{pid} | {name} | {snip}" for pid,name,snip in self.jobs_errors]
               if self.jobs_panel=="errors" else self._job_detail_lines())
        sc=self.jobs_log_sc if self.jobs_panel=="log" else 0
        for i,line in enumerate(lines[sc:sc+h-6]):
            at=(RD() if any(k in line for k in ("Error","ERROR","Traceback","FAILED","Exception"))
                else YL() if "Warning" in line
                else GR() if any(k in line for k in ("done","SUCCESS","completed","NEW DONE"))
                else WH())
            self._w(y+3+i,lw+1,line[:rw],at)
        self._w(y+h-1,0," R=refresh  S=rescan  F=scan-errors  L=list  G=log  D=delta  E=errors  ↑↓=scroll  Enter=view-log",YL())

    def _job_detail_lines(self):
        if not self.jobs_list or self.jobs_cur>=len(self.jobs_list):
            return ["No job selected — press R to refresh"]
        jd=self.jobs_list[self.jobs_cur]
        return [f"Job ID  : {jd.get('jobid','?')}",
                f"Name    : {jd.get('name','?')}",
                f"State   : {jd.get('state','?')}",
                f"Elapsed : {jd.get('elapsed','?')}",
                f"Exit    : {jd.get('exit','?')}",
                "","Press G to view log output",
                "Press D to see new completions (after S=rescan)",
                "Press E to see error summary","Press S to rescan filesystem"]

    # ── 6 Terminal ────────────────────────────────────────────────────────
    def _draw_terminal(self,y,h,w):
        aw=36; mw=w-aw-1
        bg_ind=" [RUNNING...]" if not self._bg_done else ""
        self._w(y,1,f"TERMINAL{bg_ind}  num+Enter=insert  Enter=run  ESC=leave  ↑↓=history",
                YL() if not self._bg_done else AC())
        self._w(y,mw+2,"| COMMAND ASSIST",AC())
        out_h=h-4
        for i,(txt,col) in enumerate(self.cout[-out_h:]):
            at={"out":WH(),"err":RD(),"ok":GR(),"info":AC(),"prompt":WH()|curses.A_BOLD}.get(col,WH())
            self._w(y+1+i,0,txt[:mw],at)
        self._hl(y+h-3)
        cwd_s=str(self.cwd).replace(str(Path.home()),"~")
        pr=f" {cwd_s} > "
        self._w(y+h-2,0,pr,GR())
        self._w(y+h-2,len(pr),self.cin[:mw-len(pr)],WH()|curses.A_BOLD)
        try: self.scr.move(y+h-2,min(len(pr)+len(self.cin),mw-1)); curses.curs_set(1)
        except: pass
        task=self.sel_task or "TASK"; WDs=str(WD); SDs=str(SOURCEDATA); SOs=str(SLURM_OUT)
        secs=[
            ("SLURM",[
                ("squeue",      "squeue -u rsweety"),
                ("sacct today", "sacct -u rsweety --format=JobID,JobName,State,Elapsed -S today"),
                ("failed",      "sacct -u rsweety --state=FAILED,CANCELLED,TIMEOUT --format=JobID,JobName,State -S today"),
                ("cancel all",  "scancel -u rsweety"),
                ("sinfo",       "sinfo -s"),
            ]),
            ("Q1K CLI",[
                ("init HSJ",    f"source {VENV} && q1k-init --project-path {WDs}/ --task {task} --subject SUB_ID --site HSJ"),
                ("init MHC",    f"source {VENV} && q1k-init --project-path {WDs}/ --task {task} --subject SUB_ID --site MHC"),
                ("pylossless",  f"source {VENV} && q1k-pylossless --project-path {WDs}/ --task {task} --subject SUB_ID"),
                ("sync-loss",   f"source {VENV} && q1k-sync-loss --project-path {WDs}/ --task {task} --subject SUB_ID"),
                ("segment",     f"source {VENV} && q1k-segment --project-path {WDs}/ --task {task} --subject SUB_ID"),
                ("autorej",     f"source {VENV} && q1k-autorej --project-path {WDs}/ --task {task} --subject SUB_ID"),
            ]),
            ("COUNTS",[
                ("BIDS subs",   f"ls -d {WDs}/sub-* | wc -l"),
                ("pyll done",   f"ls {WDs}/derivatives/pylossless/derivatives/pylossless/ | wc -l"),
                ("sync done",   f"ls {WDs}/derivatives/pylossless/derivatives/sync_loss/ | wc -l"),
                ("seg done",    f"ls {WDs}/derivatives/pylossless/derivatives/sync_loss/derivatives/segment/epoch_fif_files/{task}/ 2>/dev/null | wc -l"),
                ("disk quota",  "diskusage_report"),
            ]),
            ("DATA",[
                ("ls slurm",    f"ls -lt {SOs}/ | head -15"),
                ("ls HSJ/eeg",  f"ls {SDs}/HSJ/eeg | wc -l"),
                ("ls HSJ/et",   f"ls {SDs}/HSJ/et | wc -l"),
                ("ls MHC/eeg",  f"ls {SDs}/MHC/eeg | wc -l"),
                ("du workdir",  f"du -sh {WDs}"),
            ]),
            ("ENV",[
                ("activate",    f"source {VENV}"),
                ("check mne",   f"source {VENV} && python -c \"import mne;print(mne.__version__)\""),
                ("check q1k",   f"source {VENV} && python -c \"import q1k;print(q1k.__version__)\""),
                ("module list", "module list"),
            ]),
        ]
        self._acmds=[]; row=y+1; ax=mw+1
        for sname,cmds in secs:
            if row>=y+h-3: break
            self._w(row,ax,f"-{sname}-"[:aw-1],AC()); row+=1
            for lbl,cmd in cmds:
                if row>=y+h-3: break
                num=len(self._acmds)+1; self._acmds.append(cmd)
                self._w(row,ax,f"{num:>2}:{lbl}"[:aw-1],YL()); row+=1
        self._w(min(row+1,y+h-3),ax,"num+Enter=insert",WH())

    # ── 7 Debug ───────────────────────────────────────────────────────────
    def _draw_debug(self,y,h,w):
        self._w(y,1,"DEBUG  R=failed-jobs  L=load-log  F=scan-errors  Tab=run-fix  ↑↓=scroll",AC())
        self._hl(y+1)
        lw=36; rw=w-lw-2
        self._w(y+2,0,f"{'JOBID':<14}{'NAME':<14}{'STATE':<8}"[:lw],HD())
        for i,(jid,name,state) in enumerate(self.dbg_jobs[:h-6]):
            sel=i==self.dbg_cur
            at=(SL() if sel else GR() if state=="COMPLETED" else
                RD() if state in ("FAILED","CANCELLED","TIMEOUT") else
                YL() if state=="RUNNING" else WH())
            self._w(y+3+i,0,f"{jid:<14}{name[:13]:<14}{state:<8}"[:lw],at)
        self._w(y+2,lw+1,"LOG / TRACEBACK",HD())
        for i,line in enumerate(self.dbg_lines[self.dbg_sc:self.dbg_sc+h-6]):
            at=(RD() if any(k in line for k in ("Error","ERROR","Traceback","FAILED","Exception"))
                else YL() if "Warning" in line
                else GR() if any(k in line for k in ("done","SUCCESS","completed"))
                else WH())
            self._w(y+3+i,lw+1,line[:rw],at)
        sug=self._fix_suggestions()
        if sug:
            sy=y+h-len(sug)-2
            self._w(sy-1,lw+1,"── FIX SUGGESTIONS (Tab=run first) ──",YL())
            for i,(desc,_) in enumerate(sug):
                self._w(sy+i,lw+1,f"  {i+1}. {desc}"[:rw],YL())
        self._w(y+h-1,0," R=failed-jobs  L=load-log  F=scan-errors  Tab=run-fix  ↑↓=navigate  Enter=load",YL())

    def _fix_suggestions(self):
        text="\n".join(self.dbg_lines); s=[]
        if "ModuleNotFoundError" in text or "ImportError" in text:
            m=re.search(r"No module named '(\S+)'",text)
            s.append((f"Missing module '{m.group(1) if m else '?'}' — activate venv",f"source {VENV}"))
        if "No such file" in text or "FileNotFoundError" in text:
            s.append(("Missing input file — rescan to check previous stage",""))
        if "MemoryError" in text or "Killed" in text:
            s.append(("Out of memory — resubmit with #SBATCH --mem=32G",""))
        if "DUE TO TIME LIMIT" in text:
            s.append(("Timed out — resubmit with #SBATCH --time=72:00:00",""))
        if "edf2ascii" in text or ("ascii" in text.lower() and "not found" in text.lower()):
            s.append(("Missing edf2ascii — need EyeLink module","module load eyelink"))
        if "/tmp/" in text:
            s.append(("Script reads /tmp/ — must use lustre path!","# Edit script to use /lustre07/..."))
        if "DIN2" in text and "KeyError" in text:
            s.append(("GO task DIN2 KeyError — HSJ uses DIN4/DIN5. Fix already applied in tools.py",""))
        return s[:5]

    # ── 8 Help ────────────────────────────────────────────────────────────
    def _draw_help(self,y,h,w):
        lines=[
            ("Q1K Pipeline Manager v5",AC()),("",WH()),
            ("FIXES IN v5:",HD()),
            ("  Overview/Tasks: progress bars now show per-stage counts (INIT/PYLS/SYNC/SEGM)",WH()),
            ("  detect_stage: independent per-stage checks — PYLS done => INIT marked done",WH()),
            ("  Pipeline drill: INFO column shows submitted job ID if available",WH()),
            ("  Auto-scan: filesystem rescanned every 5 min automatically",WH()),
            ("  Terminal: 'init MHC' label fixed (was 'MNI')",WH()),
            ("",WH()),
            ("TABS:",HD()),
            ("  1 Overview    — dataset totals, per-stage progress per task",WH()),
            ("  2 Tasks       — select active task",WH()),
            ("  3 Pipeline    — HSJ/MHC counts, ready/blocked, drill-down list",WH()),
            ("  4 Participants— individual browse, check, detail, data verify",WH()),
            ("  5 Jobs        — live SLURM list, log tail, delta, error scan",WH()),
            ("  6 Terminal    — non-blocking shell; ESC=leave; num+Enter=insert",WH()),
            ("  7 Debug       — log viewer, traceback, auto fix suggestions",WH()),
            ("",WH()),
            ("KEY ACTIONS:",HD()),
            ("  T  = test-run on 1 subject (non-blocking, streams to Terminal)",WH()),
            ("  Y  = submit SLURM array (ready+verified subjects only)",WH()),
            ("  B  = batch submit checked participants",WH()),
            ("  S  = rescan filesystem (shows delta of new completions)",WH()),
            ("",WH()),
            (f"WD:   {WD}",AC()),
            (f"VENV: {VENV}",AC()),
            (f"Auto-scan interval: {AUTO_SCAN_INTERVAL}s",AC()),
        ]
        for i,(txt,attr) in enumerate(lines[:h-1]):
            self._w(y+i,2,txt[:w-3],attr)

    # ── event router ─────────────────────────────────────────────────────
    def run(self):
        while self.running:
            try:
                if self._poll_bg(): pass
                self._check_auto_scan()   # FIX v5: auto-scan
                curses.curs_set(0)
                self.draw()
                key=self.scr.getch()
                if key==-1: continue
                self._handle(key)
            except KeyboardInterrupt: self.running=False
            except curses.error: pass

    def _handle(self,key):
        if self.tab==5: self._h_terminal(key); return
        if key in (ord('q'),ord('Q')): self.running=False; return
        if key in (ord('s'),ord('S')): self._do_scan(); return
        if ord('1')<=key<=ord('8'): self.tab=key-ord('1'); curses.curs_set(0); return
        {0:self._h_overview,1:self._h_tasks,2:self._h_pipeline,
         3:self._h_participants,4:self._h_jobs,6:self._h_debug
         }.get(self.tab,lambda k:None)(key)

    def _h_overview(self,key):
        if key==curses.KEY_UP:
            i=self.task_list.index(self.sel_task) if self.sel_task in self.task_list else 0
            self.sel_task=self.task_list[max(0,i-1)]; self._filt()
        elif key==curses.KEY_DOWN:
            i=self.task_list.index(self.sel_task) if self.sel_task in self.task_list else 0
            self.sel_task=self.task_list[min(len(self.task_list)-1,i+1)]; self._filt()
        elif key==10: self.tab=2

    def _h_tasks(self,key):
        if key==curses.KEY_UP: self.tcur=max(0,self.tcur-1)
        elif key==curses.KEY_DOWN: self.tcur=min(len(self.task_list)-1,self.tcur+1)
        elif key in (10,curses.KEY_RIGHT):
            if self.task_list: self.sel_task=self.task_list[self.tcur]; self._filt(); self.tab=2

    def _h_pipeline(self,key):
        if self.pipe_drill:
            if key==27: self.pipe_drill=False
            elif key==curses.KEY_UP:   self.pipe_dcur=max(0,self.pipe_dcur-1)
            elif key==curses.KEY_DOWN: self.pipe_dcur=min(len(self.pipe_dlist)-1,self.pipe_dcur+1)
            elif key==ord(' '):
                if self.pipe_dlist:
                    p,_,_=self.pipe_dlist[self.pipe_dcur]
                    pid=p["id"]
                    if pid in self.selected_pids: self.selected_pids.discard(pid)
                    else: self.selected_pids.add(pid)
                    self.pipe_dcur=min(self.pipe_dcur+1,len(self.pipe_dlist)-1)
            elif key in (ord('b'),ord('B')): self._batch_submit_selected()
            elif key in (ord('a'),ord('A')): self._submit_all_ready(self.sel_task,STAGES[self.pipe_stage])
            elif key in (ord('t'),ord('T')): self._test_run_bg(self.sel_task,STAGES[self.pipe_stage])
            return
        if key==curses.KEY_UP:   self.pipe_stage=max(0,self.pipe_stage-1)
        elif key==curses.KEY_DOWN: self.pipe_stage=min(len(STAGES)-1,self.pipe_stage+1)
        elif key==10: self.pipe_drill=True; self.pipe_dcur=0; self.pipe_doff=0
        elif key in (ord('h'),ord('H')):
            opts=["ALL"]+HOSPITALS; self.pipe_hosp=opts[(opts.index(self.pipe_hosp)+1)%len(opts)]
        elif key in (ord('t'),ord('T')): self._test_run_bg(self.sel_task,STAGES[self.pipe_stage])
        elif key in (ord('y'),ord('Y')): self._submit_all_ready(self.sel_task,STAGES[self.pipe_stage])
        elif key in (ord('a'),ord('A')):
            if self.sel_task:
                for s in STAGES: self._submit_all_ready(self.sel_task,s)

    def _h_participants(self,key):
        if self.det_p:
            if key in (27,ord('q')): self.det_p=None
            elif key==curses.KEY_UP:   self.det_sc=max(0,self.det_sc-1)
            elif key==curses.KEY_DOWN: self.det_sc=min(len(STAGES)-1,self.det_sc+1)
            elif key==10: self._sub_p_stage(self.det_p,self.sel_task,STAGES[self.det_sc])
            elif key in (ord('r'),ord('R')): self._next_stage_p(self.det_p,self.sel_task)
            elif key in (ord('a'),ord('A')): self._all_stages_p(self.det_p,self.sel_task)
            elif key in (ord('t'),ord('T')): self._test_run_bg(self.sel_task,STAGES[self.det_sc],self.det_p)
            elif key in (ord('d'),ord('D'),ord('l'),ord('L')):
                self._load_dbg_participant(self.det_p); self.tab=6
            return
        if key==curses.KEY_UP:     self.pcur=max(0,self.pcur-1)
        elif key==curses.KEY_DOWN: self.pcur=min(len(self.filt_ps)-1,self.pcur+1)
        elif key==curses.KEY_PPAGE:self.pcur=max(0,self.pcur-10)
        elif key==curses.KEY_NPAGE:self.pcur=min(len(self.filt_ps)-1,self.pcur+10)
        elif key==ord(' '):
            if self.filt_ps:
                pid=self.filt_ps[self.pcur]["id"]
                if pid in self.selected_pids: self.selected_pids.discard(pid)
                else: self.selected_pids.add(pid)
                self.pcur=min(self.pcur+1,len(self.filt_ps)-1)
        elif key==10:
            if self.filt_ps: self.det_p=self.filt_ps[self.pcur]; self.det_sc=0
        elif key in (ord('r'),ord('R')):
            if self.filt_ps: self._next_stage_p(self.filt_ps[self.pcur],self.sel_task)
        elif key in (ord('t'),ord('T')):
            if self.filt_ps:
                p=self.filt_ps[self.pcur]
                s=self._next_pending(p,self.sel_task) or STAGES[0]
                self._test_run_bg(self.sel_task,s,p)
        elif key in (ord('b'),ord('B')): self._batch_submit_selected()
        elif key in (ord('c'),ord('C')): self.selected_pids.clear(); self._status("Cleared",True)
        elif key in (ord('h'),ord('H')):
            opts=["ALL","HSJ","MHC"]; self.fh=opts[(opts.index(self.fh)+1)%3]; self._filt()
        elif key in (ord('f'),ord('F')):
            opts=["ALL","DONE","ERROR","PENDING","PARTIAL"]
            self.fss=opts[(opts.index(self.fss)+1)%len(opts)]; self._filt()
        elif key==ord('/'): self._do_search()
        elif key==27: self.fsearch=""; self.fh="ALL"; self.fss="ALL"; self._filt()

    def _h_jobs(self,key):
        if key in (ord('r'),ord('R')): self._refresh_squeue()
        elif key in (ord('s'),ord('S')): self._do_scan()
        elif key in (ord('f'),ord('F')): self._scan_errors()
        elif key in (ord('l'),ord('L')): self.jobs_panel="list"
        elif key in (ord('g'),ord('G')):
            self.jobs_panel="log"
            if self.jobs_list and self.jobs_cur<len(self.jobs_list):
                self._load_job_log(self.jobs_list[self.jobs_cur])
        elif key in (ord('d'),ord('D')): self.jobs_panel="delta"
        elif key in (ord('e'),ord('E')): self.jobs_panel="errors"
        elif key==curses.KEY_UP:
            if self.jobs_panel=="log": self.jobs_log_sc=max(0,self.jobs_log_sc-1)
            else: self.jobs_cur=max(0,self.jobs_cur-1)
        elif key==curses.KEY_DOWN:
            if self.jobs_panel=="log": self.jobs_log_sc+=1
            else: self.jobs_cur=min(len(self.jobs_list)-1,self.jobs_cur+1)
        elif key==10:
            if self.jobs_list and self.jobs_cur<len(self.jobs_list):
                self._load_job_log(self.jobs_list[self.jobs_cur]); self.jobs_panel="log"

    def _h_debug(self,key):
        if key in (ord('r'),ord('R')): self._load_failed_jobs()
        elif key in (ord('l'),ord('L')):
            if self.dbg_jobs and self.dbg_cur<len(self.dbg_jobs):
                jid,name,_=self.dbg_jobs[self.dbg_cur]; self._load_slurm_log(jid,name)
        elif key in (ord('f'),ord('F')): self._scan_all_errors()
        elif key==curses.KEY_UP:
            if self.dbg_sc>0: self.dbg_sc-=1
            elif self.dbg_cur>0:
                self.dbg_cur-=1
                if self.dbg_jobs: jid,name,_=self.dbg_jobs[self.dbg_cur]; self._load_slurm_log(jid,name)
        elif key==curses.KEY_DOWN:
            if self.dbg_sc<len(self.dbg_lines)-1: self.dbg_sc+=1
            elif self.dbg_cur<len(self.dbg_jobs)-1:
                self.dbg_cur+=1; jid,name,_=self.dbg_jobs[self.dbg_cur]; self._load_slurm_log(jid,name)
        elif key==10:
            if self.dbg_jobs and self.dbg_cur<len(self.dbg_jobs):
                jid,name,_=self.dbg_jobs[self.dbg_cur]; self._load_slurm_log(jid,name)
        elif key==9:
            sug=self._fix_suggestions()
            if sug:
                _,cmd=sug[0]
                if cmd: self.tab=5; self.cin=cmd; self._status("Fix loaded in Terminal — press Enter",True)

    def _h_terminal(self,key):
        curses.curs_set(1)
        if key==27:
            if self.cin: self.cin=""
            else: self.tab=0; curses.curs_set(0)
            return
        if key==10: self._exec()
        elif key==curses.KEY_UP:
            if self.chidx>0: self.chidx-=1; self.cin=self.chist[self.chidx]
        elif key==curses.KEY_DOWN:
            if self.chidx<len(self.chist)-1: self.chidx+=1; self.cin=self.chist[self.chidx]
            else: self.chidx=len(self.chist); self.cin=""
        elif key==9: self._autocomplete()
        elif key in (curses.KEY_BACKSPACE,127,8): self.cin=self.cin[:-1]
        elif 32<=key<=126: self.cin+=chr(key)

    # ── confirmation ─────────────────────────────────────────────────────
    def _confirm(self,lines):
        h,w=self.scr.getmaxyx()
        bh=min(len(lines)+6,h-4); bw=min(max((len(l) for l in lines),default=40)+6,w-4)
        by=(h-bh)//2; bx=(w-bw)//2
        try:
            for row in range(bh): self._w(by+row,bx," "*bw,HD())
            self._w(by,      bx,"┌"+"─"*(bw-2)+"┐",AC())
            self._w(by+bh-1, bx,"└"+"─"*(bw-2)+"┘",AC())
            for row in range(1,bh-1):
                self._w(by+row,bx,"│",AC()); self._w(by+row,bx+bw-1,"│",AC())
            self._w(by+1,bx+2,"CONFIRM SUBMISSION",HD())
            self._w(by+2,bx+2,"─"*(bw-4),AC())
            for i,line in enumerate(lines[:bh-6]): self._w(by+3+i,bx+2,line[:bw-4],WH())
            self._w(by+bh-2,bx+2,"Y = submit    N / ESC = cancel",YL())
        except curses.error: pass
        self.scr.refresh()
        while True:
            k=self.scr.getch()
            if k in (ord('y'),ord('Y')): return True
            if k in (ord('n'),ord('N'),27): return False

    # ── actions ───────────────────────────────────────────────────────────
    def _do_scan(self, silent=False):
        h,w=self.scr.getmaxyx()
        if not silent:
            self._status("Scanning filesystem...",True); self.draw(); self.scr.refresh()
        def cb(i,total,pid):
            if not silent and i%20==0:
                pct=i/total*100 if total else 0
                self._w(h//2,2,f"  Scanning {i}/{total} ({pct:.0f}%) — {pid:<30}  ",AC())
                self.scr.refresh()
        try:
            old_done={p["id"]:{t:{s for s,v in p.get("task_stages",{}).get(t,{}).items()
                                  if v=="done"} for t in p.get("eeg_tasks",[])}
                      for p in self.ps}
            self.ps=scan_participants(cb); save_inv(self.ps); self._rebuild()
            delta=[]
            for p in self.ps:
                for t in p.get("eeg_tasks",[]):
                    for s in STAGES:
                        was=s in old_done.get(p["id"],{}).get(t,set())
                        now=p.get("task_stages",{}).get(t,{}).get(s)=="done"
                        if now and not was:
                            delta.append(f"NEW DONE: {p['id']} / {t} / {s}")
            self.jobs_delta=delta or ["No new completions since last scan"]
            self._last_scan_time = time.time()
            n=len(self.ps); hsj=sum(1 for p in self.ps if p["hospital"]=="HSJ")
            self._status(
                f"Scanned {n} | HSJ:{hsj} MHC:{n-hsj} | {len(self.task_list)} tasks"
                f" | {len(delta)} new completions",True)
        except Exception as e:
            import traceback; self._status(f"Scan error:{e}",False)
            self.cout.append((traceback.format_exc(),"err"))

    def _test_run_bg(self,task,stage,p=None):
        if not task or not stage: return
        if not self._bg_done:
            self._status("A background job is already running — wait for it to finish",False); return
        ps_t=[p] if p else [x for x in self.ps
                             if task in x.get("eeg_tasks",[])
                             and x.get("task_stages",{}).get(task,{}).get(stage,"pending")=="pending"
                             and verify_data(x,task,stage)[0]]
        if not ps_t: self._status(f"No verified-ready subjects for {task}/{stage}",False); return
        subject=ps_t[0]; pid=subject["id"]; site=SITE_FLAG.get(subject["hospital"],subject["hospital"])
        cli=STAGE_CLI[stage]
        subj_arg=subject.get("raw_name",pid)
        flags=f"--project-path {PROJECT_PATH} --task {task} --subject {subj_arg}"
        if stage=="INIT": flags+=f" --site {site}"
        ok,reason=verify_data(subject,task,stage)
        if not ok:
            self._status(f"Data check FAILED: {reason}",False)
            self.dbg_lines=[f"Data verification failed for {pid}/{task}/{stage}","",f"Reason: {reason}"]
            return
        if not self._confirm([
            "  TEST RUN (non-blocking — UI stays live)","",
            f"  Subject : {pid}  ({subject['hospital']})",
            f"  Task    : {task}",f"  Stage   : {stage}",
            f"  Verify  : {reason}","",
            f"  {cli} {flags}","",
            "  Output streams to Terminal tab",]): return
        self.tab=5
        cmd=venv_cmd(f"{cli} {flags}")
        self._bg_cmd=f"{cli} {pid}/{task}/{stage}"
        self._bg_done=False
        self.cout.append((f"=== TEST: {pid}/{task}/{stage} ===","info"))
        self.cout.append((f"> {cmd[:100]}","prompt"))
        self._status(f"Running {pid}/{task}/{stage} in background...",True)
        run_cmd_stream(cmd, self._bg_queue, self.cwd)

    def _submit_all_ready(self,task,stage):
        if not task: return
        prev=STAGES[STAGES.index(stage)-1] if STAGES.index(stage)>0 else None
        ready=[p for p in self.ps
               if task in p.get("eeg_tasks",[])
               and p.get("task_stages",{}).get(task,{}).get(stage,"pending")=="pending"
               and (prev is None or p.get("task_stages",{}).get(task,{}).get(prev)=="done")
               and verify_data(p,task,stage)[0]]
        if not ready: self._status(f"No verified-ready candidates for {task}/{stage}",False); return
        for site in HOSPITALS:
            sp=[p for p in ready if p["hospital"]==site]
            if sp: self._submit_array_for(task,stage,site,sp)

    def _submit_array_for(self,task,stage,site,ps_list):
        n=len(ps_list); jname=f"{task}_{STAGE_SHORT[stage]}_{site}"
        cli=STAGE_CLI[stage]
        sf=f"--site {SITE_FLAG.get(site,site)}" if stage=="INIT" else ""
        preview=ps_list[:8]
        blocked=[p for p in self.ps
                 if task in p.get("eeg_tasks",[])
                 and p.get("hospital")==site
                 and p.get("task_stages",{}).get(task,{}).get(stage,"pending")=="pending"
                 and not verify_data(p,task,stage)[0]]
        if not self._confirm([
            f"  Task     : {task}",f"  Stage    : {stage}",
            f"  Site     : {site}",f"  Jobs     : {n}  (array 1-{n}%10)",
            f"  Blocked  : {len(blocked)} excluded (data not ready)",
            f"  Command  : {cli} --task {task} {sf}",
            f"  Job name : {jname}",f"  Logs     : {SLURM_OUT}/{jname}_<id>_<arr>.out",
            "  ─────────────────────────────────────",
            f"  Subjects ({min(n,8)} of {n} shown):",
        ]+[f"    {p['id']}" for p in preview]+
         ([f"    ...+{n-8} more"] if n>8 else [])):
            self._status(f"Cancelled {task}/{stage}/{site}",True); return
        lp=WD/f"_tmp_sublist_{task}_{stage}_{site}.txt"
        sp=WD/f"_tmp_slurm_{task}_{stage}_{site}.sh"
        try:
            lp.write_text("\n".join(p.get("raw_name", p["id"]) for p in ps_list)+"\n")
            sp.write_text(make_slurm_script(task,stage,lp,site,n,jname)); sp.chmod(0o755)
            self._status(f"Submitting {n} jobs {task}/{stage}/{site}...",True)
            self.draw(); self.scr.refresh()
            out,err,code=run_cmd(f"sbatch {sp}",30)
            if code==0:
                m=re.search(r"(\d+)",out); jid=m.group(1) if m else "?"
                for p in ps_list: p.setdefault("slurm_jobs",{})[f"{task}:{stage}"]=f"{jid}_arr"
                save_inv(self.ps); self._rebuild()
                self._status(f"Submitted {n} | {task}/{stage}/{site} | JobArray:{jid}",True)
            else:
                self._status(f"sbatch failed:{err[:80]}",False)
                self.dbg_lines=[f"sbatch error {task}/{stage}/{site}:","",err]; self.dbg_sc=0
        except Exception as e: self._status(f"Submit error:{e}",False)

    def _sub_p_stage(self,p,task,stage):
        ts=p.get("task_stages",{}).get(task,{})
        if ts.get(stage,"pending") not in ("pending","error"):
            self._status(f"{stage} is {ts.get(stage)}",False); return
        ok,reason=verify_data(p,task,stage)
        if not ok:
            self._status(f"Data check failed: {reason}",False)
            self.dbg_lines=[f"Cannot submit {p['id']}/{task}/{stage}","",f"Reason: {reason}"]; return
        cli=STAGE_CLI[stage]; site=SITE_FLAG.get(p["hospital"],p["hospital"])
        subj_arg=p.get("raw_name",p["id"])
        flags=f"--project-path {PROJECT_PATH} --task {task} --subject {subj_arg}"
        if stage=="INIT": flags+=f" --site {site}"
        if stage in ("PYLOSSLESS","AUTOREJECT"): flags+=" --slurm"
        if not self._confirm([f"  {p['id']}  ({p['hospital']})",
                               f"  {task} / {stage}",f"  Verify: {reason}",
                               f"  {cli} {flags}"]): return
        cmd=venv_cmd(f"{cli} {flags}")
        self._status(f"Submitting {p['id']}/{task}/{stage}...",True); self.draw(); self.scr.refresh()
        out,err,code=run_cmd(cmd,60)
        if code==0:
            m=re.search(r"(\d+)",out); jid=m.group(1) if m else "direct"
            p.setdefault("task_stages",{}).setdefault(task,{s:"pending" for s in STAGES})[stage]="queued"
            p.setdefault("slurm_jobs",{})[f"{task}:{stage}"]=jid
            save_inv(self.ps); self._rebuild()
            self._status(f"Submitted {p['id']}/{task}/{stage} Job:{jid}",True)
        else:
            self._status(f"Failed:{err[:80]}",False)
            self.dbg_lines=[f"Error:{p['id']}/{task}/{stage}","",out,err]; self.dbg_sc=0

    def _next_stage_p(self,p,task):
        s=self._next_pending(p,task)
        if s: self._sub_p_stage(p,task,s)
        else: self._status(f"{p['id']}/{task}: nothing pending",True)

    def _all_stages_p(self,p,task):
        count=0
        for s in STAGES:
            if p.get("task_stages",{}).get(task,{}).get(s,"pending") in ("pending","error"):
                self._sub_p_stage(p,task,s); count+=1
        if not count: self._status(f"Nothing to submit for {p['id']}/{task}",True)

    def _batch_submit_selected(self):
        if not self.selected_pids: self._status("No participants checked",False); return
        if not self.sel_task: self._status("No task",False); return
        task=self.sel_task
        sel_list=[p for p in self.ps if p["id"] in self.selected_pids
                  and task in p.get("eeg_tasks",[])]
        by_stage=defaultdict(list)
        for p in sel_list:
            s=self._next_pending(p,task)
            if s and verify_data(p,task,s)[0]: by_stage[s].append(p)
        if not by_stage: self._status("No verified-ready subjects in selection",False); return
        for stage,ps_list in by_stage.items():
            for site in HOSPITALS:
                sp=[p for p in ps_list if p["hospital"]==site]
                if sp: self._submit_array_for(task,stage,site,sp)
        self.selected_pids.clear()

    # ── Jobs tab ──────────────────────────────────────────────────────────
    def _refresh_squeue(self):
        self._status("Refreshing...",True); self.draw(); self.scr.refresh()
        try:
            r=subprocess.run(["squeue","-u","rsweety",
                "--format=%i|%j|%T|%M|%V","--noheader"],
                capture_output=True,text=True,timeout=15)
            self.jobs_list=[]
            for line in r.stdout.strip().split("\n"):
                if not line.strip(): continue
                parts=line.split("|")
                if len(parts)>=3:
                    self.jobs_list.append({"jobid":parts[0].strip(),"name":parts[1].strip(),
                                           "state":parts[2].strip(),
                                           "elapsed":parts[3].strip() if len(parts)>3 else "?",
                                           "submit":parts[4].strip() if len(parts)>4 else "?"})
            r2=subprocess.run(["sacct","-u","rsweety",
                "--format=JobID,JobName,State,Elapsed,ExitCode",
                "-S","today","--noheader","--parsable2"],
                capture_output=True,text=True,timeout=15)
            for line in r2.stdout.strip().split("\n"):
                parts=line.split("|")
                if len(parts)>=3 and "." not in parts[0] and parts[0].strip():
                    self.jobs_list.append({"jobid":parts[0].strip(),
                                           "name":parts[1].strip() if len(parts)>1 else "?",
                                           "state":parts[2].strip() if len(parts)>2 else "?",
                                           "elapsed":parts[3].strip() if len(parts)>3 else "?",
                                           "exit":parts[4].strip() if len(parts)>4 else "?"})
            seen=set(); dedup=[]
            for jd in self.jobs_list:
                if jd["jobid"] not in seen: seen.add(jd["jobid"]); dedup.append(jd)
            self.jobs_list=dedup
            self._status(f"Refreshed: {len(self.jobs_list)} jobs",True)
        except Exception as e: self._status(f"squeue error:{e}",False)

    def _load_job_log(self,jd):
        jid=jd.get("jobid","?"); name=jd.get("name","?")
        self.jobs_log=[f"=== Job {jid} ({name}) ===",""]
        found=[]
        if SLURM_OUT.exists():
            found=(list(SLURM_OUT.glob(f"*_{jid}_*.out"))+
                   list(SLURM_OUT.glob(f"*_{jid}.out"))+
                   list(SLURM_OUT.glob(f"{name}*{jid}*.out")))
        found+=list(WD.glob(f"slurm-{jid}.out"))+list(WD.glob(f"slurm-{jid}_*.out"))
        if not found:
            self.jobs_log+=[f"No log found for {jid}",
                             f"Searched: {SLURM_OUT}/",
                             f"Try: ls {SLURM_OUT}/ | grep {jid}"]
        else:
            lf=sorted(set(found))[-1]
            try: self.jobs_log+=lf.read_text(errors="replace").split("\n")
            except Exception as e: self.jobs_log+=[f"Read error:{e}"]
        self.jobs_log_sc=0

    def _scan_errors(self):
        self._status("Scanning logs...",True); self.draw(); self.scr.refresh()
        self.jobs_errors=[]; self.jobs_log=["=== ERROR SCAN ===",""]
        count=0
        if SLURM_OUT.exists():
            for lf in sorted(SLURM_OUT.glob("*.out"))[-50:]:
                try:
                    content=lf.read_text(errors="replace")
                    errs=[l for l in content.split("\n")
                          if any(k in l for k in ("Error","ERROR","Traceback","FAILED","Exception","/tmp/"))]
                    if errs:
                        pid_m=re.search(r"(\d{4,6}[A-Z]\d?)",lf.name)
                        pid=pid_m.group(0) if pid_m else lf.name
                        self.jobs_errors.append((pid,lf.name,errs[0][:80]))
                        self.jobs_log+=[f"── {lf.name} ──"]+errs[:5]+[""]; count+=1
                except: pass
        if count==0: self.jobs_log+=["No errors found in recent logs — looking good!"]
        self.jobs_log_sc=0; self.jobs_panel="errors"
        self._status(f"Errors found in {count} log files",count>0)

    # ── Debug tab ─────────────────────────────────────────────────────────
    def _load_failed_jobs(self):
        self._status("Loading failed jobs...",True)
        try:
            r=subprocess.run(["sacct","-u","rsweety",
                "--format=JobID,JobName,State,Elapsed,ExitCode",
                "--state=FAILED,CANCELLED,TIMEOUT","-S","today",
                "--noheader","--parsable2"],
                capture_output=True,text=True,timeout=15)
            self.dbg_jobs=[]
            for line in r.stdout.strip().split("\n"):
                parts=line.split("|")
                if len(parts)>=3 and "." not in parts[0] and parts[0].strip():
                    self.dbg_jobs.append((parts[0].strip(),
                                          parts[1].strip() if len(parts)>1 else "?",
                                          parts[2].strip() if len(parts)>2 else "?"))
            if not self.dbg_jobs:
                self.dbg_lines=["No failed jobs today!"]; self.dbg_sc=0
            else:
                self.dbg_lines=[f"Found {len(self.dbg_jobs)} failed jobs","↑↓ to select, L to load log"]
                self.dbg_cur=0; jid,name,_=self.dbg_jobs[0]; self._load_slurm_log(jid,name)
            self._status(f"Loaded {len(self.dbg_jobs)} failed jobs",True)
        except Exception as e: self._status(f"sacct error:{e}",False)

    def _load_slurm_log(self,jid,name):
        self.dbg_lines=[f"=== Job {jid} ({name}) ===",""]
        found=[]
        if SLURM_OUT.exists():
            found=(list(SLURM_OUT.glob(f"*_{jid}_*.out"))+
                   list(SLURM_OUT.glob(f"*_{jid}.out"))+
                   list(SLURM_OUT.glob(f"{name}*{jid}*.out")))
        found+=list(WD.glob(f"slurm-{jid}.out"))+list(WD.glob(f"slurm-{jid}_*.out"))
        if not found:
            self.dbg_lines+=["No log found","Try: ls {SLURM_OUT}/ | grep {jid}"]
        else:
            lf=sorted(set(found))[-1]
            try: self.dbg_lines+=lf.read_text(errors="replace").split("\n")
            except Exception as e: self.dbg_lines+=[f"Read error:{e}"]
        self.dbg_sc=0

    def _load_dbg_participant(self,p):
        self.dbg_lines=[f"=== Logs for {p['id']} ===",""]
        found=list(SLURM_OUT.glob(f"*{p['id']}*.out")) if SLURM_OUT.exists() else []
        if not found:
            for key,jid in p.get("slurm_jobs",{}).items():
                found.extend(SLURM_OUT.glob(f"*_{jid.split('_')[0]}*.out") if SLURM_OUT.exists() else [])
        if not found: self.dbg_lines+=["No log files found"]
        else:
            lf=sorted(set(found))[-1]
            try: self.dbg_lines+=[f"File:{lf.name}",""]+lf.read_text(errors="replace").split("\n")
            except Exception as e: self.dbg_lines+=[f"Error:{e}"]
        self.dbg_sc=0
        self.dbg_jobs=[(jid,key,"?") for key,jid in p.get("slurm_jobs",{}).items()]
        self.dbg_cur=0

    def _scan_all_errors(self):
        self._status("Scanning...",True); self.draw(); self.scr.refresh()
        self.dbg_lines=["=== ERROR SCAN ===",""]; count=0
        if SLURM_OUT.exists():
            for lf in sorted(SLURM_OUT.glob("*.out"))[-50:]:
                try:
                    content=lf.read_text(errors="replace")
                    errs=[l for l in content.split("\n")
                          if any(k in l for k in ("Error","ERROR","Traceback","FAILED","Exception","/tmp/"))]
                    if errs: self.dbg_lines+=[f"── {lf.name} ──"]+errs[:5]+[""]; count+=1
                except: pass
        if count==0: self.dbg_lines+=["No errors found."]
        self.dbg_sc=0; self._status(f"Errors in {count} files",count>0)

    # ── Terminal ──────────────────────────────────────────────────────────
    def _exec(self):
        cmd=self.cin.strip()
        if not cmd: return
        if cmd.isdigit():
            idx=int(cmd)-1
            if 0<=idx<len(self._acmds): self.cin=self._acmds[idx]; return
        self.chist.append(cmd); self.chidx=len(self.chist); self.cin=""
        self.cout.append((f"> {cmd}","prompt"))
        if cmd.startswith("cd "):
            try:
                nw=Path(os.path.expanduser(cmd[3:].strip()))
                if not nw.is_absolute(): nw=self.cwd/nw
                nw=nw.resolve()
                if nw.exists(): self.cwd=nw; os.chdir(nw); self.cout.append((f"  -> {self.cwd}","ok"))
                else: self.cout.append((f"  Not found:{nw}","err"))
            except Exception as e: self.cout.append((f"  {e}","err"))
            return
        if not self._bg_done:
            self._status("Background job still running — wait or open new terminal",False); return
        self._bg_cmd=cmd; self._bg_done=False
        self._status(f"Running: {cmd[:60]}",True)
        run_cmd_stream(cmd, self._bg_queue, self.cwd)

    def _autocomplete(self):
        matches=[c for c in self._acmds if self.cin.lower() in c.lower()]
        if len(matches)==1: self.cin=matches[0]
        elif matches:
            for c in matches[:4]: self.cout.append((f"  {c}","info"))

    def _do_search(self):
        h,w=self.scr.getmaxyx(); curses.curs_set(1); query=""
        while True:
            self._w(h-1,0,f" Search: {query:<40}",SL())
            try: self.scr.move(h-1,9+len(query))
            except: pass
            self.scr.refresh(); key=self.scr.getch()
            if key in (10,27): break
            elif key in (curses.KEY_BACKSPACE,127,8): query=query[:-1]
            elif 32<=key<=126: query+=chr(key)
        self.fsearch=query; self._filt(); curses.curs_set(0)

# ── Entry ──────────────────────────────────────────────────────────────────
def main(scr): App(scr).run()
if __name__=="__main__":
    try: curses.wrapper(main)
    except KeyboardInterrupt: pass
    print("\nQ1K Pipeline Manager v5 closed finally :) :) :) !")
