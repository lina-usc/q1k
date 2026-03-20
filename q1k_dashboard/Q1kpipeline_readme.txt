# Q1K Pipeline Manager & Autopilot

A terminal-based dashboard and autonomous pipeline managerforthe
Quebec1000Families(Q1K)EEG/Eye-Trackingpreprocessingpipeline,
runningontheNarvalHPCcluster(AllianceCanada).

---

##Overview

TheQ1KpipelineconvertsrawEEG(`.mff`)andEye-Tracking(`.asc`)data
through5sequentialstagesintoanalysis-readyepochedfiles:

```
RawEEG/ET→INIT→PYLOSSLESS→SYNC_LOSS→SEGMENTATION→AUTOREJECT
(BIDS)(ICA/artifact)(EEG+ETsync)(epochs)(epochcleaning)
```

Thisrepositoryprovidestwotoolstomanagethispipelineatscale(~800subjects):

|Tool|File|Purpose|
|------|------|---------|
|Manager|`q1k_manager.py`|InteractiveTUIdashboard—monitor,submit,debug|
|Autopilot|`q1k_autopilot.py`|Autonomousjobsubmissionwithdependencychains|
|ETConversion|`mhc_go_et_convert.py`|ConvertMHCGOEyeLink.edf→.asc|
|Watcher|`q1k_autowatcher.sh`|Auto-submitqueuedjobswhenSLURMslotsfree|

---

PipelineStages

|Stage|CLICommand|Input|Output|ETRequired|
|-------|-------------|-------|--------|-------------|
|INIT|`q1k-init`|`.mff`EEG+`.asc`ET|BIDS`.edf`|No|
|PYLOSSLESS|`q1k-pylossless`|BIDS`.edf`|ICAannotations|No|
|SYNC_LOSS|`q1k-sync-loss`|PyLosslessoutput+`.asc`|CleanedEEG(+ET)|GO/NSP/PLR/VSonly|
|SEGMENTATION|`q1k-segment`|Sync_lossoutput|Epoch`.fif`files|No|
|AUTOREJECT|`q1k-autorej`|Epoch`.fif`files|Cleanepochs|No|

TasksandEye-Tracking

|Code|Task|Eye-TrackingatSYNC_LOSS|
|------|------|:---:|
|RS|RestingState|No|
|RSRio|RestingState(Riomovie)|No|
|VEP|VisualEvokedPotential|No|
|AEP|AuditoryEvokedPotential|No|
|GO|GapOverlap|Yes|
|PLR|PupillaryLightReflex|Yes|
|VS|VisualSearch|Yes|
|NSP|NaturalisticSocialPreference|Yes|
|TO|ToneOddball|No|

>VEPnote:AlthoughtheQ1KpaperlistsVEPashavingEye-Tracking,
>VEPETdatawasnotcollectedforthiscohort.VEPisthereforeexcluded
>from`ET_SYNC_TASKS`in`sync_loss/cli.py`.Donotre-additwithout
>firstconfirmingETdataavailabilitywiththedatacollectionteam.



DirectoryStructure

```
/lustre07/scratch/rsweety/white_paper/wd/
├──sourcedata/
│├──HSJ/
││├──eeg/Q1K_HSJ_100XXX_YY/#Raw.mffEEGfiles
││└──et/Q1K_HSJ_100XXX_YY/#Eye-tracking.ascfiles
│└──MHC/
│├──eeg/Q1K_MHC_200XXX_YY/#Raw.mffEEGfiles
│└──et/Q1K_MHC_200XXX_YY/#Eye-tracking.ascfiles
├──sub-{BIDS_ID}/ses-01/eeg/#BIDSoutput(Stage1)
├──derivatives/
│└──pylossless/derivatives/
│├──pylossless/sub-{ID}/#Stage2output
│└──sync_loss/sub-{ID}/#Stage3output
│└──derivatives/
│├──segment/epoch_fif_files/{TASK}/#Stage4
│└──autorej/epoch_fif_files/{TASK}/#Stage5
├──reports/
│├──init/{TASK}/#Per-subjectmarimoreports
│├──sync_loss/{TASK}/
│└──segment/{TASK}/
├──slurm_output/#AllSLURMjoblogs
├──subject_lists/#Subjectlist.txtfilesforjobs
├──q1k_manager.py#TUIDashboard
├──q1k_autopilot.py#Autonomouspipelinemanager
├──q1k_autowatcher.sh#SLURMslotwatcher
├──mhc_go_et_convert.py#MHCGOETconversion
├──submit_queue.txt#Pendingjobscriptsforwatcher
├──autopilot.log#Autopilotrunlogs
├──watcher.log#Watcherlogs
└──participant_inventory.json#Cachedscanresults



q1k_manager.py



AninteractiveterminalUI(TUI)builtwithPython`curses`.Noexternal
dependencies—purestdlib.Provides:

-Real-timepipelineoverview—per-taskcountsateachstage
-Per-subjecttracking—browse,filter,checkindividualparticipants
-Jobsubmission—submitSLURMarraysdirectlyfromtheUI
-Logviewer—browseSLURMoutputlogs,scanforerrors
-Built-interminal—runshellcommandswithoutleavingtheUI
-Debugtab—viewtracebacks,getfixsuggestions
-Auto-scan—rescansfilesystemevery5minutesautomatically

Installation
source/lustre07/scratch/rsweety/white_paper/wd/q1k_venv_scratch/bin/activate


cd/lustre07/scratch/rsweety/white_paper/wd
python3 q1k_manager.py



|Key|Action|
|-----|--------|
|`1`–`8`|Switchtabs|
|`S`|Rescanfilesystem(updatesallcounts)|
|`Q`|Quit|
|`↑↓`|Navigatelists|
|`Enter`|Select/drilldown/submit|
|`T`|Testrunon1subject(non-blocking)|
|`Y`|SubmitSLURMarrayforreadysubjects|
|`H`|ToggleHSJ/MHC/ALLfilter|
|`F`|Cyclestatusfilter(ALL/DONE/ERROR/PENDING/PARTIAL)|
|`/`|SearchbysubjectID|
|`Space`|Check/unchecksubject|
|`B`|Batchsubmitcheckedsubjects|
|`ESC`|Goback/clearfilters|

Tabs|Key|Description|
|-----|-----|-------------|
|Overview|`1`|Datasettotals,per-taskprogressbars|
|Tasks|`2`|Selectactivetask|
|Pipeline|`3`|HSJ/MHCcountsperstage,ready/blockedcounts,drill-down|
|Participants|`4`|Individualsubjectbrowserwithstagestatus|
|Jobs|`5`|LiveSLURMqueue,logviewer,errorscanner|
|Terminal|`6`|Built-inshellwithcommandshortcuts|
|Debug|`7`|Failedjoblogs,tracebacks,fixsuggestions|
|Help|`8`|Keyreference|


Themanagerdetectspipelinestagecompletionbycheckingforoutputfiles:

-INITdone:`sub-{pid}/ses-01/eeg/*_task-{task}_*_eeg.edf`exists
-PYLOSSLESSdone:`derivatives/pylossless/.../sub-{pid}/*_ll_config.yaml`exists
-SYNC_LOSSdone:`derivatives/.../sync_loss/sub-{pid}/ses-01/eeg/*_task-{task}_*`exists
-SEGMENTATIONdone:`derivatives/.../segment/epoch_fif_files/{task}/sub-{pid}_*_epo.fif`exists
-AUTOREJECTdone:`derivatives/.../autorej/epoch_fif_files/{task}/sub-{pid}_*_epo.fif`exists

Detectionisindependentperstage—ifPYLOSSLESSisdone,INITis
inferreddoneevenifBIDSfileswerewrittentoadifferentpath.



Scansallparticipants,determinesthecorrectnextpipelinestageforeach,
verifiesprerequisites,andsubmitsSLURMjobarrayswithdependencychains.


Theautopilotflagssubjectsneedinghumanreview:

|Flag|Meaning|
|------|---------|
|`ATTENTION:No.ascETfile`|GO/NSP/PLR/VSsubjectmissingET—SYNC_LOSSwillfail|
|`ATTENTION:Failed2x`|Subjectfailedin2+recentSLURMlogs(last7days)|
|`WARNING:SmallBIDSEDF`|EDFfile<5MB—possiblycorruptortruncated|
|`WARNING:Fewevents`|EventsTSVhas<10rows—possiblybadrecording|

---

##q1k_autowatcher.sh

MonitorstheSLURMqueueandauto-submitsjobsfrom`submit_queue.txt`
whenthetaskcountdropsbelowthethreshold(950bydefault).


Subject ID 

 Format | Example | Used for |
 Raw sourcedata name | `Q1K_HSJ_100123_F1` | `q1k-init --subject` argument |
 BIDS subject ID | `100123F1` | All other stages, folder names |
 ET folder ID (HSJ) | `10046S1_GO.asc` | Inside `sourcedata/HSJ/et/Q1K_HSJ_10046_S1/` |
 ET folder ID (MHC) | `265P_GO.asc` | Inside `sourcedata/MHC/et/Q1K_MHC_200265_P/` |

ID mapping rules:
`Q1K_HSJ_100123_F1` → BIDS `100123F1` (strip `Q1K_HSJ_`, remove `_`)
`Q1K_MHC_200265_P` → BIDS `200265P` (strip `Q1K_MHC_`, remove `_`)
`Q265_P` (MHC ET) → `Q1K_MHC_200265_P` (prepend `Q1K_MHC_200`)

