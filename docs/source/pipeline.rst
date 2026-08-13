Pipeline Overview
=================

SyneQxis organizes Q1K processing into explicit stages. Each stage produces
derivatives that can be inspected, tested, and reused by later stages.

Stages
------

1. ``q1k-init`` converts raw EEG and eye-tracking source files to the project
   BIDS layout and creates subject-level reports.
2. ``q1k-pylossless`` runs PyLossless for non-destructive EEG artifact
   annotation.
3. ``q1k-sync-loss`` aligns EEG and eye-tracking event streams, applies
   lossless cleaning decisions, and writes synchronized derivatives.
4. ``q1k-segment`` creates task-specific epochs using task configuration and
   event labels.
5. ``q1k-autorej`` applies AutoReject to repair or reject noisy epochs.
6. ``q1k-tracking`` audits stage completion and data loss across subjects,
   tasks, and sites.

Supported tasks
---------------

The package currently supports ``RS``, ``RSRio``, ``VEP``, ``AEP``, ``GO``,
``PLR``, ``VS``, ``NSP``, and ``TO``.

Design principles
-----------------

The release infrastructure follows the PyLossless model: testable Python
modules, command line entry points, continuous integration, documentation builds,
coverage reporting, and release automation.
