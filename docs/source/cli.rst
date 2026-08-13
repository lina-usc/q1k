Command Line Interface
======================

All processing stages share the same core pattern: provide a project path, task,
and either one subject or all available subjects.

.. code-block:: bash

   q1k-init --project-path /path/to/project --task GO --subject HSJ0043F1
   q1k-pylossless --project-path /path/to/project --task GO --subject HSJ0043F1
   q1k-sync-loss --project-path /path/to/project --task GO --subject HSJ0043F1
   q1k-segment --project-path /path/to/project --task GO --subject HSJ0043F1
   q1k-autorej --project-path /path/to/project --task GO --subject HSJ0043F1

Batch/HPC use
-------------

Stages that support Slurm expose a ``--slurm`` flag:

.. code-block:: bash

   q1k-pylossless --project-path /path/to/project --task GO --all --slurm
   q1k-autorej --project-path /path/to/project --task GO --all --slurm

Tracking
--------

The tracking command summarizes stage completion without modifying data:

.. code-block:: bash

   q1k-tracking --project-path /path/to/project --redcap-dir /path/to/redcap
