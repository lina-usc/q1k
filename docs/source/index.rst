SyneQxis Q1K Pipeline
=====================

SyneQxis is the reproducible EEG and eye-tracking preprocessing framework used
for Q1K data processing. The Python package in this repository provides command
line stages for BIDS initialization, PyLossless-based artifact annotation,
EEG/eye-tracking synchronization, task segmentation, AutoReject epoch cleaning,
and processing-stage tracking.

This documentation is intentionally practical: it describes how to install the
package, run the pipeline stages, test the software, and prepare releases.

.. grid:: 1 1 2 2
   :gutter: 2

   .. grid-item-card:: Pipeline stages
      :link: pipeline
      :link-type: doc

      Understand the end-to-end EEG/ET processing flow.

   .. grid-item-card:: Command line use
      :link: cli
      :link-type: doc

      Run each processing stage locally or on HPC.

   .. grid-item-card:: Testing and release
      :link: contributing
      :link-type: doc

      Run tests, coverage, linting, docs builds, and release checks.

   .. grid-item-card:: API reference
      :link: api
      :link-type: doc

      Inspect documented Python modules.

.. toctree::
   :maxdepth: 2
   :hidden:

   installation
   pipeline
   cli
   contributing
   api
