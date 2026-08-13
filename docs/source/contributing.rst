Testing, CI, and Releases
=========================

Local quality checks
--------------------

Run the same basic checks that GitHub Actions runs:

.. code-block:: bash

   python -m pip install -e ".[test,doc,dev]"
   ruff check q1k/bids.py q1k/config.py q1k/io.py q1k/slurm.py q1k/autorej q1k/segment/tasks.py q1k/tracking tests
   codespell q1k/bids.py q1k/config.py q1k/io.py q1k/slurm.py q1k/autorej q1k/segment/tasks.py q1k/tracking tests docs README.md
   coverage run -m pytest
   coverage report
   sphinx-build -M html docs/source docs/_build

Pre-commit hooks
----------------

Install local hooks once:

.. code-block:: bash

   pre-commit install

Then run all hooks manually when needed:

.. code-block:: bash

   pre-commit run --all-files

Continuous integration
----------------------

The repository includes GitHub Actions workflows for:

* tests and coverage on Linux and macOS;
* linting and spell checking;
* documentation builds;
* package build checks;
* PyPI publication when a GitHub release is published.

Release checklist
-----------------

1. Confirm tests, linting, docs, and build workflows pass on ``main``.
2. Update ``q1k.__version__`` and ``project.version`` in ``pyproject.toml``.
3. Create a GitHub release with the version tag.
4. PyPI publishing runs through GitHub Trusted Publishing.

PyPI setup
----------

Before the first release, configure PyPI Trusted Publishing for the GitHub
repository and the ``pypi`` environment used by the release workflow.
