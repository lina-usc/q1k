# SyneQxis/Q1K Release Engineering Plan

This repository now has the first release-quality software engineering layer
modeled on the PyLossless project.

## Implemented

- Unit and smoke tests with `pytest`.
- Coverage reporting with `coverage.py` and Codecov-ready XML output.
- Focused Ruff linting for the stable public package surface.
- Codespell spell checking for stable package files, tests, docs, and README.
- Pre-commit hooks for local lint/spell checks.
- Sphinx documentation source and local HTML build.
- ReadTheDocs configuration.
- Package build validation for sdist and wheel.
- Trusted Publishing-ready PyPI release workflow.
- AutoReject CLI batch/Slurm support and a module entry point for the Slurm wrapper.

## Verified Locally

```bash
python3 -m pytest tests
python3 -m coverage run -m pytest
python3 -m coverage report
ruff check q1k/bids.py q1k/config.py q1k/io.py q1k/slurm.py q1k/autorej q1k/segment/tasks.py q1k/tracking tests
codespell q1k/bids.py q1k/config.py q1k/io.py q1k/slurm.py q1k/autorej q1k/segment/tasks.py q1k/tracking tests docs README.md
sphinx-build -M html docs/source docs/_build
python3 -m build --sdist --wheel --outdir dist .
twine check dist/*
pre-commit run --all-files
```

Current focused release-surface test status:

- `90 passed`
- coverage gate: `37%`, with CI fail-under set to `35%`

## GitHub Setup Still Needed

1. Enable GitHub Actions for the repository.
2. Add branch protection so pull requests must pass:
   - Tests and Coverage
   - Style, Spellcheck, and Pre-commit
   - Build Documentation
   - Build Package
3. Connect the repository to Codecov.
   - If Codecov requires it, add `CODECOV_TOKEN` as a repository secret.
4. Create or connect the ReadTheDocs project.
   - Use `.readthedocs.yaml`.
   - Expected docs source is `docs/source/conf.py`.
5. Configure PyPI Trusted Publishing.
   - PyPI project: `q1k` unless the package is renamed.
   - GitHub environment name: `pypi`.
   - Workflow file: `.github/workflows/pypi.yml`.

## Next Hardening Phase

- Decide whether the public package name remains `q1k` or becomes `syneqxis`.
- Clean legacy scripts in `q1k/init`, `q1k/sync_loss`, `q1k/pylossless`, and manager files.
- Expand Ruff and codespell gates to the full repository after legacy cleanup.
- Add smoke tests for one synthetic end-to-end mini-pipeline path.
- Add tests around PyLossless and AutoReject wrappers using mocks rather than real EEG files.
- Raise coverage threshold gradually: 35% -> 50% -> 70%.
