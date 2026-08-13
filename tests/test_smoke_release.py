"""Release-oriented smoke tests for packaging and command-line entry points."""

from importlib.metadata import PackageNotFoundError, distribution, entry_points
from importlib.resources import files

import q1k


def test_package_version_is_defined():
    assert isinstance(q1k.__version__, str)
    assert q1k.__version__


def test_distribution_metadata_when_installed():
    try:
        dist = distribution("q1k")
    except PackageNotFoundError:
        return

    assert dist.metadata["Name"] == "q1k"
    assert "EEG" in dist.metadata["Summary"]


def test_cli_entry_points_are_registered_when_installed():
    expected = {
        "q1k-init",
        "q1k-pylossless",
        "q1k-sync-loss",
        "q1k-segment",
        "q1k-autorej",
        "q1k-tracking",
    }
    scripts = {
        ep.name
        for ep in entry_points(group="console_scripts")
        if ep.value.startswith("q1k.")
    }
    assert expected <= scripts


def test_required_package_data_is_included():
    package_root = files("q1k")
    assert (package_root / "pylossless" / "config.yaml").is_file()
    assert (package_root / "slurm" / "pylossless_job.sh").is_file()
    assert (package_root / "slurm" / "autorej_job.sh").is_file()
