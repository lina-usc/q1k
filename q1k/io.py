"""Path utilities for navigating the Q1K project directory structure."""

from pathlib import Path


def get_project_site_path(root=None):
    """Get the project experimental data root path.

    Parameters
    ----------
    root : str or Path, optional
        Override the default root (``~/projects/def-emayada``).

    Returns
    -------
    Path
        ``root / q1k / experimental``
    """
    if root is None:
        root = Path.home() / "projects" / "def-emayada"
    else:
        root = Path(root)
    return root / "q1k" / "experimental"


def get_preproc_path(root=None):
    """Get the pylossless preprocessing derivatives path.

    Returns
    -------
    Path
        ``project_path / derivatives / pylossless``
    """
    return get_project_site_path(root) / "derivatives" / "pylossless"


def get_sync_loss_path(root=None):
    """Get the sync_loss derivatives path.

    Returns
    -------
    Path
        ``pylossless_path / derivatives / sync_loss``
    """
    return get_preproc_path(root) / "derivatives" / "sync_loss"


def get_segment_path(root=None, derivative_base="sync_loss"):
    """Get the segmentation derivatives path.

    Parameters
    ----------
    root : str or Path, optional
        Project root override.
    derivative_base : str
        Which derivative chain to use. ``"sync_loss"`` (default) gives
        the path through sync_loss; ``"postproc"`` gives the legacy
        postproc path.

    Returns
    -------
    Path
        Path to the segment derivatives directory.
    """
    if derivative_base == "sync_loss":
        return get_sync_loss_path(root) / "derivatives" / "segment"
    else:
        return get_preproc_path(root) / "derivatives" / derivative_base


def get_autorej_path(root=None, derivative_base="sync_loss"):
    """Get the autoreject derivatives path.

    Returns
    -------
    Path
        Path to the autorej derivatives directory.
    """
    return get_segment_path(root, derivative_base) / "derivatives" / "autorej"


def get_epoch_path(task, root=None, derivative_base="sync_loss"):
    """Get the epoch files directory for a specific task.

    Parameters
    ----------
    task : str
        Task code (e.g., ``"RS"``, ``"VEP"``).
    root : str or Path, optional
        Project root override.
    derivative_base : str
        Derivative chain to use. Default ``"sync_loss"``.

    Returns
    -------
    Path
        ``segment_path / epoch_fif_files / task``
    """
    return get_segment_path(root, derivative_base) / "epoch_fif_files" / task


def get_epoch_files(*args, file_pattern="*eeg_epo.fif", **kwargs):
    """Get epoch files matching a glob pattern.

    Parameters
    ----------
    *args, **kwargs
        Passed to :func:`get_epoch_path`.
    file_pattern : str
        Glob pattern for epoch files. Default ``"*eeg_epo.fif"``.

    Returns
    -------
    list[Path]
        Matching epoch file paths.
    """
    epoch_path = get_epoch_path(*args, **kwargs)
    return list(epoch_path.glob(file_pattern))


def get_report_path(stage, task, root=None):
    """Get the directory for per-subject report notebooks/HTML.

    Parameters
    ----------
    stage : str
        Pipeline stage name (e.g., ``"init"``, ``"segment"``).
    task : str
        Task code.
    root : str or Path, optional
        Project root override.

    Returns
    -------
    Path
        ``project_path / reports / stage / task``
    """
    project_path = get_project_site_path(root)
    return project_path / "reports" / stage / task


def get_bids_root(root=None):
    """Get the BIDS root directory for raw data.

    Returns
    -------
    Path
        ``project_path`` (the BIDS root is the experimental dir itself).
    """
    return get_project_site_path(root)


def get_tracking_output_path(root=None):
    """Get the tracking output directory.

    Returns
    -------
    Path
        ``project_root / tracking`` (sibling of ``experimental/``).
    """
    return get_project_site_path(root).parent / "tracking"


def get_redcap_path(root=None):
    """Get the REDCap demographics source directory.

    Returns
    -------
    Path
        ``project_root / source / demographics_redcap``
    """
    return get_project_site_path(root).parent / "source" / "demographics_redcap"
