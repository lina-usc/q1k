from pathlib import Path


def get_project_site_path(root=None):
    if root is None:
        root = Path.home() / "projects" / "def-emayada"
    else:
        root = Path(root)
    return root / "q1k" / "experimental" 


def get_preproc_path(root=None):
    project_site_path = get_project_site_path(root)
    return  project_site_path / "derivatives" / "pylossless"


def get_epoch_path(task, root=None):
    pylossless_path = get_preproc_path(root)
    postproc_path = pylossless_path / "derivatives" / "sync_loss" / "derivatives" / "segment"
    return postproc_path / 'epoch_fif_files' / task


def get_epoch_files(*args, file_pattern="*eeg_epo.fif", **kwargs):
    epoch_path = get_epoch_path(*args, **kwargs)
    return list(epoch_path.glob(file_pattern))
