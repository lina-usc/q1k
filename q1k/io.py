from pathlib import Path


def get_project_site_path(site, root=None):
    if root is None:
        root = Path.home() / "projects" / "def-emayada"
    return root / "q1k" / "experimental" / site
    

def get_epoch_path(task, site, root=None):
    project_site_path = get_project_site_path(site, root)
    pylossless_path = project_site_path / "derivatives" / "pylossless"
    postproc_path = pylossless_path / "derivatives" / "postproc"
    return postproc_path / 'epoch_fif_files' / task

    
def get_epoch_files(*args, **kwargs):
    epoch_path = get_epoch_path(*args, **kwargs)
    return list(epoch_path.glob('*epochs.fif'))

