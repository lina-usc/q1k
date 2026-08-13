"""Tests for q1k.io — path utility functions."""

from pathlib import Path

from q1k.io import (
    get_autorej_path,
    get_bids_root,
    get_epoch_path,
    get_preproc_path,
    get_project_site_path,
    get_redcap_path,
    get_segment_path,
    get_sync_loss_path,
    get_tracking_output_path,
)


def test_project_site_path_default():
    p = get_project_site_path()
    assert p == Path.home() / "projects" / "def-emayada" / "q1k" / "experimental"


def test_project_site_path_custom_root(tmp_path):
    p = get_project_site_path(root=tmp_path)
    assert p == tmp_path / "q1k" / "experimental"


def test_preproc_path_ends_with_pylossless(tmp_path):
    p = get_preproc_path(root=tmp_path)
    assert p.parts[-2:] == ("derivatives", "pylossless")


def test_sync_loss_path(tmp_path):
    p = get_sync_loss_path(root=tmp_path)
    assert p.parts[-2:] == ("derivatives", "sync_loss")


def test_segment_path_sync_loss(tmp_path):
    p = get_segment_path(root=tmp_path, derivative_base="sync_loss")
    assert p.parts[-2:] == ("derivatives", "segment")


def test_segment_path_postproc(tmp_path):
    p = get_segment_path(root=tmp_path, derivative_base="postproc")
    assert p.parts[-1] == "postproc"


def test_autorej_path(tmp_path):
    p = get_autorej_path(root=tmp_path)
    assert p.parts[-2:] == ("derivatives", "autorej")


def test_epoch_path(tmp_path):
    p = get_epoch_path("RS", root=tmp_path)
    assert p.parts[-2:] == ("epoch_fif_files", "RS")


def test_epoch_path_vep(tmp_path):
    p = get_epoch_path("VEP", root=tmp_path)
    assert p.parts[-1] == "VEP"


def test_bids_root(tmp_path):
    p = get_bids_root(root=tmp_path)
    assert p == get_project_site_path(root=tmp_path)


def test_tracking_output_path(tmp_path):
    p = get_tracking_output_path(root=tmp_path)
    assert p.parts[-1] == "tracking"


def test_redcap_path(tmp_path):
    p = get_redcap_path(root=tmp_path)
    assert p.parts[-1] == "demographics_redcap"
    assert p.parts[-2] == "source"
