"""Tests for CLI argument parsing across all pipeline stages."""

import pytest


class TestInitCli:
    def test_parser_required_args(self):
        from q1k.init.cli import create_parser

        parser = create_parser()
        args = parser.parse_args([
            "--project-path", "/data/exp",
            "--task", "RS",
            "--subject", "0042P",
        ])
        assert args.project_path == "/data/exp"
        assert args.task == "RS"
        assert args.subject == "0042P"

    def test_parser_defaults(self):
        from q1k.init.cli import create_parser

        parser = create_parser()
        args = parser.parse_args([
            "--project-path", "/data/exp",
            "--task", "VEP",
            "--all",
        ])
        assert args.session == "01"
        assert args.run == "1"
        assert args.site == "HSJ"
        assert args.process_all is True

    def test_parser_invalid_task(self):
        from q1k.init.cli import create_parser

        parser = create_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--project-path", "/data", "--task", "INVALID"])


class TestPylosslessCli:
    def test_parser_required_args(self):
        from q1k.pylossless.cli import create_parser

        parser = create_parser()
        args = parser.parse_args([
            "--project-path", "/data/exp",
            "--task", "RS",
            "--subject", "0042P",
        ])
        assert args.project_path == "/data/exp"
        assert args.slurm is False

    def test_parser_slurm_flag(self):
        from q1k.pylossless.cli import create_parser

        parser = create_parser()
        args = parser.parse_args([
            "--project-path", "/data/exp",
            "--task", "RS",
            "--subject", "0042P",
            "--slurm",
        ])
        assert args.slurm is True


class TestSyncLossCli:
    def test_parser_required_args(self):
        from q1k.sync_loss.cli import create_parser

        parser = create_parser()
        args = parser.parse_args([
            "--project-path", "/data/exp",
            "--task", "VEP",
            "--subject", "0042P",
        ])
        assert args.task == "VEP"


class TestSegmentCli:
    def test_parser_required_args(self):
        from q1k.segment.cli import create_parser

        parser = create_parser()
        args = parser.parse_args([
            "--project-path", "/data/exp",
            "--task", "GO",
            "--subject", "0042P",
        ])
        assert args.task == "GO"
        assert args.derivative_base == "sync_loss"

    def test_parser_derivative_base(self):
        from q1k.segment.cli import create_parser

        parser = create_parser()
        args = parser.parse_args([
            "--project-path", "/data/exp",
            "--task", "RS",
            "--subject", "0042P",
            "--derivative-base", "postproc",
        ])
        assert args.derivative_base == "postproc"


class TestAutorejCli:
    def test_parser_required_args(self):
        from q1k.autorej.cli import create_parser

        parser = create_parser()
        args = parser.parse_args([
            "--project-path", "/data/exp",
            "--task", "RS",
            "--subject", "0042P",
        ])
        assert args.task == "RS"
        assert args.slurm is False

    def test_parser_slurm_and_derivative(self):
        from q1k.autorej.cli import create_parser

        parser = create_parser()
        args = parser.parse_args([
            "--project-path", "/data/exp",
            "--task", "VEP",
            "--all",
            "--slurm",
            "--derivative-base", "postproc",
        ])
        assert args.slurm is True
        assert args.derivative_base == "postproc"
        assert args.process_all is True


class TestTrackingCli:
    def test_parser_required_args(self):
        from q1k.tracking.cli import create_parser

        parser = create_parser()
        args = parser.parse_args([
            "--project-path", "/data/exp",
            "--redcap-dir", "/data/redcap",
        ])
        assert args.project_path == "/data/exp"
        assert args.redcap_dir == "/data/redcap"
        assert args.task is None
        assert args.sharepoint is None

    def test_parser_with_task(self):
        from q1k.tracking.cli import create_parser

        parser = create_parser()
        args = parser.parse_args([
            "--project-path", "/data/exp",
            "--redcap-dir", "/data/redcap",
            "--task", "RS",
        ])
        assert args.task == "RS"

    def test_parser_sharepoint_mode(self):
        from q1k.tracking.cli import create_parser

        parser = create_parser()
        args = parser.parse_args([
            "--project-path", "/data/exp",
            "--redcap-dir", "/data/redcap",
            "--sharepoint", "/data/sharepoint.xlsx",
            "--mni-upload-date", "2024-06-01",
            "--hsj-upload-date", "2024-07-01",
        ])
        assert args.sharepoint == "/data/sharepoint.xlsx"
        assert args.mni_upload_date == "2024-06-01"
        assert args.hsj_upload_date == "2024-07-01"
