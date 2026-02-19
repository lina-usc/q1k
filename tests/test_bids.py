"""Tests for q1k.bids — ID formatting and BIDS utilities."""

import pytest

from q1k.bids import eb_id_transform, extract_bids_info, format_id, q1k_to_bids


class TestFormatId:
    def test_with_underscore(self):
        assert format_id("42_P") == "0042_P"

    def test_without_underscore(self):
        assert format_id("42P") == "0042_P"

    def test_already_padded(self):
        assert format_id("0042_P") == "0042_P"

    def test_long_number(self):
        assert format_id("12345_P") == "12345_P"

    def test_multi_letter_suffix(self):
        assert format_id("42_F1") == "0042_F1"

    def test_multi_letter_no_underscore(self):
        assert format_id("42F1") == "0042_F1"


class TestQ1kToBids:
    def test_hsj_proband(self):
        assert q1k_to_bids("Q1K_HSJ_10043_P") == "0043P"

    def test_mhc_proband(self):
        assert q1k_to_bids("Q1K_MHC_20042_P") == "0042P"

    def test_nim_family_id(self):
        assert q1k_to_bids("Q1K_NIM_3530-3062_F1") == "3062F1"

    def test_without_q1k_prefix(self):
        # Should still work if Q1K_ prefix is missing
        result = q1k_to_bids("HSJ_10043_P")
        assert result == "0043P"

    def test_hsj_father(self):
        assert q1k_to_bids("Q1K_HSJ_10043_F1") == "0043F1"


class TestEbIdTransform:
    def test_with_q_prefix(self):
        assert eb_id_transform("Q248_P") == "0248_P"

    def test_without_q_prefix(self):
        assert eb_id_transform("281_M1") == "0281_M1"

    def test_no_underscore(self):
        assert eb_id_transform("Q248P") == "0248_P"

    def test_lowercase_q(self):
        assert eb_id_transform("q248_P") == "0248_P"


class TestExtractBidsInfo:
    def test_edf_file(self):
        subject, session, task, run = extract_bids_info(
            "sub-0042P_ses-01_task-RS_run-1_eeg.edf"
        )
        assert subject == "0042P"
        assert session == "01"
        assert task == "RS"
        assert run == "1"

    def test_fif_file(self):
        subject, session, task, run = extract_bids_info(
            "sub-0042P_ses-01_task-VEP_run-1_eeg.fif"
        )
        assert subject == "0042P"
        assert task == "VEP"

    def test_invalid_filename_raises(self):
        with pytest.raises(ValueError, match="does not match BIDS pattern"):
            extract_bids_info("not_a_bids_file.edf")

    def test_missing_extension_raises(self):
        with pytest.raises(ValueError):
            extract_bids_info("sub-0042P_ses-01_task-RS_run-1_eeg.set")
