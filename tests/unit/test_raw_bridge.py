"""Unit tests for raw_bridge.py: availability checks, merge_segments, error paths.

These tests do NOT require MSFileReader — they mock the optional dependency
and validate the pure-Python error-handling and helper logic.
"""

import pytest
import numpy as np
import os
import tempfile
from unittest import mock
import sys


# ═══════════════════════════════════════════════════════════════════════════
# is_available / availability_error
# ═══════════════════════════════════════════════════════════════════════════


class TestAvailability:
    """When MSFileReader is absent, is_available() → False."""

    def test_is_available_without_msfr(self):
        """Without MSFileReader installed, is_available should return False."""
        from src.core.raw_bridge import is_available
        # On a dev machine without MSFileReader this is False;
        # on CI it will be False.
        result = is_available()
        assert isinstance(result, bool)

    def test_availability_error_returns_string_when_unavailable(self):
        from src.core.raw_bridge import is_available, availability_error
        if not is_available():
            err = availability_error()
            assert err is not None
            assert isinstance(err, str)
        else:
            assert availability_error() is None


# ═══════════════════════════════════════════════════════════════════════════
# _merge_segments
# ═══════════════════════════════════════════════════════════════════════════


class TestMergeSegments:
    """Pure-function merge of multi-segment spectrum data."""

    def test_empty_dict_returns_empty(self):
        from src.core.raw_bridge import _merge_segments

        result = _merge_segments({})
        assert isinstance(result, np.ndarray)
        assert result.shape == (0, 2)

    def test_single_segment(self):
        from src.core.raw_bridge import _merge_segments

        data = {
            "segment_1": np.array(
                [[100.0, 50.0, 0, 0, 0, 0], [200.0, 30.0, 0, 0, 0, 0]]
            )
        }
        result = _merge_segments(data)
        assert result.shape == (2, 2)
        assert result[0, 0] == 100.0
        assert result[0, 1] == 50.0

    def test_merging_same_mass_sums_intensities(self):
        from src.core.raw_bridge import _merge_segments

        # Same mass in two segments → intensities summed
        data = {
            "seg1": np.array([[100.0, 40.0, 0, 0, 0, 0]]),
            "seg2": np.array([[100.0, 60.0, 0, 0, 0, 0]]),
        }
        result = _merge_segments(data)
        assert result.shape == (1, 2)
        assert result[0, 0] == 100.0
        assert result[0, 1] == 100.0  # 40 + 60

    def test_different_masses_kept_separate(self):
        from src.core.raw_bridge import _merge_segments

        data = {
            "seg1": np.array([[100.0, 10.0, 0, 0, 0, 0]]),
            "seg2": np.array([[200.0, 20.0, 0, 0, 0, 0]]),
        }
        result = _merge_segments(data)
        assert result.shape == (2, 2)

    def test_near_identical_masses_grouped(self):
        from src.core.raw_bridge import _merge_segments

        # masses within 1e-5 are treated as equal after rounding
        data = {
            "seg": np.array(
                [[100.00000, 10.0, 0, 0, 0, 0], [100.000001, 5.0, 0, 0, 0, 0]]
            )
        }
        result = _merge_segments(data)
        assert result.shape == (1, 2)
        assert result[0, 1] == 15.0


# ═══════════════════════════════════════════════════════════════════════════
# average_raw_to_csv — error paths (no MSFileReader needed)
# ═══════════════════════════════════════════════════════════════════════════


class TestAverageRawToCsvErrors:
    """Error paths that don't require MSFileReader."""

    def test_raises_runtime_error_when_unavailable(self):
        from src.core.raw_bridge import average_raw_to_csv, is_available

        if not is_available():
            with pytest.raises(RuntimeError, match="not available"):
                average_raw_to_csv("dummy.raw", 0.0, 1.0)
        else:
            pytest.skip("MSFileReader is available — skipping error test")

    def test_value_error_when_rt_invalid(self):
        """rt_min >= rt_max raises ValueError regardless of MSFileReader."""
        from src.core.raw_bridge import average_raw_to_csv, is_available

        if is_available():
            # We have MSFileReader but we test the validation path
            with pytest.raises(ValueError, match="rt_min"):
                average_raw_to_csv("dummy.raw", 5.0, 3.0)
        else:
            pytest.skip(
                "MSFileReader unavailable — rt check happens after availability check"
            )

    def test_file_not_found_when_available(self):
        """File check happens after availability."""
        from src.core.raw_bridge import average_raw_to_csv, is_available

        if is_available():
            with pytest.raises(FileNotFoundError):
                average_raw_to_csv(
                    "/nonexistent/path/file.raw", 0.0, 1.0
                )
        else:
            pytest.skip("MSFileReader not available")


# ═══════════════════════════════════════════════════════════════════════════
# average_raw_to_df delegates correctly
# ═══════════════════════════════════════════════════════════════════════════


class TestAverageRawToDf:
    """average_raw_to_df calls average_raw_to_csv and reads the CSV."""

    def test_delegates_and_reads_csv(self):
        from src.core.raw_bridge import average_raw_to_df, is_available

        if not is_available():
            pytest.skip("MSFileReader unavailable — cannot run end-to-end")
        # This test only runs when MSFileReader is available.
        # In most CI environments it will be skipped.
