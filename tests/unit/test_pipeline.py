"""Smoke tests for pipeline.py: imports, dataclasses, helper functions."""

import pytest
import pandas as pd
import numpy as np
import tempfile
from pathlib import Path


class TestPipelineImports:
    """Verify critical pipeline symbols can be imported."""

    def test_import_run_pipeline(self):
        from src.core.pipeline import run_pipeline
        assert callable(run_pipeline)

    def test_import_dataclasses(self):
        from src.core.pipeline import (
            PipelineStats,
            PipelineRunResult,
            TestSetResult,
            SeriesStats,
        )

    def test_import_helpers(self):
        from src.core.pipeline import _ppm_error, _normalize_brutto, _match_row_by_mass


class TestPipelineStats:
    """Default values of PipelineStats."""

    def test_defaults(self):
        from src.core.pipeline import PipelineStats

        s = PipelineStats()
        assert s.src_loaded == 0
        assert s.src_denoised == 0
        assert s.assigned_count == 0
        assert s.assigned_ratio == 0.0
        assert s.result_rows == 0
        assert s.result_n_cooh_gt0 == 0
        assert s.result_n_oh_gt0 == 0

    def test_series_stats_defaults(self):
        from src.core.pipeline import SeriesStats

        s = SeriesStats()
        assert s.rows == 0
        assert s.max_groups == 0
        assert s.missing_total == 0


class TestPipelineRunResult:
    """PipelineRunResult dataclass behaviour."""

    def test_construction(self):
        from src.core.pipeline import PipelineRunResult, PipelineStats

        stats = PipelineStats()
        result = PipelineRunResult(table=pd.DataFrame(), stats=stats, messages=[])
        assert result.table is not None
        assert isinstance(result.stats, PipelineStats)
        assert result.messages == []

    def test_default_messages(self):
        from src.core.pipeline import PipelineRunResult, PipelineStats

        result = PipelineRunResult(table=pd.DataFrame(), stats=PipelineStats())
        assert result.messages == []


class TestTestSetResult:
    """TestSetResult dataclass and properties."""

    def test_construction(self):
        from src.core.pipeline import TestSetResult

        r = TestSetResult(set_name="set_01")
        assert r.set_name == "set_01"
        assert r.total_signals == 0
        assert r.denoised_kept == 0
        assert r.assigned_ok == 0
        assert r.dmet_found == 0
        assert r.dmet_matched == 0
        assert r.dmet_wrong == 0
        assert r.dacet_found == 0
        assert r.dacet_matched == 0
        assert r.dacet_wrong == 0
        assert r.errors == []

    def test_denoise_recall_property(self):
        from src.core.pipeline import TestSetResult

        r = TestSetResult(set_name="set_01", total_signals=100, denoised_kept=95)
        assert abs(r.denoise_recall - 0.95) < 1e-9

    def test_denoise_recall_zero_total(self):
        from src.core.pipeline import TestSetResult

        r = TestSetResult(set_name="set_01")
        assert r.denoise_recall == 0.0

    def test_assign_recall_property(self):
        from src.core.pipeline import TestSetResult

        r = TestSetResult(set_name="set_01", total_signals=100, assigned_ok=80)
        assert abs(r.assign_recall - 0.80) < 1e-9

    def test_assign_recall_zero_total(self):
        from src.core.pipeline import TestSetResult

        r = TestSetResult(set_name="set_01")
        assert r.assign_recall == 0.0


class TestNormalizeBrutto:
    """Supplement normalize_brutto tests (import from pipeline)."""

    def test_nan(self):
        from src.core.pipeline import _normalize_brutto
        assert _normalize_brutto(pd.NA) is None

    def test_canonical(self):
        from src.core.pipeline import _normalize_brutto
        assert _normalize_brutto("C7H6O2") == "C7H6O2"

    def test_reorder(self):
        from src.core.pipeline import _normalize_brutto
        assert _normalize_brutto("O2C7H6") == "C7H6O2"


class TestPpmError:
    """_ppm_error from pipeline (cross-check)."""

    def test_zero_theoretical(self):
        from src.core.pipeline import _ppm_error
        assert _ppm_error(100.0, 0.0) == float("inf")

    def test_identical(self):
        from src.core.pipeline import _ppm_error
        assert _ppm_error(200.0, 200.0) == 0.0

    def test_1ppm_at_1000(self):
        from src.core.pipeline import _ppm_error
        result = _ppm_error(1000.001, 1000.0)
        assert abs(result - 1.0) < 0.01
