"""Unit tests for isotope filter: _beynon_m1_ratio, _measure_m1_ratio."""

import pytest
import numpy as np
import pandas as pd
from src.core.spectrum_ops import _beynon_m1_ratio


class TestBeynonM1Ratio:
    """_beynon_m1_ratio: theoretical (M+1)/M from Beynon formula."""

    def test_c7h6o2(self):
        """C7H6O2: 7×1.1 + 6×0.015 + 2×0.04 = 7.87% → 0.0787."""
        ratio = _beynon_m1_ratio({"C": 7, "H": 6, "O": 2})
        expected = (7 * 1.1 + 6 * 0.015 + 2 * 0.04) / 100.0
        assert abs(ratio - expected) < 1e-9

    def test_pure_carbon(self):
        """C60: 60×1.1% = 0.66."""
        ratio = _beynon_m1_ratio({"C": 60})
        expected = 60 * 1.1 / 100.0
        assert abs(ratio - expected) < 1e-9

    def test_with_nitrogen(self):
        """C6H5N: 6×1.1 + 5×0.015 + 1×0.37 = 7.045%."""
        ratio = _beynon_m1_ratio({"C": 6, "H": 5, "N": 1})
        expected = (6 * 1.1 + 5 * 0.015 + 1 * 0.37) / 100.0
        assert abs(ratio - expected) < 1e-9

    def test_empty_dict(self):
        assert _beynon_m1_ratio({}) == 0.0

    def test_unknown_elements_ignored(self):
        """S, P, Cl etc. don't contribute to Beynon."""
        ratio = _beynon_m1_ratio({"C": 10, "S": 2, "Cl": 1})
        expected = 10 * 1.1 / 100.0
        assert abs(ratio - expected) < 1e-9

    def test_ratio_increases_with_mass(self):
        """Larger molecule → higher M+1/M."""
        small = _beynon_m1_ratio({"C": 1})
        large = _beynon_m1_ratio({"C": 50})
        assert large > small

    def test_methane(self):
        """CH4: 1×1.1 + 4×0.015 = 1.16%."""
        ratio = _beynon_m1_ratio({"C": 1, "H": 4})
        assert abs(ratio - 0.0116) < 0.001


class TestIsotopePenaltyThreshold:
    """Verify the 20% deviation threshold is correctly calculated."""

    def _compute_penalty(self, m1_theor, m1_real):
        """Simulate the penalty logic from assign_formulas."""
        from src.core.spectrum_ops import _ISOTOPE_TOLERANCE, _ISOTOPE_PENALTY
        dev = abs(m1_real - m1_theor) / m1_theor
        return _ISOTOPE_PENALTY if dev > _ISOTOPE_TOLERANCE else 0.0

    def test_exact_match_no_penalty(self):
        assert self._compute_penalty(0.078, 0.078) == 0.0

    def test_within_10_percent_no_penalty(self):
        """10% deviation < 20% threshold."""
        assert self._compute_penalty(0.100, 0.110) == 0.0

    def test_exceeds_20_percent_triggers_penalty(self):
        """25% deviation > 20% threshold."""
        assert self._compute_penalty(0.100, 0.126) == 2.0

    def test_large_deviation_penalty(self):
        """M+1/M differs by factor of 2."""
        assert self._compute_penalty(0.050, 0.100) == 2.0
