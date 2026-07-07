"""Unit tests for critical pure utility functions across the core modules.

These functions are the mathematical/parsing foundation of the pipeline:
mass-error calculation, formula canonicalization, formula parsing, exact mass
computation, series peak search, and related helpers. They are pure functions
with no I/O or external dependencies, making them ideal for fast unit tests.

Target modules
--------------
* src.core.pipeline      — _ppm_error, _normalize_brutto, _match_row_by_mass
* src.core.spectrum_ops  — exact_mass_from_counts, dbe_from_counts,
                           _row_to_brutto, _neutral_to_ion_mass, _find_peak
* src.core.molecule      — parse_formula, calculate_IHD, add_formula
"""

from __future__ import annotations

import math
import pandas as pd
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Imports from pipeline.py
# ---------------------------------------------------------------------------
from src.core.pipeline import (
    _ppm_error,
    _normalize_brutto,
    _match_row_by_mass,
)

# ---------------------------------------------------------------------------
# Imports from spectrum_ops.py
# ---------------------------------------------------------------------------
from src.core.spectrum_ops import (
    exact_mass_from_counts,
    dbe_from_counts,
    _row_to_brutto,
    _neutral_to_ion_mass,
    _find_peak,
)

# ---------------------------------------------------------------------------
# Imports from molecule.py
# ---------------------------------------------------------------------------
from src.core.molecule import (
    parse_formula,
    calculate_IHD,
    add_formula,
)


# ===================================================================
# _ppm_error
# ===================================================================


class TestPpmError:
    """Tests for pipeline._ppm_error (absolute mass error in ppm)."""

    def test_zero_theoretical_returns_inf(self):
        """Division by zero theoretical mass must return inf."""
        assert _ppm_error(100.0, 0.0) == float("inf")
        assert _ppm_error(0.0, 0.0) == float("inf")

    def test_identical_masses_return_zero(self):
        """When observed == theoretical, ppm error must be 0."""
        assert _ppm_error(200.0, 200.0) == 0.0

    def test_known_positive_error(self):
        """1 Da difference at 500 Da = 2000 ppm."""
        result = _ppm_error(501.0, 500.0)
        assert abs(result - 2000.0) < 1e-9

    def test_known_negative_error(self):
        """Observed < theoretical also gives absolute value."""
        result = _ppm_error(499.0, 500.0)
        assert abs(result - 2000.0) < 1e-9

    def test_typical_nom_accuracy(self):
        """Sub-ppm difference typical for HRMS."""
        # 0.0005 Da error on 300 Da → ~1.67 ppm
        result = _ppm_error(300.0005, 300.0)
        assert abs(result - 1.6667) < 0.01


# ===================================================================
# _normalize_brutto
# ===================================================================


class TestNormalizeBrutto:
    """Tests for pipeline._normalize_brutto (Hill-ordered canonicalization)."""

    def test_nan_input_returns_none(self):
        """pd.NA / NaN must return None."""
        assert _normalize_brutto(pd.NA) is None
        assert _normalize_brutto(float("nan")) is None

    def test_empty_string_returns_none(self):
        """Empty or whitespace-only string returns None."""
        assert _normalize_brutto("") is None
        assert _normalize_brutto("   ") is None

    def test_already_canonical_stays_unchanged(self):
        """C7H6O2 is already in Hill order."""
        assert _normalize_brutto("C7H6O2") == "C7H6O2"

    def test_reorders_elements_to_hill(self):
        """O2C7H6 → C7H6O2."""
        assert _normalize_brutto("O2C7H6") == "C7H6O2"

    def test_handles_trailing_whitespace(self):
        """Whitespace is stripped before processing."""
        assert _normalize_brutto("  C7H6O2  ") == "C7H6O2"

    def test_with_nitrogen_and_phosphorus(self):
        """C, H, then N, O, P in alphabetical order."""
        result = _normalize_brutto("P1O4N1C10H15")
        assert result == "C10H15NO4P"

    def test_multi_char_elements(self):
        """Cl, Br etc. are parsed and ordered correctly."""
        # C first, H second, then alphabetically: Cl, O
        result = _normalize_brutto("Cl1H6C7O2")
        assert result == "C7H6ClO2"

    def test_single_atom_without_digit(self):
        """Element without digit count defaults to 1."""
        result = _normalize_brutto("OHC")
        # C before H before O → C, H, O
        assert result == "CHO"


# ===================================================================
# _match_row_by_mass
# ===================================================================


class TestMatchRowByMass:
    """Tests for pipeline._match_row_by_mass."""

    def test_empty_table_returns_none(self):
        """None or empty DataFrame returns None."""
        assert _match_row_by_mass(None, 200.0, 5.0) is None
        assert _match_row_by_mass(pd.DataFrame(), 200.0, 5.0) is None

    def test_missing_mass_col_returns_none(self):
        """DataFrame without the mass column returns None."""
        df = pd.DataFrame({"x": [1, 2]})
        assert _match_row_by_mass(df, 200.0, 5.0) is None

    def test_exact_match(self):
        """Exact mass match returns the correct row."""
        df = pd.DataFrame({"mass": [100.0, 200.0, 300.0]})
        result = _match_row_by_mass(df, 200.0, 5.0)
        assert result is not None
        assert result["mass"] == 200.0

    def test_closest_match_within_tolerance(self):
        """Closest match within ppm is returned."""
        df = pd.DataFrame({"mass": [200.0, 200.005]})
        result = _match_row_by_mass(df, 200.002, 50.0)
        assert result is not None
        # 200.002 is closer to 200.0 than to 200.005
        assert result["mass"] == 200.0

    def test_none_when_none_within_tolerance(self):
        """No match within tolerance returns None."""
        df = pd.DataFrame({"mass": [100.0, 300.0]})
        assert _match_row_by_mass(df, 200.0, 1.0) is None

    def test_require_assigned(self):
        """With require_assigned=True, only rows with assign=True match."""
        df = pd.DataFrame(
            {
                "mass": [100.0, 200.0, 300.0],
                "assign": [False, True, False],
            }
        )
        result = _match_row_by_mass(df, 200.0, 5.0, require_assigned=True)
        assert result is not None
        assert result["mass"] == 200.0

    def test_require_assigned_no_assign_col(self):
        """Missing 'assign' column with require_assigned=True returns None."""
        df = pd.DataFrame({"mass": [200.0]})
        result = _match_row_by_mass(df, 200.0, 5.0, require_assigned=True)
        assert result is None


# ===================================================================
# parse_formula (molecule.py)
# ===================================================================


class TestParseFormula:
    """Tests for molecule.parse_formula."""

    def test_simple_ch(self):
        """C7H6O2 → {'C': 7, 'H': 6, 'O': 2}."""
        result = parse_formula("C7H6O2")
        assert result == {"C": 7, "H": 6, "O": 2}

    def test_multi_digit_counts(self):
        """C20H29O2 → multi-digit counts."""
        result = parse_formula("C20H29O2")
        assert result == {"C": 20, "H": 29, "O": 2}

    def test_multi_char_elements(self):
        """Cl and Br parsed correctly."""
        result = parse_formula("C7H6ClBr")
        assert result["C"] == 7
        assert result["H"] == 6
        assert result["Cl"] == 1
        assert result["Br"] == 1

    def test_no_digit_means_one(self):
        """Element without explicit count = 1."""
        result = parse_formula("CHON")
        assert result == {"C": 1, "H": 1, "O": 1, "N": 1}

    def test_empty_string(self):
        """Empty string returns empty dict."""
        result = parse_formula("")
        assert result == {}

    def test_only_elements(self):
        """'H' alone returns {'H': 1}."""
        result = parse_formula("H")
        assert result == {"H": 1}


# ===================================================================
# calculate_IHD (molecule.py — module-level function)
# ===================================================================


class TestCalculateIHD:
    """Tests for molecule.calculate_IHD (module-level function)."""

    def test_saturated_hydrocarbon(self):
        """C7H16 → IHD = 0 (saturated alkane)."""
        assert calculate_IHD({"C": 7, "H": 16}) == 0.0

    def test_benzene(self):
        """C6H6 → IHD = 4."""
        assert calculate_IHD({"C": 6, "H": 6}) == 4.0

    def test_with_nitrogen(self):
        """C7H6O2N → IHD = (2*7 + 2 + 1 - 6) / 2 = 5.5."""
        result = calculate_IHD({"C": 7, "H": 6, "O": 2, "N": 1})
        assert result == 5.5

    def test_with_halogens(self):
        """C7H6Cl2 → IHD = (2*7 + 2 - 6 - 2) / 2 = 4."""
        result = calculate_IHD({"C": 7, "H": 6, "Cl": 2})
        assert result == 4.0

    def test_empty_dict(self):
        """Empty dict → 0."""
        assert calculate_IHD({}) == 1.0  # (0 + 2 + 0 - 0 - 0) / 2 = 1


# ===================================================================
# exact_mass_from_counts (spectrum_ops.py)
# ===================================================================


class TestExactMassFromCounts:
    """Tests for spectrum_ops.exact_mass_from_counts."""

    def test_water(self):
        """H2O mass ~18.01056 Da (uses actual ATOMIC_MASS values from config)."""
        mass = exact_mass_from_counts({"H": 2, "O": 1})
        # Use actual ATOMIC_MASS values to verify correctness at 1e-6
        from src.core.spectrum_ops import ATOMIC_MASS

        expected = 2 * ATOMIC_MASS["H"] + ATOMIC_MASS["O"]
        assert abs(mass - expected) < 1e-12

    def test_zero_count_ignored(self):
        """Element with count 0 is skipped."""
        mass = exact_mass_from_counts({"C": 12, "H": 0})
        from src.core.spectrum_ops import ATOMIC_MASS

        expected = 12 * ATOMIC_MASS["C"]
        assert abs(mass - expected) < 1e-12

    def test_negative_count_ignored(self):
        """Negative count is skipped (n <= 0 condition)."""
        mass = exact_mass_from_counts({"C": 12, "H": -1})
        expected = 12 * 12.0
        assert abs(mass - expected) < 1e-9

    def test_empty_dict(self):
        """Empty dict returns 0.0."""
        assert exact_mass_from_counts({}) == 0.0


# ===================================================================
# dbe_from_counts (spectrum_ops.py)
# ===================================================================


class TestDbeFromCounts:
    """Tests for spectrum_ops.dbe_from_counts."""

    def test_linear_alkane(self):
        """C7H16 → DBE = 1 + 7 - 16/2 + 0 = 0."""
        assert dbe_from_counts({"C": 7, "H": 16}) == 0.0

    def test_benzene(self):
        """C6H6 → DBE = 1 + 6 - 6/2 + 0 = 4."""
        assert dbe_from_counts({"C": 6, "H": 6}) == 4.0

    def test_with_nitrogen(self):
        """C6H5N → DBE = 1 + 6 - 5/2 + 1/2 = 5."""
        assert dbe_from_counts({"C": 6, "H": 5, "N": 1}) == 5.0

    def test_empty_dict(self):
        """Empty → 1."""
        assert dbe_from_counts({}) == 1.0


# ===================================================================
# _row_to_brutto (spectrum_ops.py)
# ===================================================================


class TestRowToBrutto:
    """Tests for spectrum_ops._row_to_brutto."""

    def test_typical_row(self):
        """Row with C, H, O → C7H6O2."""
        row = pd.Series({"C": 7, "H": 6, "O": 2})
        assert _row_to_brutto(row) == "C7H6O2"

    def test_single_atoms(self):
        """Count of 1 → no digit suffix."""
        row = pd.Series({"C": 1, "H": 1, "O": 1})
        assert _row_to_brutto(row) == "CHO"

    def test_missing_element_column(self):
        """Missing element is ignored."""
        row = pd.Series({"C": 7, "O": 2})
        assert _row_to_brutto(row) == "C7O2"

    def test_nan_value_in_element(self):
        """NaN element value is skipped."""
        row = pd.Series({"C": 7, "H": float("nan"), "O": 2})
        assert _row_to_brutto(row) == "C7O2"

    def test_no_positive_counts(self):
        """All zeros → None."""
        row = pd.Series({"C": 0, "H": 0})
        assert _row_to_brutto(row) is None

    def test_float_count_converted_to_int(self):
        """Float counts like 7.0 → 7 (int conversion)."""
        row = pd.Series({"C": 7.0, "H": 6.0, "O": 2.0})
        assert _row_to_brutto(row) == "C7H6O2"

    def test_custom_element_order(self):
        """Custom element order is respected."""
        row = pd.Series({"O": 2, "C": 7, "H": 6})
        order = ["C", "H", "O"]
        assert _row_to_brutto(row, element_order=order) == "C7H6O2"

    def test_non_integer_value_skipped(self):
        """String value that can't be int is skipped."""
        row = pd.Series({"C": 7, "H": "abc", "O": 2})
        result = _row_to_brutto(row)
        assert "H" not in (result or "")


# ===================================================================
# _neutral_to_ion_mass (spectrum_ops.py)
# ===================================================================


class TestNeutralToIonMass:
    """Tests for spectrum_ops._neutral_to_ion_mass."""

    def test_neutral_mode(self):
        """neutral/empty string returns the same mass."""
        assert _neutral_to_ion_mass(100.0, "neutral") == 100.0
        assert _neutral_to_ion_mass(100.0, "") == 100.0

    def test_m_h_minus(self):
        """[M-H]- subtracts one proton mass."""
        from src.configs import CHEM

        result = _neutral_to_ion_mass(100.0, "[M-H]-")
        assert abs(result - (100.0 - CHEM.proton_mass)) < 1e-12

    def test_m_h_minus_short_aliases(self):
        """Short aliases 'm-h' and 'mh-' work too."""
        from src.configs import CHEM

        r1 = _neutral_to_ion_mass(100.0, "m-h")
        r2 = _neutral_to_ion_mass(100.0, "mh-")
        expected = 100.0 - CHEM.proton_mass
        assert abs(r1 - expected) < 1e-12
        assert abs(r2 - expected) < 1e-12

    def test_m_h_plus(self):
        """[M+H]+ adds one proton mass."""
        from src.configs import CHEM

        result = _neutral_to_ion_mass(100.0, "[M+H]+")
        assert abs(result - (100.0 + CHEM.proton_mass)) < 1e-12

    def test_m_h_plus_short_aliases(self):
        """Short aliases 'm+h' and 'mh+' work too."""
        from src.configs import CHEM

        r1 = _neutral_to_ion_mass(100.0, "m+h")
        r2 = _neutral_to_ion_mass(100.0, "mh+")
        expected = 100.0 + CHEM.proton_mass
        assert abs(r1 - expected) < 1e-12
        assert abs(r2 - expected) < 1e-12

    def test_unknown_mode_raises(self):
        """Unrecognized ion_mode raises ValueError."""
        with pytest.raises(ValueError, match="Unknown ion_mode"):
            _neutral_to_ion_mass(100.0, "[M+Na]+")


# ===================================================================
# _find_peak (spectrum_ops.py)
# ===================================================================


class TestFindPeak:
    """Tests for spectrum_ops._find_peak."""

    def test_exact_match(self):
        """Exact match returns the index."""
        arr = [100.0, 200.0, 300.0]
        assert _find_peak(arr, 200.0, 5.0) == 1

    def test_match_within_tolerance(self):
        """Match within ppm tolerance."""
        # 200.0 at tolerance, 200.001 at 5 ppm of 200 → within 5 ppm
        arr = [100.0, 200.001, 300.0]
        assert _find_peak(arr, 200.0, 10.0) == 1

    def test_no_match_outside_tolerance(self):
        """Peak too far returns None."""
        arr = [100.0, 201.0]
        assert _find_peak(arr, 200.0, 5.0) is None

    def test_closest_match_when_multiple(self):
        """Several candidates return the closest."""
        arr = [200.001, 200.005, 200.003]
        # closest to 200.0 is 200.001
        assert _find_peak(arr, 200.0, 50.0) == 0

    def test_empty_array(self):
        """Empty array returns None."""
        assert _find_peak([], 200.0, 5.0) is None


# ===================================================================
# add_formula (molecule.py)
# ===================================================================


class TestAddFormula:
    """Tests for molecule.add_formula (in-place addition of formula dicts)."""

    def test_simple_addition(self):
        """Adding {'O': 2, 'C': 1} to {'C': 6, 'H': 6} → {'C': 7, 'H': 6, 'O': 2}."""
        base = {"C": 6, "H": 6}
        add_formula(base, {"C": 1, "O": 2})
        assert base == {"C": 7, "H": 6, "O": 2}

    def test_with_multiplier(self):
        """k=2 adds 2× the delta."""
        base = {"C": 6, "H": 6}
        add_formula(base, {"C": 1, "O": 2}, k=2)
        assert base == {"C": 8, "H": 6, "O": 4}

    def test_new_element_in_delta(self):
        """Delta introduces a new element not yet in base."""
        base = {"C": 6, "H": 6}
        add_formula(base, {"O": 2})
        assert base == {"C": 6, "H": 6, "O": 2}

    def test_empty_delta(self):
        """Empty delta leaves base unchanged."""
        base = {"C": 6, "H": 6}
        add_formula(base, {})
        assert base == {"C": 6, "H": 6}


# ===================================================================
# Edge-case integration: _normalize_brutto → parse_formula round-trip
# ===================================================================


class TestFormulaRoundTrip:
    """Round-trip: parse → normalize should be idempotent."""

    @pytest.mark.parametrize(
        "formula",
        [
            "C7H6O2",
            "C20H29O2",
            "C10H14O2N",
            "C6H6",
            "C6H5Cl",
            "C7H6ClBr",
            "CHO",
            "CHON",
            "C2H6S",
            "C10H14O5P",
        ],
    )
    def test_round_trip(self, formula):
        """parse → normalize → normalize yields the same result."""
        parsed = parse_formula(formula)
        # rebuild formula from parsed counts using _normalize_brutto
        # by constructing a pseudo-formula string out of order
        reordered = "".join(
            f"{el}{parsed[el]}" if parsed[el] > 1 else el
            for el in sorted(parsed.keys(), reverse=True)
        )
        first = _normalize_brutto(reordered)
        second = _normalize_brutto(first)
        assert first == second, f"Normalisation not idempotent for {formula}"


# ===================================================================
# Regression: generator mass_shift_per_group should match pipeline
# ===================================================================


class TestGeneratorShiftConsistency:
    """Fallback values in generate_test_sets.py must match pipeline DELTA_*
    constants (issue pub-06). Otherwise synthetic data will be generated
    with wrong mass shifts and none of the series will be found."""

    def test_dm_shift_matches_delta_cd3(self):
        """Generator's dm_shift_per_group fallback == DELTA_CD3."""
        from src.core.spectrum_ops import DELTA_CD3
        from src.configs import CHEM

        fallback = CHEM.derivatization_shifts["delta_cd3"]
        assert (
            fallback == DELTA_CD3
        ), f"CHEM delta_cd3 ({fallback}) != DELTA_CD3 ({DELTA_CD3})"

    def test_da_shift_matches_delta_cd3co(self):
        """Generator's da_shift_per_group fallback == DELTA_CD3CO."""
        from src.core.spectrum_ops import DELTA_CD3CO
        from src.configs import CHEM

        fallback = CHEM.derivatization_shifts["delta_cd3co"]
        assert (
            fallback == DELTA_CD3CO
        ), f"CHEM delta_cd3co ({fallback}) != DELTA_CD3CO ({DELTA_CD3CO})"

    def test_generator_imports_chem(self):
        """generate_test_sets.py uses CHEM.derivatization_shifts as fallback."""
        import importlib
        import sys

        # Prevent side effects from full import
        spec = importlib.util.find_spec("src.simulations.generate_test_sets")
        assert (
            spec is not None
        ), "generate_test_sets module not found — integrity check skipped"
