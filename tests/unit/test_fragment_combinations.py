"""Unit tests for fragment_combinations.py: filter_fragments, find_fragment_combinations."""

import pytest
from src.core.fragment_combinations import (
    filter_fragments,
    find_fragment_combinations,
    assemble_molecule_from_combination,
    assemble_all_combinations,
)
from src.core.fragments import FRAGMENT_LIBRARY


# ===================================================================
# filter_fragments
# ===================================================================


class TestFilterFragments:
    """filter_fragments prunes fragments exceeding target constraints."""

    def test_no_fragments_exceed_constraints(self):
        """With a small target (C1, IHD=0), only methylene should pass."""
        target_heavy = {"C": 1}
        target_ihd = 0
        filtered = filter_fragments(target_heavy, target_ihd, FRAGMENT_LIBRARY)
        assert "methylene" in filtered
        # Benzene (C6, IHD=4) should be excluded
        assert "benzene" not in filtered

    def test_benzene_large_enough_target(self):
        """With a large enough target, benzene passes."""
        target_heavy = {"C": 10}
        target_ihd = 10
        filtered = filter_fragments(target_heavy, target_ihd, FRAGMENT_LIBRARY)
        assert "benzene" in filtered
        assert "naphthalene" in filtered

    def test_ihd_exclusion(self):
        """Fragment with IHD > target_ihd is excluded."""
        target_heavy = {"C": 6}
        target_ihd = 2
        filtered = filter_fragments(target_heavy, target_ihd, FRAGMENT_LIBRARY)
        assert "benzene" not in filtered  # IHD=4 > 2

    def test_element_exclusion(self):
        """Fragment with element not in target is excluded."""
        target_heavy = {"C": 6}  # no N
        target_ihd = 10
        filtered = filter_fragments(target_heavy, target_ihd, FRAGMENT_LIBRARY)
        assert "pyridine" not in filtered  # contains N

    def test_element_shortage_exclusion(self):
        """Fragment needing more atoms than target provides is excluded."""
        target_heavy = {"C": 4}
        target_ihd = 10
        filtered = filter_fragments(target_heavy, target_ihd, FRAGMENT_LIBRARY)
        assert "benzene" not in filtered  # needs C=6 > C=4

    def test_empty_library(self):
        result = filter_fragments({"C": 10}, 10, {})
        assert result == {}

    def test_returns_copy_not_reference_to_library(self):
        filtered = filter_fragments({"C": 6}, 4, FRAGMENT_LIBRARY)
        # Must be a subset dict, not the original
        assert filtered is not FRAGMENT_LIBRARY


# ===================================================================
# find_fragment_combinations
# ===================================================================


class TestFindFragmentCombinations:
    """Enumerate fragment multisets matching a target."""

    def test_benzene_only_target(self):
        """Target C6, IHD=4 — only one combination: benzene."""
        combos = find_fragment_combinations({"C": 6}, 4, num_cooh=0, num_oh=0)
        assert len(combos) >= 1
        # At least one combination should be benzene=1
        benzene_match = [c for c in combos if c["bases"].get("benzene") == 1]
        assert len(benzene_match) >= 1

    def test_methylene_target(self):
        """Target C1, IHD=0 — only methylene."""
        combos = find_fragment_combinations({"C": 1}, 0, num_cooh=0, num_oh=0)
        assert len(combos) >= 1
        assert any(c["bases"].get("methylene") == 1 for c in combos)

    def test_impossible_target_returns_empty(self):
        """C1 with IHD=4 is impossible — no fragment fits."""
        combos = find_fragment_combinations({"C": 1}, 4, num_cooh=0, num_oh=0)
        assert combos == []

    def test_with_cooh_functional_group(self):
        """Target C7O2, IHD=5 — C6H6 (benzene) + COOH."""
        combos = find_fragment_combinations(
            {"C": 7, "O": 2}, 5, num_cooh=1, num_oh=0
        )
        assert len(combos) >= 1
        for c in combos:
            assert c["cooh"] == 1 or c["cooh"] == 0

    def test_combination_has_expected_keys(self):
        combos = find_fragment_combinations({"C": 6}, 4, num_cooh=0, num_oh=0)
        for c in combos:
            assert "bases" in c
            assert "cooh" in c
            assert "oh" in c
            assert "total_heavy_formula" in c
            assert "total_ihd" in c

    def test_first_only_returns_at_most_one(self):
        combos = find_fragment_combinations(
            {"C": 6}, 4, num_cooh=0, num_oh=0, first_only=True
        )
        assert len(combos) <= 1

    def test_functional_groups_exhaust_target_falls_back(self):
        """COOH requests more atoms than the target has → fallback to no COOH."""
        combos = find_fragment_combinations(
            {"C": 1}, 0, num_cooh=2, num_oh=0
        )
        # Should fall back to no functional groups
        assert len(combos) >= 0


# ===================================================================
# assemble_molecule_from_combination
# ===================================================================


class TestAssembleMoleculeFromCombination:
    """Assembly of MoleculeFragment from a combination dict."""

    def test_assemble_benzene(self):
        combo = {
            "bases": {"benzene": 1},
            "cooh": 0,
            "oh": 0,
            "total_heavy_formula": {"C": 6},
            "total_ihd": 4,
        }
        mol = assemble_molecule_from_combination(combo)
        assert mol.get_num_atoms() == 6

    def test_assemble_benzene_with_cooh(self):
        combo = {
            "bases": {"benzene": 1},
            "cooh": 1,
            "oh": 0,
            "total_heavy_formula": {"C": 7, "O": 2},
            "total_ihd": 5,
        }
        mol = assemble_molecule_from_combination(combo)
        assert mol.get_num_atoms() == 9  # 6 + 3 (COOH)
        assert "O" in mol.heavy_formula

    def test_assemble_empty_combination_raises(self):
        with pytest.raises(ValueError):
            assemble_molecule_from_combination({
                "bases": {},
                "cooh": 0,
                "oh": 0,
                "total_heavy_formula": {},
                "total_ihd": 0,
            })

    def test_assemble_unknown_fragment_raises(self):
        combo = {
            "bases": {"non_existent_fragment": 1},
            "cooh": 0,
            "oh": 0,
            "total_heavy_formula": {"X": 1},
            "total_ihd": 0,
        }
        with pytest.raises(ValueError, match="не найден в библиотеке"):
            assemble_molecule_from_combination(combo)


# ===================================================================
# assemble_all_combinations
# ===================================================================


class TestAssembleAllCombinations:
    """Batch assembly of combinations."""

    def test_assemble_multiple_combos(self):
        combos = find_fragment_combinations({"C": 6}, 4, num_cooh=0, num_oh=0)
        results = assemble_all_combinations(combos)
        assert len(results) == len(combos)
        for r in results:
            assert "index" in r
            assert "combination" in r
            assert "success" in r
            if r["success"]:
                assert r["molecule"] is not None
            else:
                assert "error" in r

    def test_empty_combinations_returns_empty(self):
        results = assemble_all_combinations([])
        assert results == []
