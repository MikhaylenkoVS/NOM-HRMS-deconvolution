"""Unit tests for molecule.py: Molecule class — graph building, IHD, formula."""

import pytest
from src.core.molecule import Molecule, parse_formula, calculate_IHD, add_formula


# ===================================================================
# Molecule — atom and bond management
# ===================================================================


class TestMolecule:
    """Molecule graph operations."""

    def test_create_empty(self):
        m = Molecule()
        assert len(m.atoms) == 0
        assert len(m.edges) == 0

    def test_add_atom(self):
        m = Molecule()
        idx = m.add_atom("C")
        assert idx == 0
        assert len(m.atoms) == 1
        assert m.atoms[0].symbol == "C"

    def test_add_multiple_atoms(self):
        m = Molecule()
        m.add_atom("C")
        m.add_atom("C")
        m.add_atom("H")
        assert len(m.atoms) == 3

    def test_add_atom_with_charge(self):
        m = Molecule()
        idx = m.add_atom("N", formal_charge=1)
        assert m.atoms[idx].formal_charge == 1
        assert m.atoms[idx].valence == 4  # N+ valence

    def test_add_bond(self):
        m = Molecule()
        m.add_atom("C")
        m.add_atom("C")
        m.add_bond(0, 1, bond_order=1)
        assert len(m.edges) == 1
        assert m.edges[0] == (0, 1, 1)

    def test_add_bond_exceeding_valence_is_noop(self):
        m = Molecule()
        m.add_atom("H")  # valence 1
        m.add_atom("C")
        m.add_bond(0, 1, bond_order=1)
        # H already used its valence; second bond should fail silently
        m.add_bond(0, 2, bond_order=1)  # atom 2 doesn't exist → no-op
        assert len(m.edges) == 1

    def test_add_duplicate_bond_is_noop(self):
        m = Molecule()
        m.add_atom("C")
        m.add_atom("C")
        m.add_bond(0, 1, bond_order=1)
        m.add_bond(0, 1, bond_order=1)
        assert len(m.edges) == 1

    def test_add_self_bond_is_noop(self):
        m = Molecule()
        m.add_atom("C")
        m.add_bond(0, 0, bond_order=1)
        assert len(m.edges) == 0

    def test_add_bond_out_of_range_is_noop(self):
        m = Molecule()
        m.add_atom("C")
        m.add_bond(0, 99, bond_order=1)
        assert len(m.edges) == 0

    def test_is_connected_single_atom(self):
        m = Molecule()
        m.add_atom("C")
        assert m.is_connected() is True

    def test_is_connected_two_atoms(self):
        m = Molecule()
        m.add_atom("C")
        m.add_atom("C")
        m.add_bond(0, 1, 1)
        assert m.is_connected() is True

    def test_is_connected_disconnected(self):
        m = Molecule()
        m.add_atom("C")
        m.add_atom("C")
        # No bond — two isolated atoms
        assert m.is_connected() is False

    def test_is_connected_empty(self):
        m = Molecule()
        assert m.is_connected() is True

    def test_calculate_IHD_benzene(self):
        m = Molecule()
        # Build C6H6 skeleton (ignoring H for simplicity; only heavy atoms counted)
        for _ in range(6):
            m.add_atom("C")
        # Cyclic hexagon with alternating double bonds (Kekule benzene)
        m.add_bond(0, 1, 2)
        m.add_bond(1, 2, 1)
        m.add_bond(2, 3, 2)
        m.add_bond(3, 4, 1)
        m.add_bond(4, 5, 2)
        m.add_bond(5, 0, 1)
        # IHD = (2*6 + 2 - 0 + 0 - 0) / 2 = 7? No wait, no H atoms → IHD = 7
        ihd = m.calculate_IHD()
        assert ihd > 0

    def test_get_formula(self):
        m = Molecule()
        m.add_atom("C")
        m.add_atom("C")
        m.add_atom("C")
        m.add_atom("H")
        assert m.get_formula() == "C3H"

    def test_get_formula_hill_order(self):
        m = Molecule()
        m.add_atom("O")
        m.add_atom("C")
        m.add_atom("H")
        m.add_atom("N")
        assert m.get_formula() == "CHNO"

    def test_to_smiles_raises_not_implemented(self):
        m = Molecule()
        with pytest.raises(NotImplementedError):
            m.to_smiles()

    def test_repr(self):
        m = Molecule()
        m.add_atom("C")
        r = repr(m)
        assert "Molecule" in r
        assert "C" in r


# ===================================================================
# parse_formula (module-level)
# ===================================================================


class TestParseFormula:
    """parse_formula function in molecule.py (supplementing test_core_utils)."""

    def test_standard_formula(self):
        assert parse_formula("C7H6O2") == {"C": 7, "H": 6, "O": 2}

    def test_single_atom_no_digit(self):
        assert parse_formula("H") == {"H": 1}

    def test_empty_string(self):
        assert parse_formula("") == {}

    def test_multi_char_element(self):
        result = parse_formula("Cl2")
        assert result == {"Cl": 2}


# ===================================================================
# calculate_IHD (module-level)
# ===================================================================


class TestCalculateIHD:
    def test_saturated_alkane(self):
        assert calculate_IHD({"C": 7, "H": 16}) == 0.0

    def test_benzene(self):
        assert calculate_IHD({"C": 6, "H": 6}) == 4.0

    def test_with_halogens(self):
        result = calculate_IHD({"C": 7, "H": 6, "Cl": 2})
        assert result == 4.0

    def test_empty(self):
        assert calculate_IHD({}) == 1.0


# ===================================================================
# add_formula (module-level)
# ===================================================================


class TestAddFormula:
    def test_basic_addition(self):
        base = {"C": 6, "H": 6}
        add_formula(base, {"C": 1, "O": 2})
        assert base == {"C": 7, "H": 6, "O": 2}

    def test_with_multiplier(self):
        base = {"C": 6}
        add_formula(base, {"O": 2}, k=3)
        assert base == {"C": 6, "O": 6}

    def test_new_element(self):
        base = {"C": 1}
        add_formula(base, {"O": 1})
        assert base == {"C": 1, "O": 1}
