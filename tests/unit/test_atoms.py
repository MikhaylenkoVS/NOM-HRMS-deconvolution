"""Unit tests for atoms.py: ELEMENT_DATA, Hybridization, Atom."""

import pytest
from src.core.atoms import ELEMENT_DATA, Hybridization, Atom


# ===================================================================
# ELEMENT_DATA
# ===================================================================


class TestElementData:
    """Validate the periodic-table data used by Atom."""

    SUPPORTED = {"H", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I"}
    CHARGE_AWARE = {"N", "O", "P", "S"}

    def test_all_supported_elements_present(self):
        """Every element in the expected set has an entry."""
        for el in self.SUPPORTED:
            assert el in ELEMENT_DATA, f"Missing element: {el}"

    def test_each_entry_has_atomic_number_and_valence(self):
        """Every element entry must carry atomic_number and valence."""
        for el, data in ELEMENT_DATA.items():
            assert "atomic_number" in data, f"{el}: missing atomic_number"
            assert "valence" in data, f"{el}: missing valence"
            assert isinstance(data["atomic_number"], int), f"{el}: atomic_number not int"
            assert isinstance(data["valence"], int), f"{el}: valence not int"

    def test_charge_aware_elements_have_valence_charged(self):
        """N, O, P, S must expose valence_charged dict."""
        for el in self.CHARGE_AWARE:
            assert "valence_charged" in ELEMENT_DATA[el], (
                f"{el}: missing valence_charged"
            )
            vc = ELEMENT_DATA[el]["valence_charged"]
            assert isinstance(vc, dict), f"{el}: valence_charged not a dict"

    def test_hydrogen_valence_is_1(self):
        assert ELEMENT_DATA["H"]["valence"] == 1

    def test_carbon_valence_is_4(self):
        assert ELEMENT_DATA["C"]["valence"] == 4


# ===================================================================
# Hybridization
# ===================================================================


class TestHybridization:
    """Hybridization enum values."""

    def test_enum_values(self):
        assert Hybridization.SP3.value == "sp3"
        assert Hybridization.SP2.value == "sp2"
        assert Hybridization.SP.value == "sp"
        assert Hybridization.UNKNOWN.value == "unknown"

    def test_enum_membership(self):
        assert isinstance(Hybridization.SP3, Hybridization)
        assert Hybridization("sp2") is Hybridization.SP2


# ===================================================================
# Atom
# ===================================================================


class TestAtom:
    """Atom creation, valence, and bonding."""

    def test_create_carbon(self):
        a = Atom("C", 0)
        assert a.symbol == "C"
        assert a.number == 0
        assert a.valence == 4
        assert a.used_valence == 0
        assert a.hybridization == Hybridization.UNKNOWN
        assert a.is_aromatic is False

    def test_create_nitrogen_neutral(self):
        a = Atom("N", 0, formal_charge=0)
        assert a.valence == 3

    def test_create_nitrogen_charged_minus1(self):
        a = Atom("N", 0, formal_charge=-1)
        assert a.valence == 2

    def test_create_nitrogen_charged_plus1(self):
        a = Atom("N", 0, formal_charge=1)
        assert a.valence == 4

    def test_create_oxygen_neutral(self):
        a = Atom("O", 0)
        assert a.valence == 2

    def test_create_oxygen_charged_minus1(self):
        a = Atom("O", 0, formal_charge=-1)
        assert a.valence == 1

    def test_unsupported_element_raises(self):
        with pytest.raises(ValueError, match="Неподдерживаемый элемент"):
            Atom("Xe", 0)

    def test_add_single_bond(self):
        a = Atom("C", 0)
        assert a.add_bond(1, bond_order=1) is True
        assert 1 in a.connections
        assert a.used_valence == 1
        assert a.get_bond_order_to(1) == 1

    def test_add_duplicate_bond_returns_false(self):
        a = Atom("C", 0)
        a.add_bond(1, bond_order=1)
        assert a.add_bond(1, bond_order=1) is False  # already exists

    def test_exceed_valence_returns_false(self):
        a = Atom("H", 0)  # valence = 1
        a.add_bond(1, bond_order=1)
        assert a.add_bond(2, bond_order=1) is False

    def test_carbon_hybridization_sp3(self):
        a = Atom("C", 0)
        a.add_bond(1, bond_order=1)
        a.add_bond(2, bond_order=1)
        assert a.hybridization == Hybridization.SP3

    def test_carbon_hybridization_sp2_double(self):
        a = Atom("C", 0)
        a.add_bond(1, bond_order=2)
        assert a.hybridization == Hybridization.SP2

    def test_carbon_hybridization_sp2_aromatic(self):
        a = Atom("C", 0)
        a.is_aromatic = True
        a.add_bond(1, bond_order=1)
        assert a.hybridization == Hybridization.SP2

    def test_carbon_hybridization_sp(self):
        a = Atom("C", 0)
        a.add_bond(1, bond_order=3)
        assert a.hybridization == Hybridization.SP

    def test_non_carbon_hybridization_unchanged(self):
        a = Atom("O", 0)
        a.add_bond(1, bond_order=2)
        assert a.hybridization == Hybridization.UNKNOWN

    def test_get_bond_order_to_existing(self):
        a = Atom("C", 0)
        a.add_bond(1, bond_order=2)
        assert a.get_bond_order_to(1) == 2

    def test_get_bond_order_to_non_existent(self):
        a = Atom("C", 0)
        assert a.get_bond_order_to(99) == 0

    def test_repr(self):
        a = Atom("C", 5)
        r = repr(a)
        assert "C" in r
        assert "#5" in r
