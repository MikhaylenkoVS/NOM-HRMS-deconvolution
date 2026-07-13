"""Unit tests for fragments.py: FRAGMENT_LIBRARY, FUNCTIONAL_GROUPS, MoleculeFragment."""

import pytest
from src.core.fragments import (
    MoleculeFragment,
    FRAGMENT_LIBRARY,
    FUNCTIONAL_GROUPS,
    ALL_FRAGMENTS,
    create_benzene,
    create_cooh,
    create_oh,
    create_methylene,
)


# ===================================================================
# FRAGMENT_LIBRARY
# ===================================================================


class TestFragmentLibrary:
    """Validate structure of FRAGMENT_LIBRARY entries."""

    REQUIRED_KEYS = {"heavy_formula", "ihd", "attachment_points", "description"}

    def test_every_entry_has_required_keys(self):
        for name, data in FRAGMENT_LIBRARY.items():
            missing = self.REQUIRED_KEYS - set(data.keys())
            assert not missing, f"{name}: missing keys {missing}"

    def test_heavy_formula_is_dict_of_positive_ints(self):
        for name, data in FRAGMENT_LIBRARY.items():
            hf = data["heavy_formula"]
            assert isinstance(hf, dict), f"{name}: heavy_formula not dict"
            for el, n in hf.items():
                assert isinstance(n, int) and n > 0, (
                    f"{name}: heavy_formula[{el}] = {n}"
                )

    def test_ihd_is_non_negative_int(self):
        for name, data in FRAGMENT_LIBRARY.items():
            ihd = data["ihd"]
            assert isinstance(ihd, int) and ihd >= 0, f"{name}: ihd = {ihd}"

    def test_attachment_points_is_positive_int(self):
        for name, data in FRAGMENT_LIBRARY.items():
            ap = data["attachment_points"]
            assert isinstance(ap, int) and ap > 0, f"{name}: attachment_points = {ap}"

    def test_description_is_non_empty_string(self):
        for name, data in FRAGMENT_LIBRARY.items():
            desc = data["description"]
            assert isinstance(desc, str) and len(desc) > 0, f"{name}: description empty"

    def test_benzene_present(self):
        assert "benzene" in FRAGMENT_LIBRARY
        assert FRAGMENT_LIBRARY["benzene"]["heavy_formula"] == {"C": 6}
        assert FRAGMENT_LIBRARY["benzene"]["ihd"] == 4
        assert FRAGMENT_LIBRARY["benzene"]["attachment_points"] == 6

    def test_methylene_present(self):
        assert "methylene" in FRAGMENT_LIBRARY
        assert FRAGMENT_LIBRARY["methylene"]["heavy_formula"] == {"C": 1}
        assert FRAGMENT_LIBRARY["methylene"]["ihd"] == 0


# ===================================================================
# FUNCTIONAL_GROUPS
# ===================================================================


class TestFunctionalGroups:
    """Validate structure of FUNCTIONAL_GROUPS entries."""

    REQUIRED_KEYS = {"heavy_formula", "ihd", "description"}

    def test_every_entry_has_required_keys(self):
        for name, data in FUNCTIONAL_GROUPS.items():
            missing = self.REQUIRED_KEYS - set(data.keys())
            assert not missing, f"{name}: missing keys {missing}"

    def test_cooh_present(self):
        assert "cooh" in FUNCTIONAL_GROUPS
        assert FUNCTIONAL_GROUPS["cooh"]["heavy_formula"] == {"C": 1, "O": 2}
        assert FUNCTIONAL_GROUPS["cooh"]["ihd"] == 1

    def test_oh_present(self):
        assert "oh" in FUNCTIONAL_GROUPS
        assert FUNCTIONAL_GROUPS["oh"]["heavy_formula"] == {"O": 1}
        assert FUNCTIONAL_GROUPS["oh"]["ihd"] == 0


# ===================================================================
# MoleculeFragment
# ===================================================================


class TestMoleculeFragment:
    """MoleculeFragment creation, inspection, and connection."""

    def test_create_simple_fragment(self):
        f = MoleculeFragment("test", {"C": 1}, 0, ["C"], [], [0, 0])
        assert f.name == "test"
        assert f.heavy_formula == {"C": 1}
        assert f.ihd == 0
        assert f.get_num_atoms() == 1
        assert len(f.atoms) == 1
        assert len(f.bonds) == 0

    def test_get_free_attachment_points(self):
        f = MoleculeFragment("test", {"C": 1}, 0, ["C"], [], [0, 0])
        free = f.get_free_attachment_points()
        assert free == [0, 0]  # two free bonds on atom 0

    def test_has_free_attachment_point(self):
        f = MoleculeFragment("test", {"C": 1}, 0, ["C"], [], [0, 0])
        assert f.has_free_attachment_point(0) is True
        assert f.has_free_attachment_point(1) is False

    def test_connect_two_fragments(self):
        a = MoleculeFragment("a", {"C": 1}, 0, ["C"], [], [0])
        b = MoleculeFragment("b", {"O": 1}, 0, ["O"], [], [0])

        joined = a.connect_to(b, my_point=0, other_point=0, bond_order=1)
        assert joined.name == "a+b"
        assert joined.heavy_formula == {"C": 1, "O": 1}
        assert joined.get_num_atoms() == 2
        assert len(joined.bonds) == 1
        # The connecting bond should be (0, 1, 1)
        assert (0, 1, 1) in joined.bonds or (0, 1, 1) in joined.bonds

    def test_connect_consumes_attachment_points(self):
        a = MoleculeFragment("a", {"C": 1}, 0, ["C"], [], [0])
        b = MoleculeFragment("b", {"O": 1}, 0, ["O"], [], [0])

        joined = a.connect_to(b, my_point=0, other_point=0, bond_order=1)
        assert joined.get_free_attachment_points() == []

    def test_connect_exhausted_point_raises(self):
        a = MoleculeFragment("a", {"C": 1}, 0, ["C"], [], [0])
        b = MoleculeFragment("b", {"O": 1}, 0, ["O"], [], [0])

        joined = a.connect_to(b, my_point=0, other_point=0, bond_order=1)
        # Both attachment points are exhausted on 'joined'
        with pytest.raises(ValueError, match="уже занята"):
            joined.connect_to(b, my_point=0, other_point=0, bond_order=1)

    def test_repr(self):
        f = MoleculeFragment("test", {"C": 1, "O": 2}, 1, ["C", "O", "O"], [(0, 1, 2), (0, 2, 1)], [0])
        r = repr(f)
        assert "test" in r
        assert "atoms=3" in r

    def test_adjacency_built(self):
        f = MoleculeFragment("test", {"C": 2}, 0, ["C", "C"], [(0, 1, 1)], [0, 1])
        assert 0 in f.adjacency
        assert (1, 1) in f.adjacency[0]
        assert (0, 1) in f.adjacency[1]


# ===================================================================
# Factory functions (create_*)
# ===================================================================


class TestFactoryFunctions:
    """Every key in ALL_FRAGMENTS maps to a callable that returns a MoleculeFragment."""

    def test_all_factories_return_molecule_fragment(self):
        for name, factory in ALL_FRAGMENTS.items():
            result = factory()
            assert isinstance(result, MoleculeFragment), (
                f"create_{name}() did not return MoleculeFragment"
            )
            assert result.name == name, f"create_{name}() name mismatch: {result.name}"

    def test_create_benzene(self):
        b = create_benzene()
        assert b.name == "benzene"
        assert b.heavy_formula == {"C": 6}
        assert b.ihd == 4
        assert b.get_num_atoms() == 6
        assert len(b.bonds) == 6
        assert len(b.get_free_attachment_points()) == 6

    def test_create_cooh(self):
        c = create_cooh()
        assert c.name == "cooh"
        assert c.heavy_formula == {"C": 1, "O": 2}
        assert c.ihd == 1
        assert c.get_num_atoms() == 3

    def test_create_oh(self):
        o = create_oh()
        assert o.name == "oh"
        assert o.heavy_formula == {"O": 1}
        assert o.ihd == 0
        assert o.get_num_atoms() == 1
        assert o.get_free_attachment_points() == [0]

    def test_create_methylene(self):
        m = create_methylene()
        assert m.name == "methylene"
        assert m.heavy_formula == {"C": 1}
        assert m.ihd == 0
        assert m.get_free_attachment_points() == [0, 0]

    def test_chain_connection(self):
        """Build CH3-CH2-OH stepwise."""
        methyl = create_methylene()  # -CH2-
        oh = create_oh()            # -OH

        # First connection: methyl + methyl = ethane-like
        ethane = methyl.connect_to(methyl, my_point=0, other_point=0, bond_order=1)
        assert ethane.get_num_atoms() == 2

        # Second: ethane + OH
        ethanol = ethane.connect_to(oh, my_point=0, other_point=0, bond_order=1)
        assert ethanol.get_num_atoms() == 3
        assert ethanol.heavy_formula == {"C": 2, "O": 1}
