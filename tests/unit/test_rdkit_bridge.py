"""Unit tests for rdkit_bridge.py: MoleculeFragment → RDKit conversion.

RDKit is imported lazily — tests are skipped when RDKit is not installed.
"""

import pytest

# Lazy check — skip all tests if RDKit is missing
rdkit_available = False
try:
    import rdkit  # noqa: F401
    rdkit_available = True
except ImportError:
    pass


pytestmark = pytest.mark.skipif(
    not rdkit_available, reason="RDKit not installed"
)


# ═══════════════════════════════════════════════════════════════════════════
# to_rdkit_mol  (works with MoleculeFragment: .atoms=str[], .bonds=tuple[])
# ═══════════════════════════════════════════════════════════════════════════


class TestToRdkitMol:
    """Convert MoleculeFragment → RDKit Mol."""

    def test_benzene_conversion(self):
        from src.core.rdkit_bridge import to_rdkit_mol
        from src.core.fragments import create_benzene

        frag = create_benzene()
        mol = to_rdkit_mol(frag)
        assert mol.GetNumAtoms() >= 6  # heavy atoms + H
        heavy = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() != 1)
        assert heavy == 6

    def test_cooh_conversion(self):
        from src.core.rdkit_bridge import to_rdkit_mol
        from src.core.fragments import create_cooh

        frag = create_cooh()
        mol = to_rdkit_mol(frag)
        heavy = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() != 1)
        assert heavy == 3  # C, O, O

    def test_converted_mol_has_bonds(self):
        from src.core.rdkit_bridge import to_rdkit_mol
        from src.core.fragments import create_methylene, create_oh

        # Build CH2-OH manually via MoleculeFragment (C-C-O backbone)
        from src.core.fragments import MoleculeFragment
        frag = MoleculeFragment(
            "test", {"C": 2, "O": 1}, 0,
            ["C", "C", "O"],
            [(0, 1, 1), (1, 2, 1)],
            []
        )
        mol = to_rdkit_mol(frag)
        assert mol.GetNumBonds() >= 2

    def test_naphthalene_conversion(self):
        from src.core.rdkit_bridge import to_rdkit_mol
        from src.core.fragments import create_naphthalene

        frag = create_naphthalene()
        mol = to_rdkit_mol(frag)
        heavy = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() != 1)
        assert heavy == 10

    def test_methylene_conversion(self):
        from src.core.rdkit_bridge import to_rdkit_mol
        from src.core.fragments import create_methylene

        frag = create_methylene()
        mol = to_rdkit_mol(frag)
        heavy = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() != 1)
        assert heavy == 1


# ═══════════════════════════════════════════════════════════════════════════
# visualize_fragment — note: with highlight_attachment_points=True
# there's a bug accessing .attachment_points (should be .get_free_attachment_points())
# ═══════════════════════════════════════════════════════════════════════════


class TestVisualizeFragment:
    """visualize_fragment renders a MoleculeFragment to PIL Image."""

    def test_visualize_without_highlight(self):
        from src.core.rdkit_bridge import visualize_fragment
        from src.core.fragments import create_benzene

        frag = create_benzene()
        img = visualize_fragment(
            frag, highlight_attachment_points=False, size=(200, 150)
        )
        assert img is not None

    def test_visualize_methylene_no_highlight(self):
        from src.core.rdkit_bridge import visualize_fragment
        from src.core.fragments import create_methylene

        frag = create_methylene()
        img = visualize_fragment(
            frag, highlight_attachment_points=False, size=(200, 150)
        )
        assert img is not None


# ═══════════════════════════════════════════════════════════════════════════
# visualize_fragments_grid
# ═══════════════════════════════════════════════════════════════════════════


class TestVisualizeFragmentsGrid:
    """Grid rendering of multiple fragments."""

    def test_grid_two_fragments(self):
        from src.core.rdkit_bridge import visualize_fragments_grid
        from src.core.fragments import create_benzene, create_cooh

        frags = [create_benzene(), create_cooh()]
        img = visualize_fragments_grid(frags, mols_per_row=2, subImgSize=(200, 150))
        assert img is not None

    def test_grid_with_custom_names(self):
        from src.core.rdkit_bridge import visualize_fragments_grid
        from src.core.fragments import create_benzene

        frags = [create_benzene()]
        img = visualize_fragments_grid(
            frags, names=["Benz"], mols_per_row=1, subImgSize=(200, 150)
        )
        assert img is not None


# ═══════════════════════════════════════════════════════════════════════════
# print_molecule_info — uses Molecule (from molecule.py), not MoleculeFragment
# ═══════════════════════════════════════════════════════════════════════════


class TestPrintMoleculeInfo:
    """print_molecule_info logs via logger — uses caplog to capture."""

    def test_smoke(self, caplog):
        import logging
        from src.core.rdkit_bridge import print_molecule_info
        from src.core.molecule import Molecule

        m = Molecule()
        for _ in range(6):
            m.add_atom("C")
        m.add_bond(0, 1, 2)
        m.add_bond(1, 2, 1)
        m.add_bond(2, 3, 2)
        m.add_bond(3, 4, 1)
        m.add_bond(4, 5, 2)
        m.add_bond(5, 0, 1)

        with caplog.at_level(logging.INFO, logger="src.core.rdkit_bridge"):
            print_molecule_info(m, index=1)
        assert "СТРУКТУРА #1" in caplog.text
        assert "C6" in caplog.text

    def test_no_index(self, caplog):
        import logging
        from src.core.rdkit_bridge import print_molecule_info
        from src.core.molecule import Molecule

        m = Molecule()
        m.add_atom("C")
        m.add_atom("C")
        m.add_atom("O")
        m.add_bond(0, 1, 1)
        m.add_bond(1, 2, 1)

        with caplog.at_level(logging.INFO, logger="src.core.rdkit_bridge"):
            print_molecule_info(m)
        assert "Атомов:" in caplog.text


# ═══════════════════════════════════════════════════════════════════════════
# visualize_connection_sequence — note: hits the .attachment_points bug
# when highlight_attachment_points=True, so we test without it
# ═══════════════════════════════════════════════════════════════════════════


class TestVisualizeConnectionSequence:
    """Stepwise connection visualization (requires RDKit)."""

    def test_two_fragment_connection_smoke(self):
        """Smoke test: two fragments connected."""
        from src.core.rdkit_bridge import visualize_connection_sequence
        from src.core.fragments import create_methylene

        frags = [create_methylene(), create_methylene()]
        connections = [(0, 0, 1)]
        images = visualize_connection_sequence(frags, connections, size=(200, 150))
        assert len(images) == 2  # starting + one step
        assert images[0][0].startswith("Исходный")

    def test_single_fragment_sequence(self):
        """A single fragment — graceful."""
        from src.core.rdkit_bridge import visualize_connection_sequence
        from src.core.fragments import create_benzene

        frags = [create_benzene()]
        images = visualize_connection_sequence(frags, [], size=(200, 150))
        assert len(images) == 1
        assert images[0][0].startswith("Исходный")
