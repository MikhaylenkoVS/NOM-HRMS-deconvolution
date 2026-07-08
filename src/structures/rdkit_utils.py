# structures/rdkit_utils.py
from typing import Optional

try:
    from PIL import Image, ImageTk

    PIL_OK = True
except ImportError:
    PIL_OK = False

try:
    from rdkit import Chem
    from rdkit.Chem import Draw, AllChem

    RDKIT_OK = True
except ImportError:
    RDKIT_OK = False

from src.ui import IMG_W, IMG_H

# ═══════════════════════════════════════════════════════════════════════════════
#  ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ═══════════════════════════════════════════════════════════════════════════════


def fragment_to_rdkit(mol_fragment):
    """Convert a ``MoleculeFragment`` to an RDKit ``Mol`` with 2D coordinates.

    Parameters
    ----------
    mol_fragment : MoleculeFragment
        Fragment exposing ``atoms`` and ``bonds``.

    Returns
    -------
    rdkit.Chem.Mol or None
        RDKit molecule with explicit hydrogens and computed 2D
        coordinates, or ``None`` if RDKit is unavailable or conversion
        fails.

    Notes
    -----
    Delegates to :func:`src.core.to_rdkit_mol` first; on failure it falls
    back to a direct atom-by-atom conversion with best-effort
    sanitization.
    """
    if not RDKIT_OK:
        return None
    try:
        from ..core import to_rdkit_mol

        return to_rdkit_mol(mol_fragment)
    except Exception:
        pass
    # Запасной путь — прямая конвертация
    try:
        rw = Chem.RWMol()
        atom_map = {}
        for i, sym in enumerate(mol_fragment.atoms):
            idx = rw.AddAtom(Chem.Atom(sym))
            atom_map[i] = idx
        bond_types = {
            1: Chem.BondType.SINGLE,
            2: Chem.BondType.DOUBLE,
            3: Chem.BondType.TRIPLE,
        }
        for a, b, order in mol_fragment.bonds:
            rw.AddBond(
                atom_map[a], atom_map[b], bond_types.get(order, Chem.BondType.SINGLE)
            )
        mol = rw.GetMol()
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            pass
        mol = Chem.AddHs(mol)
        AllChem.Compute2DCoords(mol)
        return mol
    except Exception:
        return None


def mol_to_pil(rdmol, size=(IMG_W, IMG_H)) -> Optional["Image.Image"]:
    """Render an RDKit molecule to a PIL image.

    Parameters
    ----------
    rdmol : rdkit.Chem.Mol
        Molecule to draw.
    size : tuple of (int, int), optional
        Image size in pixels ``(width, height)``. Defaults to the module
        constants ``(IMG_W, IMG_H)``.

    Returns
    -------
    PIL.Image.Image or None
        Rendered depiction, or ``None`` if RDKit/PIL are unavailable,
        ``rdmol`` is ``None``, or rendering fails.
    """
    if not RDKIT_OK or not PIL_OK or rdmol is None:
        return None
    try:
        img = Draw.MolToImage(rdmol, size=size)
        return img
    except Exception:
        return None


def pil_to_tk(pil_img) -> Optional["ImageTk.PhotoImage"]:
    """Wrap a PIL image as a Tkinter-compatible ``PhotoImage``.

    Parameters
    ----------
    pil_img : PIL.Image.Image
        Image to convert.

    Returns
    -------
    PIL.ImageTk.PhotoImage or None
        Tkinter image, or ``None`` if PIL is unavailable or ``pil_img``
        is ``None``.
    """
    if not PIL_OK or pil_img is None:
        return None
    return ImageTk.PhotoImage(pil_img)


def save_mol(rdmol, path: str):
    """Write an RDKit molecule to an MDL MOL file.

    Parameters
    ----------
    rdmol : rdkit.Chem.Mol
        Molecule to serialize.
    path : str
        Destination file path.

    Returns
    -------
    None
        The MOL block is written to ``path`` as UTF-8 text.
    """
    block = Chem.MolToMolBlock(rdmol)
    with open(path, "w", encoding="utf-8") as f:
        f.write(block)


def save_png(rdmol, path: str, size=(800, 600)):
    """Render an RDKit molecule and save it as a PNG file.

    Parameters
    ----------
    rdmol : rdkit.Chem.Mol
        Molecule to draw.
    path : str
        Destination PNG file path.
    size : tuple of (int, int), optional
        Image size in pixels ``(width, height)``. Default ``(800, 600)``.

    Returns
    -------
    None
        The rendered image is written to ``path``.
    """
    img = Draw.MolToImage(rdmol, size=size)
    img.save(path)
