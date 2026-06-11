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

from ..ui.theme import IMG_W, IMG_H

# ═══════════════════════════════════════════════════════════════════════════════
#  ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ═══════════════════════════════════════════════════════════════════════════════

def fragment_to_rdkit(mol_fragment):
    """Конвертирует MoleculeFragment → rdkit.Mol (с 2D-координатами)."""
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
            rw.AddBond(atom_map[a], atom_map[b],
                       bond_types.get(order, Chem.BondType.SINGLE))
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
    if not RDKIT_OK or not PIL_OK or rdmol is None:
        return None
    try:
        img = Draw.MolToImage(rdmol, size=size)
        return img
    except Exception:
        return None


def pil_to_tk(pil_img) -> Optional["ImageTk.PhotoImage"]:
    if not PIL_OK or pil_img is None:
        return None
    return ImageTk.PhotoImage(pil_img)


def save_mol(rdmol, path: str):
    block = Chem.MolToMolBlock(rdmol)
    with open(path, "w", encoding="utf-8") as f:
        f.write(block)


def save_png(rdmol, path: str, size=(800, 600)):
    img = Draw.MolToImage(rdmol, size=size)
    img.save(path)
