# ============================================================
# src/testing/structure_export.py
# ============================================================
"""
Headless экспорт структур с защитой от зависаний и комбинаторного взрыва.
Использует find_and_visualize_molecules с ограниченным числом базовых фрагментов.
Время выполнения одного соединения ограничено таймаутом (ThreadPoolExecutor).
PNG‑генерация может быть отключена для ускорения.
"""

from pathlib import Path
from typing import List, Optional
import traceback
import warnings
import logging
import concurrent.futures

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Настройки (можно переопределить через переменные окружения)
# ----------------------------------------------------------------------
MAX_BASES = 3                # максимальное число базовых фрагментов
STRUCTURE_TIMEOUT = 20        # секунд на одно соединение
EXPORT_PNG = False           # генерировать PNG (медленно)

# ----------------------------------------------------------------------
# Проверка зависимостей
# ----------------------------------------------------------------------
try:
    from src.core.fragment_combinations import find_and_visualize_molecules
    _HAS_FRAGMENT_SEARCH = True
except ImportError as e:
    warnings.warn(f"Fragment search not available: {e}")
    _HAS_FRAGMENT_SEARCH = False

try:
    from rdkit import Chem
    from rdkit.Chem import Draw, AllChem
    _HAS_RDKIT = True
except ImportError:
    _HAS_RDKIT = False

# ----------------------------------------------------------------------
def export_structures_for_compound(
    brutto: str,
    n_cooh: int,
    n_oh: int,
    output_dir: Path,
    max_bases: int = MAX_BASES,
    timeout_sec: int = STRUCTURE_TIMEOUT,
) -> List[Path]:
    """
    Экспортирует структуры с таймаутом.
    Возвращает список созданных путей (может быть пустым при ошибке/таймауте).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                _export_structures_impl,
                brutto, n_cooh, n_oh, output_dir, max_bases
            )
            return future.result(timeout=timeout_sec)
    except concurrent.futures.TimeoutError:
        logger.warning(f"TIMEOUT ({timeout_sec}s) while exporting structures for {brutto}")
        return []
    except Exception as e:
        logger.warning(f"Structure export failed for {brutto}: {e}")
        return []

def _export_structures_impl(
    brutto: str,
    n_cooh: int,
    n_oh: int,
    output_dir: Path,
    max_bases: int,
) -> List[Path]:
    """Реализация без таймаута (выполняется в отдельном потоке)."""
    generated = []
    if not _HAS_FRAGMENT_SEARCH:
        raise RuntimeError("Fragment search module not available")

    # 1. Поиск комбинаций фрагментов
    try:
        result = find_and_visualize_molecules(
            brutto,
            num_cooh=n_cooh,
            num_oh=n_oh,
            max_bases=max_bases,
            show_images=False
        )
    except Exception as e:
        raise RuntimeError(f"Fragment search failed: {e}") from e

    molecules = result.get('molecules', [])
    logger.info(f"Found {len(molecules)} structures for {brutto}")

    # 2. Экспорт каждой структуры
    for i, mol_info in enumerate(molecules):
        try:
            frag = mol_info.get('fragment_object')
            if frag is None:
                continue
            # Сохраняем .mol
            rdmol = _fragment_to_rdkit(frag)
            if rdmol is not None:
                mol_path = output_dir / f"structure_{i+1}.mol"
                Chem.MolToMolFile(rdmol, str(mol_path))
                generated.append(mol_path)

                # Опционально PNG
                if EXPORT_PNG and _HAS_RDKIT:
                    png_path = output_dir / f"structure_{i+1}.png"
                    _save_png(rdmol, png_path)
                    generated.append(png_path)
        except Exception as exc:
            logger.warning(f"Failed to export structure {i+1} for {brutto}: {exc}")

    return generated

def _fragment_to_rdkit(fragment):
    """Конвертирует MoleculeFragment в rdkit.Mol (headless, безопасно)."""
    if not _HAS_RDKIT:
        return None
    try:
        from src.core.rdkit_bridge import to_rdkit_mol
        return to_rdkit_mol(fragment)
    except Exception:
        # Fallback: ручная сборка
        try:
            rw = Chem.RWMol()
            atom_map = {}
            for idx, sym in enumerate(fragment.atoms):
                atom = Chem.Atom(sym)
                atom_idx = rw.AddAtom(atom)
                atom_map[idx] = atom_idx
            for a, b, order in fragment.bonds:
                bond_type = {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE,
                             3: Chem.BondType.TRIPLE}.get(order, Chem.BondType.SINGLE)
                rw.AddBond(atom_map[a], atom_map[b], bond_type)
            mol = rw.GetMol()
            # Санитизация
            try:
                Chem.SanitizeMol(mol, sanitizeOps=Chem.SANITIZE_ALL ^ Chem.SANITIZE_PROPERTIES)
            except Exception:
                pass
            mol = Chem.AddHs(mol)
            AllChem.Compute2DCoords(mol)
            return mol
        except Exception as e:
            logger.debug(f"RDKit fallback failed: {e}")
            return None

def _save_png(rdmol, path: Path):
    """Сохраняет PNG изображение молекулы."""
    if not _HAS_RDKIT:
        return
    try:
        img = Draw.MolToImage(rdmol, size=(400, 300))
        img.save(str(path))
    except Exception as e:
        logger.debug(f"PNG export failed: {e}")