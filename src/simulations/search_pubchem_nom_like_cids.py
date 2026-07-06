"""Mine NOM-like reference molecules from a random PubChem sample.

Draws random PubChem CIDs, fetches each compound via ``pubchempy``,
filters for neutral molecules carrying 1-10 -COOH and -OH groups with a
mass of 100-600 Da (RDKit substructure counting), and accumulates the
survivors into a reference CSV for building test sets.

Notes
-----
:func:`get_random_cids` is the extension point for CID selection; the
default draws random integers from a heuristic CID range.
"""

from pathlib import Path
from typing import List, Dict
import csv
import random
import pubchempy as pcp
from rdkit import Chem
from rdkit.Chem import Descriptors

SUBPROJECT_ROOT = Path(__file__).resolve().parent.parent
REF_DIR = SUBPROJECT_ROOT / "ref_data"
REF_DIR.mkdir(parents=True, exist_ok=True)

# Файл для set_01 (если нужно по-прежнему отдельный)
REF_PATH_SET01 = REF_DIR / "ref_molecules_set_01_pubchem.csv"

# Общий файл накопления всех кандидатов
REF_PATH_ALL = REF_DIR / "ref_molecules_all_pubchem.csv"


def get_random_cids(max_count: int = 100) -> list[int]:
    """Collect random valid PubChem CIDs.

    Draws random integers from a heuristic CID range, keeping only those
    that resolve to a compound with a non-empty SMILES and molecular
    formula. The number of attempts is capped to bound runtime.

    Parameters
    ----------
    max_count : int, optional
        Number of valid CIDs to collect. Default 100.

    Returns
    -------
    list of int
        Collected CIDs (may be shorter than ``max_count`` if the attempt
        budget is exhausted).
    """

    # Диапазон CID — эвристический (PubChem CID начинаются с 1, верхняя граница условная).
    cid_min = 1
    cid_max = 500_000

    target = max_count
    collected: list[int] = []
    seen: set[int] = set()

    # ограничим общее число попыток, чтобы не зависнуть
    max_attempts = max_count * 20  # например, до 20 попыток на один успешный CID

    attempts = 0
    while len(collected) < target and attempts < max_attempts:
        attempts += 1
        cid = random.randint(cid_min, cid_max)
        if cid in seen:
            continue
        seen.add(cid)

        try:
            comp = pcp.Compound.from_cid(cid)
        except Exception:
            continue

        if comp is None:
            continue

        smiles = comp.canonical_smiles
        formula = comp.molecular_formula
        if not smiles or not formula:
            continue

        collected.append(cid)

    return collected


def fetch_compounds_by_cids(cids: List[int]) -> List[Dict[str, str]]:
    """Fetch raw candidate records for a list of PubChem CIDs.

    Parameters
    ----------
    cids : list of int
        PubChem compound IDs.

    Returns
    -------
    list of dict
        One record per resolvable compound with ``cid``, ``name``,
        ``smiles``, ``formula``, ``exact_mass`` and ``source`` keys.
    """

    candidates: List[Dict[str, str]] = []

    for cid in cids:
        try:
            comp = pcp.Compound.from_cid(cid)
        except Exception:
            continue

        if comp is None:
            continue

        smiles = comp.canonical_smiles
        formula = comp.molecular_formula
        if not smiles or not formula:
            continue

        candidates.append(
            {
                "cid": str(comp.cid),
                "name": comp.iupac_name or (comp.synonyms[0] if comp.synonyms else ""),
                "smiles": smiles,
                "formula": formula,
                "exact_mass": str(comp.exact_mass or ""),
                "source": "PubChem",
            }
        )

    return candidates


def count_carboxyl_and_hydroxyl(smiles: str) -> Dict[str, int]:
    """Count -COOH and -OH groups (and net charge) via RDKit SMARTS.

    Parameters
    ----------
    smiles : str
        SMILES string of the molecule.

    Returns
    -------
    dict of {str: int}
        Keys ``carboxyl_count`` (``[CX3](=O)[OX2H1]`` matches),
        ``hydroxyl_count`` (``[OX2H]`` matches) and ``charge`` (net formal
        charge). All zero if the SMILES cannot be parsed.
    """

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"carboxyl_count": 0, "hydroxyl_count": 0, "charge": 0}

    charge = Chem.GetFormalCharge(mol)

    carboxyl_smarts = "[CX3](=O)[OX2H1]"    # -C(=O)OH
    hydroxyl_smarts = "[OX2H]"              # общие -OH

    carboxyl = mol.GetSubstructMatches(Chem.MolFromSmarts(carboxyl_smarts))
    hydroxyl = mol.GetSubstructMatches(Chem.MolFromSmarts(hydroxyl_smarts))

    return {
        "carboxyl_count": len(carboxyl),
        "hydroxyl_count": len(hydroxyl),
        "charge": charge,
    }


def filter_candidates(
    raw_candidates: List[Dict[str, str]],
    max_take: int = 40,
) -> List[Dict[str, str]]:
    """Filter raw candidates to neutral NOM-like molecules.

    Keeps candidates that are neutral, carry 1-10 -COOH and 1-10 -OH
    groups, and have a mass in 100-600 Da; annotates each kept record with
    the counts, charge and mass.

    Parameters
    ----------
    raw_candidates : list of dict
        Records from :func:`fetch_compounds_by_cids`.
    max_take : int, optional
        Maximum number of candidates to keep. Default 40.

    Returns
    -------
    list of dict
        Filtered and annotated candidate records.
    """

    result: List[Dict[str, str]] = []

    for cand in raw_candidates:
        smiles = cand["smiles"]
        exact_mass = cand.get("exact_mass")

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue

        props = count_carboxyl_and_hydroxyl(smiles)
        carboxyl = props["carboxyl_count"]
        hydroxyl = props["hydroxyl_count"]
        charge = props["charge"]

        if charge != 0:
            continue

        if not (1 <= carboxyl <= 10 and 1 <= hydroxyl <= 10):
            continue

        if exact_mass:
            try:
                mass = float(exact_mass)
            except ValueError:
                mass = Descriptors.MolWt(mol)
        else:
            mass = Descriptors.MolWt(mol)

        if not (100 <= mass <= 600):
            continue

        cand["carboxyl_count"] = carboxyl
        cand["hydroxyl_count"] = hydroxyl
        cand["charge"] = charge
        cand["mass"] = mass

        result.append(cand)

        if len(result) >= max_take:
            break

    return result


def export_candidates_to_ref_csv(
    candidates: List[Dict[str, str]],
    set_id: str = "set_01",
    ref_path: Path | None = None,
) -> None:
    """Append candidates to the shared reference CSV, de-duplicating by CID.

    Reads any existing reference file, adds only candidates whose
    ``pubchem_cid`` is not already present, renumbers ``compound_number``
    and ``compound_id`` sequentially and writes the result back.

    Parameters
    ----------
    candidates : list of dict
        Filtered candidate records from :func:`filter_candidates`.
    set_id : str, optional
        Set identifier stored on new rows. Default ``"set_01"``.
    ref_path : pathlib.Path or None, optional
        Target CSV; defaults to :data:`REF_PATH_ALL`.

    Returns
    -------
    None
        The accumulated reference table is written to ``ref_path``.
    """

    if ref_path is None:
        ref_path = REF_PATH_ALL

    fieldnames = [
        "set_id",
        "compound_id",
        "compound_number",
        "name",
        "smiles",
        "inchi",
        "formula",
        "charge",
        "mode",
        "carboxyl_count",
        "hydroxyl_count",
        "other_fg",
        "nom_like_flag",
        "source",
        "pubchem_cid",
        "comment",
    ]

    # 1. читаем существующие записи (если файл есть)
    existing: list[dict] = []
    existing_cids: set[str] = set()

    if ref_path.exists():
        with ref_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing.append(row)
                cid = row.get("pubchem_cid")
                if cid:
                    existing_cids.add(cid)

    # 2. добавляем новых кандидатов (если pubchem_cid ещё нет)
    for cand in candidates:
        cid = cand.get("cid")
        if not cid:
            continue
        if cid in existing_cids:
            continue

        existing_cids.add(cid)

        existing.append(
            {
                "set_id": set_id,
                "compound_id": f"PC{len(existing_cids):05d}",
                "compound_number": None,  # пока заполним ниже
                "name": cand.get("name") or "",
                "smiles": cand["smiles"],
                "inchi": "",  # при желании можно заполнить
                "formula": cand["formula"],
                "charge": cand["charge"],
                "mode": "neg",
                "carboxyl_count": cand["carboxyl_count"],
                "hydroxyl_count": cand["hydroxyl_count"],
                "other_fg": "",
                "nom_like_flag": True,
                "source": cand.get("source") or "PubChem",
                "pubchem_cid": cid,
                "comment": "NOM-like candidate (random CID)",
            }
        )

    # 3. пере-нумеровываем compound_number по порядку и выравниваем compound_id
    for idx, row in enumerate(existing, start=1):
        row["compound_number"] = idx
        # compound_id можно оставить как есть, если не хотим переписывать;
        # но для единообразия можно переопределить:
        row["compound_id"] = f"PC{idx:05d}"

    # 4. сохраняем обратно
    with ref_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in existing:
            writer.writerow(row)

    print(f"Total accumulated candidates in {ref_path}: {len(existing)}")

def main():
    """Run the full random-sampling and filtering workflow once.

    Returns
    -------
    None
        Fetches random compounds, filters them and appends the survivors
        to :data:`REF_PATH_ALL`.
    """
    cids = get_random_cids(max_count=1000)
    print(f"Got {len(cids)} random CIDs to fetch (example: {cids[:10]})")
    raw = fetch_compounds_by_cids(cids)
    print(f"Raw fetched compounds: {len(raw)}")
    filtered = filter_candidates(raw, max_take=100)
    print(f"Filtered candidates in this run: {len(filtered)}")
    export_candidates_to_ref_csv(filtered, set_id="set_01", ref_path=REF_PATH_ALL)

if __name__ == "__main__":
    main()
