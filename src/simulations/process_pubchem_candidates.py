"""Filter PubChem candidates into balanced NOM-like reference sets.

Reads a raw PubChem export, keeps neutral CHNO-only molecules, computes
Van Krevelen ratios (H/C, O/C) and a NOM-like flag, plots the diagram and
splits the survivors into five balanced ``molecules.csv`` test sets.
"""
from pathlib import Path
from typing import Dict
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt

SUBPROJECT_ROOT = Path(__file__).resolve().parent.parent
REF_DIR = SUBPROJECT_ROOT / "ref_data"
INPUT_PATH = REF_DIR / "ref_molecules_all_pubchem.csv"
OUTPUT_FILTERED_PATH = REF_DIR / "ref_molecules_all_pubchem_filtered.csv"
OUTPUT_VK_PATH = REF_DIR / "ref_molecules_all_pubchem_filtered.csv"
VK_PLOT_PATH = REF_DIR / "van_krevelen_nom_like.png"

def parse_formula(formula: str) -> Dict[str, int]:
    """Parse a brutto formula into an ``{element: count}`` dict.

    Parameters
    ----------
    formula : str
        Molecular formula without brackets or charges.

    Returns
    -------
    dict of {str: int}
        Element counts; empty dict for empty/non-string input.

    Raises
    ------
    RuntimeError
        If the formula has an unparsed trailing segment.
    """

    if not isinstance(formula, str) or not formula:
        return {}

    pattern = re.compile(r"([A-Z][a-z]?)(\d*)")
    pos = 0
    composition: Dict[str, int] = {}

    for match in pattern.finditer(formula):
        element, count_str = match.groups()
        count = int(count_str) if count_str else 1
        composition[element] = composition.get(element, 0) + count
        pos = match.end()

    if pos != len(formula):
        raise RuntimeError(f"есть непарсенный хвост в {formula}")
        # есть непарсенный хвост (можно логировать/игнорировать)
        return {}

    return composition


def is_chno_only(formula: str) -> bool:
    """Return whether a formula contains only C, H, N and O.

    Parameters
    ----------
    formula : str
        Molecular formula.

    Returns
    -------
    bool
        ``True`` if every element is one of C/H/N/O.

    Raises
    ------
    ValueError
        If the formula cannot be parsed.
    """

    comp = parse_formula(formula)
    if not comp:
        raise ValueError(f"{formula} can't be parsed")
        return False

    allowed = {"C", "H", "N", "O"}
    return all(elem in allowed for elem in comp.keys())

def filter_candidates() -> pd.DataFrame:
    """Keep neutral, CHNO-only candidates from the raw PubChem export.

    Returns
    -------
    pandas.DataFrame
        Filtered candidates; also written to
        :data:`OUTPUT_FILTERED_PATH`.
    """
    df = pd.read_csv(INPUT_PATH)

    # приведём charge к числу
    def to_int_or_zero(x):
        try:
            return int(x)
        except Exception:
            return 0

    df["charge"] = df["charge"].apply(to_int_or_zero)

    # масса: используем столбец mass, если он есть;
    # если его нет, можно временно пропустить фильтр по массе или пересчитать отдельно.
    if "mass" in df.columns:
        mass = df["mass"]
    else:
        mass = pd.Series([None] * len(df))

    df["mass_val"] = mass

    # фильтр по массе
    mask_mass = True

    # фильтр по заряду: нейтральные молекулы
    mask_charge = df["charge"] == 0

    # фильтр по составу CHNO
    mask_chno = df["formula"].apply(is_chno_only)

    filtered = df[mask_mass & mask_charge & mask_chno].copy()

    print(f"Total rows: {len(df)}, after filters (mass+charge+CHNO): {len(filtered)}")

    filtered.to_csv(OUTPUT_FILTERED_PATH, index=False)
    return filtered


def add_van_krevelen_and_nom_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Add H/C, O/C ratios and a Van Krevelen NOM-like flag.

    Parameters
    ----------
    df : pandas.DataFrame
        Candidates with a ``formula`` column.

    Returns
    -------
    pandas.DataFrame
        Input frame with added ``H_C``, ``O_C``, ``nom_like_flag_vk`` and
        ``nom_like_flag_combined`` columns; also written to
        :data:`OUTPUT_VK_PATH`.

    Notes
    -----
    A molecule is flagged NOM-like when ``0.2 <= O/C <= 0.7`` and
    ``0.7 <= H/C <= 1.5`` (approximate humic/fulvic ranges).
    """
    comps = df["formula"].apply(parse_formula)

    c = comps.apply(lambda d: d.get("C", 0))
    h = comps.apply(lambda d: d.get("H", 0))
    o = comps.apply(lambda d: d.get("O", 0))

    # избежим деления на ноль
    df["H_C"] = h / c.replace(0, pd.NA)
    df["O_C"] = o / c.replace(0, pd.NA)

    # простейшее определение NOM-like по ван-Кревелену:
    # например, берём классические границы для гумино-/фульвокислот:
    #   0.2 <= O/C <= 0.7, 0.7 <= H/C <= 1.5 (примерные значения)
    def is_nom_like_row(row):
        oc = row["O_C"]
        hc = row["H_C"]
        if pd.isna(oc) or pd.isna(hc):
            return False
        return (0.2 <= oc <= 0.7) and (0.7 <= hc <= 1.5)

    df["nom_like_flag_vk"] = df.apply(is_nom_like_row, axis=1)

    # при желании можно объединить с существующим nom_like_flag (логическое И/ИЛИ)
    if "nom_like_flag" in df.columns:
        df["nom_like_flag_combined"] = df["nom_like_flag"] & df["nom_like_flag_vk"]
    else:
        df["nom_like_flag_combined"] = df["nom_like_flag_vk"]

    df.to_csv(OUTPUT_VK_PATH, index=False)
    return df

def plot_van_krevelen(df: pd.DataFrame) -> None:
    """Plot and save a Van Krevelen (H/C vs O/C) diagram.

    Parameters
    ----------
    df : pandas.DataFrame
        Candidates with ``H_C``, ``O_C`` and ``nom_like_flag_vk`` columns.

    Returns
    -------
    None
        The figure is written to :data:`VK_PLOT_PATH`.
    """
    plt.figure(figsize=(6, 6))

    # выделим nom_like и не-nom_like разными цветами
    nom = df[df["nom_like_flag_vk"]]
    non_nom = df[~df["nom_like_flag_vk"]]

    plt.scatter(non_nom["O_C"], non_nom["H_C"], s=10, alpha=0.3, label="non-NOM-like")
    plt.scatter(nom["O_C"], nom["H_C"], s=20, alpha=0.8, label="NOM-like (VK)")

    plt.xlabel("O/C")
    plt.ylabel("H/C")
    plt.title("Van Krevelen diagram for PubChem candidates")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(VK_PLOT_PATH, dpi=200)
    plt.close()

def debug_input():
    """Print a quick summary of the raw PubChem input file.

    Returns
    -------
    None
        Column names, the first rows of ``formula``/``charge`` and mass
        statistics (when present) are printed to stdout.
    """
    df = pd.read_csv(INPUT_PATH)
    print("Columns:", df.columns.tolist())
    print(df[["formula", "charge"]].head(20))

    # если есть mass / exact_mass — покажем
    if "mass" in df.columns:
        print("mass describe:\n", df["mass"].describe())
    if "exact_mass" in df.columns:
        print("exact_mass describe:\n", df["exact_mass"].describe())

def split_into_sets(df: pd.DataFrame, n_sets: int = 5) -> list[pd.DataFrame]:
    """Split candidates into balanced subsets by round-robin dealing.

    Parameters
    ----------
    df : pandas.DataFrame
        Candidates with ``O_C``, ``H_C`` and ``mass_val`` columns.
    n_sets : int, optional
        Number of subsets. Default 5.

    Returns
    -------
    list of pandas.DataFrame
        One frame per subset, balanced by sorting on O/C, H/C and mass
        then dealing rows cyclically.
    """
    df = df.copy()

    # выберем параметры для балансировки
    # сортируем по O/C, затем по H/C, затем по массе
    df_sorted = df.sort_values(["O_C", "H_C", "mass_val"]).reset_index(drop=True)

    # перемешаем блоками: будем раздавать по кругу
    sets = [ [] for _ in range(n_sets) ]

    for idx, row in df_sorted.iterrows():
        set_idx = idx % n_sets
        sets[set_idx].append(row)

    set_dfs = [pd.DataFrame(rows) for rows in sets]
    return set_dfs

def save_sets(set_dfs: list[pd.DataFrame]) -> None:
    """Write each subset to ``data/test_sets/set_0i/molecules.csv``.

    Parameters
    ----------
    set_dfs : list of pandas.DataFrame
        Subsets from :func:`split_into_sets`.

    Returns
    -------
    None
        Each subset is re-indexed with fresh ``set_id``,
        ``compound_number`` and ``compound_id`` values before saving.
    """
    for i, sdf in enumerate(set_dfs, start=1):
        set_id = f"set_0{i}"
        out_path = SUBPROJECT_ROOT / f"data/test_sets/{set_id}" / "molecules.csv"

        # обновим set_id и compound_number 1..N внутри каждого набора
        sdf = sdf.copy()
        sdf["set_id"] = set_id
        sdf = sdf.reset_index(drop=True)
        sdf["compound_number"] = sdf.index + 1
        sdf["compound_id"] = sdf["compound_number"].apply(lambda x: f"{set_id.upper()}_{x:03d}")

        sdf.to_csv(out_path, index=False)
        print(f"Saved {len(sdf)} rows to {out_path}")

if __name__ == "__main__":
    filtered_df = filter_candidates()
    vk_df = add_van_krevelen_and_nom_flag(filtered_df)
    plot_van_krevelen(vk_df)

    set_dfs = split_into_sets(vk_df, n_sets=5)
    save_sets(set_dfs)