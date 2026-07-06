#!/usr/bin/env python
"""Shift masses in annotation and spectrum CSVs by minus one proton.

Utility for converting neutral masses to ``[M-H]-`` ion masses (or fixing
files stored with the wrong convention) by subtracting the proton mass
from the relevant columns. Runs in dry-run mode unless ``--apply`` is
given, and writes a ``.bak`` backup before overwriting.
"""
import argparse
import pathlib
import sys

import pandas as pd

# Allow running this standalone script from any cwd: put the repo root
# (which contains the ``src`` package) on the import path.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.configs import CHEM

# Proton mass (H+); single source of truth: chemistry.json.
MASS_H = CHEM.proton_mass


def shift_annotations(path: pathlib.Path, dry_run: bool = True) -> None:
    """Subtract the proton mass from an annotations file's mass columns.

    Parameters
    ----------
    path : pathlib.Path
        Path to an ``annotations*.csv`` file; it must have ``mass_obs``
        and ``mass_theor`` columns or it is skipped.
    dry_run : bool, optional
        If ``True`` (default) only print the intended change; if ``False``
        overwrite the file (after writing a ``.bak`` backup).

    Returns
    -------
    None
    """
    print(f"Processing annotations: {path}")
    df = pd.read_csv(path)

    required_cols = {"mass_obs", "mass_theor"}
    missing = required_cols - set(df.columns)
    if missing:
        print(f"  Skipped: missing columns {missing}")
        return

    before_obs = df["mass_obs"].head(3).tolist()
    before_theor = df["mass_theor"].head(3).tolist()

    df["mass_obs"] = df["mass_obs"] - MASS_H
    df["mass_theor"] = df["mass_theor"] - MASS_H

    after_obs = df["mass_obs"].head(3).tolist()
    after_theor = df["mass_theor"].head(3).tolist()

    print("  mass_obs before:", before_obs)
    print("  mass_obs after :", after_obs)
    print("  mass_theor before:", before_theor)
    print("  mass_theor after :", after_theor)

    if dry_run:
        print("  Dry-run mode, file not saved")
        return

    backup = path.with_suffix(path.suffix + ".bak")
    path.rename(backup)
    df.to_csv(path, index=False)
    print(f"  Saved, backup: {backup.name}")


def shift_spectrum(path: pathlib.Path, dry_run: bool = True) -> None:
    """Subtract the proton mass from a spectrum file's ``mass`` column.

    Parameters
    ----------
    path : pathlib.Path
        Path to a spectrum CSV; it must have a ``mass`` column or it is
        skipped.
    dry_run : bool, optional
        If ``True`` (default) only print the intended change; if ``False``
        overwrite the file (after writing a ``.bak`` backup).

    Returns
    -------
    None
    """
    print(f"Processing spectrum: {path}")
    df = pd.read_csv(path)

    if "mass" not in df.columns:
        print("  Skipped: no 'mass' column")
        return

    before = df["mass"].head(3).tolist()
    df["mass"] = df["mass"] - MASS_H
    after = df["mass"].head(3).tolist()

    print("  mass before:", before)
    print("  mass after :", after)

    if dry_run:
        print("  Dry-run mode, file not saved")
        return

    backup = path.with_suffix(path.suffix + ".bak")
    path.rename(backup)
    df.to_csv(path, index=False)
    print(f"  Saved, backup: {backup.name}")


def main(argv=None) -> int:
    """Command-line entry point: shift masses under a directory tree.

    Parameters
    ----------
    argv : list of str or None, optional
        Argument vector; defaults to ``sys.argv`` when ``None``.

    Returns
    -------
    int
        Process exit code (``0`` on success).
    """
    parser = argparse.ArgumentParser(
        description=(
            "Сдвигает массы в annotations и спектрах на -массу протона.\n"
            "По умолчанию работает в dry-run режиме (только показывает изменения)."
        )
    )
    parser.add_argument(
        "root",
        nargs="?",
        default=".",
        help="Корневая директория с тестовыми наборами (по умолчанию текущая).",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Сохранить изменения (иначе только dry-run).",
    )
    args = parser.parse_args(argv)

    root = pathlib.Path(args.root).resolve()
    dry_run = not args.apply

    print(f"Root: {root}")
    print(f"Dry-run: {dry_run}")

    # 1. annotations*.csv
    for path in root.rglob("annotations*.csv"):
        shift_annotations(path, dry_run=dry_run)

    # 2. спектры: original, deutermethylated, deuteroacylated
    spectrum_patterns = [
        "original*.csv",
        "deutermethylated*.csv",
        "deuteroacylated*.csv",
    ]
    for pattern in spectrum_patterns:
        for path in root.rglob(pattern):
            shift_spectrum(path, dry_run=dry_run)

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())