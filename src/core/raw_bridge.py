"""ThermoRAW → averaged CSV bridge (optional, requires MSFileReader + comtypes).

This module wraps the external ``PyMSFileReader`` (GPL-3.0) to average mass
spectra from a ThermoRAW file over a retention-time window and export the
result as a CSV compatible with the ``load_spectrum()`` pipeline step.

Usage
-----
    from src.core.raw_bridge import average_raw_to_csv

    csv_path = average_raw_to_csv("sample.raw", rt_min=0.0, rt_max=30.0)

Requirements
------------
* Windows only (COM-dependent).
* ``MSFileReader 3.1 SP4`` library installed.
* ``comtypes`` Python package.

License note
------------
This module is GPL-3.0-licensed.  At runtime it imports GPL-3.0 code from
``external/usrednenie_spectrov_i_hromatogramm/src/``.  The combined work is
subject to the GPL-3.0 when the optional dependency is active.
"""

from __future__ import annotations

import os
import sys
import csv
from typing import Callable, Optional

import pandas as pd
import numpy as np


# ── lazy import of PyMSFileReader ────────────────────────────────────────────

_EXTERNAL_ROOT = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "external",
    "usrednenie_spectrov_i_hromatogramm",
    "src",
)

_MSFR_AVAILABLE = False
_MSFR_ERROR: Optional[str] = None

try:
    if _EXTERNAL_ROOT not in sys.path:
        sys.path.insert(0, _EXTERNAL_ROOT)
    import pymsfilereader as _msfr  # type: ignore[import-untyped]

    _MSFR_AVAILABLE = True
except ImportError as err:
    _MSFR_ERROR = str(err)


def is_available() -> bool:
    """Check whether ThermoRAW processing is available on this machine."""
    return _MSFR_AVAILABLE


def availability_error() -> Optional[str]:
    """Return a human-readable error if RAW support is unavailable, else None."""
    return _MSFR_ERROR if not _MSFR_AVAILABLE else None


# ── core API ─────────────────────────────────────────────────────────────────


def average_raw_to_csv(
    raw_path: str,
    rt_min: float,
    rt_max: float,
    output_csv: Optional[str] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> str:
    """Average a ThermoRAW file over [rt_min, rt_max] and write a CSV.

    The output CSV contains ``mass,intensity`` columns and is compatible
    with :func:`src.core.spectrum_ops.load_spectrum`.

    Parameters
    ----------
    raw_path : str
        Path to the ``.raw`` file.
    rt_min : float
        Start of retention-time window (minutes).
    rt_max : float
        End of retention-time window (minutes).  Must be > *rt_min*.
    output_csv : str or None, optional
        Where to write the CSV.  If ``None``, a temporary file is created
        in the same directory as *raw_path* (named ``<basename>_avrg.csv``).
    progress_callback : callable or None, optional
        If given, called with a short status string at key stages
        (e.g. ``"Усреднение спектров…"``). Useful for GUI progress
        indication.

    Returns
    -------
    str
        Absolute path to the CSV file that was written.

    Raises
    ------
    RuntimeError
        If ``MSFileReader`` or ``comtypes`` are not available.
    ValueError
        If *rt_min* >= *rt_max*.
    """
    if not _MSFR_AVAILABLE:
        raise RuntimeError(
            "ThermoRAW averaging is not available:\n"
            f"{_MSFR_ERROR}\n\n"
            "Install MSFileReader 3.1 SP4 + comtypes package on Windows."
        )

    if rt_min >= rt_max:
        raise ValueError(f"rt_min ({rt_min}) must be < rt_max ({rt_max})")

    if not os.path.isfile(raw_path):
        raise FileNotFoundError(f"RAW file not found: {raw_path}")

    # ── open RAW file ────────────────────────────────────────────────────
    if progress_callback:
        progress_callback("Усреднение спектров…")
    xrf = _msfr.PyMSFileReader(raw_path)
    c_double = type(_msfr.c_double(0))  # resolve ctypes type

    # ── average ─────────────────────────────────────────────────────────
    spectrum_lists = xrf.get_averaged_spectrum_list_from_RT(
        start_rt=c_double(rt_min),
        end_rt=c_double(rt_max),
    )

    # Merge all scan-type segments into one spectrum
    if progress_callback:
        progress_callback("Объединение сегментов…")
    merged = _merge_segments(spectrum_lists)

    # ── write CSV ───────────────────────────────────────────────────────
    if progress_callback:
        progress_callback("Запись CSV…")
    if output_csv is None:
        base = os.path.splitext(os.path.basename(raw_path))[0]
        out_dir = os.path.dirname(raw_path) or "."
        output_csv = os.path.join(out_dir, f"{base}_avrg.csv")

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["mass", "intensity"])
        for row in merged:
            writer.writerow([f"{row[0]:.6f}", f"{row[1]:.2f}"])

    return os.path.abspath(output_csv)


def average_raw_to_df(
    raw_path: str,
    rt_min: float,
    rt_max: float,
) -> pd.DataFrame:
    """Average a ThermoRAW file and return a DataFrame with ``mass``, ``intensity`` columns."""
    csv_path = average_raw_to_csv(raw_path, rt_min, rt_max)
    return pd.read_csv(csv_path)


# ── helpers ──────────────────────────────────────────────────────────────────


def _merge_segments(
    spectrum_lists: dict,
) -> np.ndarray:
    """Merge multiple scan-type segments into a single averaged spectrum.

    When a RAW file uses segment-scan acquisition, the same nominal m/z
    may appear in multiple segments.  This function aggregates all rows,
    grouping by m/z (with a 1e-5 Da tolerance) and summing intensities.
    """
    if not spectrum_lists:
        return np.array([]).reshape(0, 2)

    all_rows = np.vstack(list(spectrum_lists.values()))
    # columns: [mass, intensity, resolution, baseline, noise, charge] — see pymsfilereader
    masses = all_rows[:, 0]
    intensities = all_rows[:, 1]

    # Round m/z to 5 decimal places to group near-identical values
    rounded = np.round(masses, 5)
    unique, inverse = np.unique(rounded, return_inverse=True)

    summed = np.zeros(len(unique))
    np.add.at(summed, inverse, intensities)

    return np.column_stack([unique, summed])
