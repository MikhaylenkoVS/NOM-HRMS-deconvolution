"""Spectrum operations for HRMS deconvolution of natural organic matter.

This module implements the core mass-spectrometry steps of the pipeline:
loading and denoising spectra, assigning CHON brutto formulas in negative
ion mode ([M-H]-), detecting isotope-labelled derivatization series, and
assembling the final functional-group count table.

Notes
-----
Two derivatization mass increments drive the method:

* ``DELTA_CD3`` = 17.03448 Da per group — deuteromethylation of carboxyl
  groups (-COOH -> -COOCD3), used to count -COOH.
* ``DELTA_CD3CO`` = 45.02939 Da per group — deuteroacylation of hydroxyl
  groups (-OH -> -OCOCD3), used to count -OH.

A homologous "series" is a chain of peaks spaced by an integer multiple of
one of these increments; the length of the series equals the number of
reactive functional groups on the parent molecule.
"""
import pandas as pd
import logging
from nomspectra.spectrum import Spectrum
import warnings
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from dataclasses import dataclass
import itertools
import math
import re
from src.simulations.generate_test_sets import exact_mass_from_formula
from src.configs import CHEM, PIPELINE
from typing import Literal, Sequence

# ---------------------------------------------------------------------------
# Константы (единый источник — src/configs/chemistry.json)
# ---------------------------------------------------------------------------
DELTA_CD3   = CHEM.derivatization_shifts["delta_cd3"]    # Da: сдвиг m/z при замене COOH -> COOCD3
DELTA_CD3CO = CHEM.derivatization_shifts["delta_cd3co"]  # Da: сдвиг m/z при замене OH  -> OCOCD3


# ===========================================================================
# Загрузка спектров
# ===========================================================================

logger = logging.getLogger(__name__)
# Monoisotopic masses of the elements handled by the [M-H]- assignment
# (single source of truth: chemistry.json -> monoisotopic_masses).
ATOMIC_MASS = {
    el: CHEM.monoisotopic_masses[el] for el in CHEM.atomic_mass_elements
}

# Formula-search defaults (single source of truth: pipeline.json -> formula_search).
# JSON stores ranges as [min, max] lists; convert to tuples to preserve the
# exact original types expected downstream.
_FORMULA_SEARCH = PIPELINE.formula_search
_FS_ELEMENTS: tuple[str, ...] = tuple(_FORMULA_SEARCH["elements"])
_FS_RANGES: dict[str, tuple[int, int]] = {
    el: tuple(rng) for el, rng in _FORMULA_SEARCH["ranges"].items()
}


@dataclass
class FormulaSearchConfig:
    """Configuration for brute-force CHON formula generation.

    Defines which elements to enumerate, their per-element count ranges, and
    the chemical plausibility filters used to keep only NOM-like formulas.

    Attributes
    ----------
    elements : tuple of str
        Elements to enumerate, in output order. Default ``("C", "H", "O", "N")``.
    ranges : dict of {str: tuple of (int, int)}, optional
        Inclusive ``(min, max)`` count range per element. If ``None``,
        NOM-oriented defaults are filled in by ``__post_init__``.
    max_hc : float
        Maximum allowed H/C atomic ratio. Default 3.0.
    max_oc : float
        Maximum allowed O/C atomic ratio. Default 1.2.
    max_nc : float
        Maximum allowed N/C atomic ratio. Default 1.0.
    max_dbe : float
        Maximum allowed double-bond equivalent (DBE). Default 30.0.
    min_c : int
        Minimum number of carbon atoms. Default 1.

    Raises
    ------
    ValueError
        If any element in ``elements`` lacks a range in ``ranges``.
    """
    elements: tuple[str, ...] = _FS_ELEMENTS
    ranges: dict[str, tuple[int, int]] | None = None
    # Plausibility filters (defaults from pipeline.json -> formula_search).
    max_hc: float = _FORMULA_SEARCH["max_hc"]      # H/C <= 3
    max_oc: float = _FORMULA_SEARCH["max_oc"]      # O/C <= 1.2
    max_nc: float = _FORMULA_SEARCH["max_nc"]      # N/C <= 1.0
    max_dbe: float = _FORMULA_SEARCH["max_dbe"]    # DBE <= 30
    min_c: int = _FORMULA_SEARCH["min_c"]          # minimum carbons

    def __post_init__(self):
        if self.ranges is None:
            # Default per-element count ranges (see pipeline.json).
            self.ranges = dict(_FS_RANGES)
        for el in self.elements:
            if el not in self.ranges:
                raise ValueError(f"Для элемента {el!r} не задан диапазон в ranges")


def exact_mass_from_counts(counts: dict[str, int]) -> float:
    """Compute the exact (monoisotopic) neutral mass from element counts.

    Parameters
    ----------
    counts : dict of {str: int}
        Element counts, e.g. ``{'C': 7, 'H': 6, 'O': 2}``. Non-positive
        counts are ignored.

    Returns
    -------
    float
        Monoisotopic mass in daltons, summed from ``ATOMIC_MASS``.
    """
    mass = 0.0
    for elem, n in counts.items():
        if n <= 0:
            continue
        mass += ATOMIC_MASS[elem] * n
    return mass


def dbe_from_counts(counts: dict[str, int]) -> float:
    """Compute the double-bond equivalent (DBE) for a CHON formula.

    Parameters
    ----------
    counts : dict of {str: int}
        Element counts; keys ``"C"``, ``"H"``, ``"N"`` are used.

    Returns
    -------
    float
        DBE (rings plus pi-bonds), computed as ``1 + C - H/2 + N/2``.
    """
    c = counts.get("C", 0)
    h = counts.get("H", 0)
    n = counts.get("N", 0)
    return 1 + c - h / 2.0 + n / 2.0

def _row_to_brutto(row, element_order=None):
    """Build a Hill-like brutto formula string from element columns of a row.

    Parameters
    ----------
    row : pandas.Series or mapping
        Row containing per-element integer counts under element-symbol keys.
    element_order : list of str, optional
        Elements to include, in output order. Defaults to
        ``["C", "H", "O", "N", "S", "P"]``.

    Returns
    -------
    str or None
        Concatenated formula (e.g. ``"C7H6O2"``), or ``None`` if no positive
        element counts are present.
    """
    if element_order is None:
        element_order = ["C", "H", "O", "N", "S", "P"]

    parts = []
    for el in element_order:
        if el in row and pd.notna(row[el]):
            val = row[el]
            try:
                val = int(val)
            except Exception:
                continue
            if val > 0:
                parts.append(el if val == 1 else f"{el}{val}")
    return "".join(parts) if parts else None

def load_spectrum(
    path,
    mapper=None,
    sep=",",
    mass_min=PIPELINE.load_spectrum_defaults["mass_min"],
    mass_max=PIPELINE.load_spectrum_defaults["mass_max"],
    metadata=None,
):
    """Load a mass spectrum from a CSV file into a Spectrum object.

    Parameters
    ----------
    path : str or path-like
        Path to the CSV file with mass and intensity columns.
    mapper : dict, optional
        Extra column-rename rules merged over the built-in defaults
        (which map ``m/z``, ``mz``, ``I`` etc. to ``mass``/``intensity``).
    sep : str, optional
        Field separator. Empty/``None`` falls back to ``","``. Default ``","``.
    mass_min, mass_max : float, optional
        Inclusive m/z window (Da) to keep. Defaults 200.0 and 700.0.
    metadata : optional
        Metadata forwarded to the ``Spectrum`` constructor.

    Returns
    -------
    nomspectra.spectrum.Spectrum
        Spectrum whose table has ``mass`` and ``intensity`` columns,
        filtered to the requested window.

    Raises
    ------
    ValueError
        If the file cannot be read or no peaks fall within the m/z window.
    KeyError
        If ``mass``/``intensity`` columns are absent after renaming.
    """

    _sep = sep or ","

    try:
        df = pd.read_csv(path, sep=_sep, encoding="utf-8")
    except Exception as e:
        # Логируем на уровне core для разработчика
        logger.exception("Ошибка чтения CSV-файла %r", path)
        # Поднимаем дальше осмысленное исключение
        raise ValueError(f"Не удалось прочитать CSV-файл '{path}': {e}") from e

    df.columns = [c.strip() for c in df.columns]

    _default_mapper = {
        "m/z": "mass",
        "M/Z": "mass",
        "mz": "mass",
        "Intensity": "intensity",
        "I": "intensity",
    }

    final_mapper = _default_mapper.copy()
    if mapper:
        final_mapper.update(mapper)

    df = df.rename(columns=final_mapper)

    logger.debug(
        "Файл %r: колонки после rename: %r",
        path,
        df.columns.tolist(),
    )

    required = ["mass", "intensity"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(
            f"Колонки {missing} не найдены после переименования. "
            f"Доступные: {df.columns.tolist()}"
        )

    df = df[["mass", "intensity"]].copy()

    df = df[(df["mass"] >= mass_min) & (df["mass"] <= mass_max)].reset_index(drop=True)

    if len(df) == 0:
        raise ValueError(
            f"Для файла '{path}' не найдено ни одного пика "
            f"в диапазоне {mass_min}–{mass_max} Da"
        )

    sp = Spectrum(table=df, metadata=metadata)
    return sp

# ===========================================================================
# Шумоподавление
# ===========================================================================

def denoise(
    spec,
    *,
    force=1.5,
    intensity=None,
    quantile=None,
):
    """Remove noise peaks from a spectrum.

    Thin wrapper around ``Spectrum.noise_filter()``.

    Parameters
    ----------
    spec : nomspectra.spectrum.Spectrum
        Input spectrum.
    force : float, keyword-only, optional
        Multiplier applied to the auto-detected noise level. Default 1.5.
    intensity : float, keyword-only, optional
        Hard absolute intensity threshold. Takes priority when given.
    quantile : float, keyword-only, optional
        Lower intensity quantile in [0, 1]. Used if ``intensity`` is None.

    Returns
    -------
    nomspectra.spectrum.Spectrum
        Denoised spectrum.

    Notes
    -----
    Parameter priority is ``intensity`` > ``quantile`` > ``force``.
    """
    return spec.noise_filter(force=force, intensity=intensity, quantile=quantile)


# ===========================================================================
# ЭТАП 2b: Назначение брутто-формул
# ===========================================================================

# Default per-element count ranges for brutto assignment.
# Source: pipeline.json -> default_brutto_dict. JSON stores [min, max] lists;
# convert to tuples to preserve the exact original types.
DEFAULT_BRUTTO_DICT = {
    el: tuple(rng) for el, rng in PIPELINE.default_brutto_dict.items()
}

def _generate_candidate_formulas(
    mass_min: float,
    mass_max: float,
    cfg: FormulaSearchConfig,
    mode: str = "nom_like",
) -> list[tuple[str, float]]:
    """Enumerate candidate CHON formulas within a neutral-mass window.

    Parameters
    ----------
    mass_min, mass_max : float
        Neutral-mass window (Da). A small margin (+/-1 %) is added at the
        edges to tolerate rounding.
    cfg : FormulaSearchConfig
        Element ranges and chemical filters.
    mode : {"nom_like", "soft"}, optional
        ``"nom_like"`` applies all chemical filters (H/C, O/C, N/C, DBE,
        minimum carbon), close to NOMspectra behaviour. ``"soft"`` applies
        only element ranges and the mass window. Default ``"nom_like"``.

    Returns
    -------
    list of tuple of (str, float)
        Pairs of ``(formula_string, exact_neutral_mass)``.

    Raises
    ------
    ValueError
        If ``mode`` is not one of the supported values.
    """
    mode = mode.lower()
    if mode not in ("nom_like", "soft"):
        raise ValueError(f"Unknown generate mode: {mode}")

    mass_min_abs = mass_min * 0.99
    mass_max_abs = mass_max * 1.01

    c_min, c_max = cfg.ranges["C"]
    h_min, h_max = cfg.ranges["H"]
    o_min, o_max = cfg.ranges.get("O", (0, 0))
    n_min, n_max = cfg.ranges.get("N", (0, 0))

    result: list[tuple[str, float]] = []

    # в "soft" режиме можно не заставлять C >= cfg.min_c, если хочешь максимально мягко
    c_start = max(c_min, cfg.min_c) if mode == "nom_like" else c_min

    for c in range(c_start, c_max + 1):
        counts = {"C": c}
        base_mass_c = c * ATOMIC_MASS["C"]
        if base_mass_c > mass_max_abs:
            break

        for h in range(h_min, h_max + 1):
            counts["H"] = h
            mass_ch = exact_mass_from_counts(counts)
            if mass_ch > mass_max_abs:
                break
            if mass_ch < mass_min_abs:
                continue

            for o in range(o_min, o_max + 1):
                counts["O"] = o
                mass_cho = exact_mass_from_counts(counts)
                if mass_cho > mass_max_abs:
                    break

                for n in range(n_min, n_max + 1):
                    counts["N"] = n
                    mass = exact_mass_from_counts(counts)
                    if mass < mass_min_abs:
                        continue
                    if mass > mass_max_abs:
                        break

                    if mode == "nom_like":
                        c_val = c
                        if c_val <= 0:
                            continue

                        hc = h / c_val
                        oc = o / c_val if c_val > 0 else 0.0
                        nc = n / c_val if c_val > 0 else 0.0

                        if hc > cfg.max_hc:
                            continue
                        if oc > cfg.max_oc:
                            continue
                        if nc > cfg.max_nc:
                            continue

                        dbe = dbe_from_counts(counts)
                        if dbe < 0 or dbe > cfg.max_dbe:
                            continue
                    # в режиме "soft" никакие хим. фильтры не применяем

                    # Строим строку формулы
                    parts: list[str] = []
                    for el in cfg.elements:
                        val = counts.get(el, 0)
                        if val <= 0:
                            continue
                        if val == 1:
                            parts.append(el)
                        else:
                            parts.append(f"{el}{val}")
                    formula_str = "".join(parts)

                    result.append((formula_str, mass))

    return result

def _neutral_to_ion_mass(neutral_mass: float, ion_mode: str) -> float:
    """Convert a neutral mass to observed m/z for a given ion type.

    Parameters
    ----------
    neutral_mass : float
        Neutral monoisotopic mass (Da).
    ion_mode : str
        Ionization mode. Recognised (case-insensitive): ``"neutral"``/empty
        (no shift), ``"[M-H]-"`` (subtract one proton mass), ``"[M+H]+"``
        (add one proton mass).

    Returns
    -------
    float
        The corresponding m/z value.

    Raises
    ------
    ValueError
        If ``ion_mode`` is not recognised.
    """
    ion_mode = ion_mode.lower()

    if ion_mode in ("neutral", None, ""):
        return neutral_mass

    # отрицательный режим [M-H]-
    if ion_mode in ("[m-h]-", "m-h", "mh-"):
        return neutral_mass - ATOMIC_MASS["H"]

    # положительный режим [M+H]+ — на будущее
    if ion_mode in ("[m+h]+", "m+h", "mh+"):
        return neutral_mass + ATOMIC_MASS["H"]

    # можно добавить другие аддукты позже
    raise ValueError(f"Unknown ion_mode: {ion_mode}")

def assign_formulas_simple(
    src,
    rel_error_ppm: float = 1.0,
    mass_min: float | None = None,
    mass_max: float | None = None,
    search_config: FormulaSearchConfig | None = None,
    brutto_generation_mode: str = "nom_like",  # "nom_like" или "soft"
    ion_mode: str = "[M-H]-",                 # тип иона; по умолчанию [M-H]-
):
    """Assign brutto formulas by brute-force CHON enumeration.

    Generates candidate CHON formulas over the mass window, converts them to
    m/z according to ``ion_mode``, and picks for each peak the formula with
    the smallest ppm deviation within ``rel_error_ppm``.

    Parameters
    ----------
    src : nomspectra.spectrum.Spectrum
        Spectrum whose ``table`` (with a ``mass`` column) is annotated.
    rel_error_ppm : float, optional
        Maximum allowed mass error (ppm) for a match. Default 1.0.
    mass_min, mass_max : float or None, optional
        Neutral-mass window bounds; if None, taken from the observed masses.
    search_config : FormulaSearchConfig or None, optional
        Formula-generation configuration. A default config is used if None.
    brutto_generation_mode : {"nom_like", "soft"}, optional
        Passed to the candidate generator. Default ``"nom_like"``.
    ion_mode : str, optional
        Ionization mode for the neutral-to-ion conversion. Default ``"[M-H]-"``.

    Returns
    -------
    nomspectra.spectrum.Spectrum
        The same spectrum with ``table["brutto"]`` (formula str or None) and
        ``table["assign"]`` (bool) columns filled in.

    Notes
    -----
    When several candidates tie on ppm, the first generated formula wins;
    no NOM-space tie-break is currently applied.
    """
    if search_config is None:
        search_config = FormulaSearchConfig()

    table = src.table.copy()
    mass_series = table["mass"]

    if mass_min is None:
        mass_min_local = float(mass_series.min())
    else:
        mass_min_local = float(mass_min)

    if mass_max is None:
        mass_max_local = float(mass_series.max())
    else:
        mass_max_local = float(mass_max)

    # Генерируем кандидатов (нейтральные массы)
    candidates = _generate_candidate_formulas(
        mass_min=mass_min_local,
        mass_max=mass_max_local,
        cfg=search_config,
        mode=brutto_generation_mode,  # "soft" / "nom_like"
    )

    if not candidates:
        table["brutto"] = None
        table["assign"] = False
        src.table = table
        return src

    # Разделим формулы и НЕЙТРАЛЬНЫЕ массы
    cand_formulas = np.array([f for f, m in candidates], dtype=object)
    cand_masses_neutral = np.array([m for f, m in candidates], dtype=float)

    # Переводим нейтральные массы в m/z с учётом режима ионизации
    cand_masses_ion = np.array(
        [_neutral_to_ion_mass(m, ion_mode) for m in cand_masses_neutral],
        dtype=float,
    )

    table["brutto"] = None
    table["assign"] = False

    for idx, row in table.iterrows():
        mass_obs = float(row["mass"])

        # считаем ppm-разницу по ИОННЫМ массам
        ppm = (cand_masses_ion - mass_obs) / mass_obs * 1e6
        abs_ppm = np.abs(ppm)

        mask = abs_ppm <= rel_error_ppm
        if not mask.any():
            continue

        # TODO [NOM-soft]: добавить мягкий NOM-приоритет при выборе формулы.
        # Сейчас, если в окне ±rel_error_ppm несколько формул с одинаковым ppm,
        # выбирается первая по порядку генерации cand_formulas.
        # К релизу:
        # - считать для кандидатов простые NOM-показатели (H/C, O/C, N/C, DBE),
        # - среди формул с близким ppm выбирать ту, что ближе к типичному NOM-полю
        #   (например, через штраф в виде w_ppm * |ppm| + w_nom * distance_to_nom).

        best_local = np.argmin(abs_ppm[mask])
        global_indices = np.where(mask)[0]
        chosen_global = global_indices[best_local]

        best_formula = cand_formulas[chosen_global]
        table.at[idx, "brutto"] = best_formula
        table.at[idx, "assign"] = True

    src.table = table
    return src

AssignMode = Literal["simple", "nomspectra"]


def _row_to_brutto_from_elements(row, element_order=None):
    """Build a brutto formula string from per-element columns of a row.

    Parameters
    ----------
    row : pandas.Series or mapping
        Row with integer element counts under element-symbol keys.
    element_order : list of str, optional
        Elements to include, in output order. Defaults to
        ``["C", "H", "O", "N", "S", "P"]``.

    Returns
    -------
    str or None
        Concatenated formula, or ``None`` if no positive counts are present.
    """
    if element_order is None:
        element_order = ["C", "H", "O", "N", "S", "P"]

    parts = []
    for el in element_order:
        if el not in row:
            continue
        val = row[el]
        if pd.isna(val):
            continue
        try:
            n = int(val)
        except Exception:
            continue
        if n <= 0:
            continue
        parts.append(el if n == 1 else f"{el}{n}")

    return "".join(parts) if parts else None


def _ensure_brutto_from_element_columns(src):
    """Guarantee ``assign`` and ``brutto`` columns after formula assignment.

    If ``brutto`` is missing but per-element columns (C, H, O, N, ...) are
    present, the formula string is reconstructed from them for assigned rows.
    All other columns are preserved.

    Parameters
    ----------
    src : nomspectra.spectrum.Spectrum
        Spectrum whose ``table`` is checked and, if needed, augmented.

    Returns
    -------
    nomspectra.spectrum.Spectrum
        The same spectrum with a guaranteed ``brutto`` column.

    Raises
    ------
    TypeError
        If ``src`` has no ``table`` attribute.
    RuntimeError
        If the ``assign`` column is missing from the table.
    """
    if not hasattr(src, "table"):
        raise TypeError("Ожидается объект Spectrum с атрибутом .table")

    df = src.table.copy()

    if "assign" not in df.columns:
        raise RuntimeError(
            "После назначения формул в src.table отсутствует колонка 'assign'"
        )

    if "brutto" not in df.columns:
        df["brutto"] = None

    element_order = ["C", "H", "O", "N", "S", "P"]
    element_cols = [c for c in element_order if c in df.columns]

    if element_cols:
        assigned_mask = df["assign"] == True
        df.loc[assigned_mask, "brutto"] = df.loc[assigned_mask].apply(
            lambda row: _row_to_brutto_from_elements(row, element_order=element_order),
            axis=1,
        )

    src.table = df
    return src

def assign_formulas_nomspectra(
    src,
    *,
    brutto_dict=None,
    rel_error=0.5,
    sign='-',
    mass_min=None,
    mass_max=None,
):
    """Assign brutto formulas to the source spectrum via NOMspectra.

    Parameters
    ----------
    src : nomspectra.spectrum.Spectrum
        Source spectrum to annotate.
    brutto_dict : dict of {str: tuple of (int, int)}, optional
        Per-element count ranges. Defaults to ``DEFAULT_BRUTTO_DICT``.
    rel_error : float, keyword-only, optional
        Mass tolerance (ppm) for assignment. Negative values are made
        positive with a warning. Default 0.5.
    sign : {'-', '+'}, keyword-only, optional
        Ionization sign; ``'-'`` corresponds to [M-H]-. Default ``'-'``.
    mass_min, mass_max : float or None, keyword-only, optional
        Optional m/z window; swapped with a warning if given in wrong order.

    Returns
    -------
    nomspectra.spectrum.Spectrum
        Spectrum with guaranteed boolean ``assign`` and string ``brutto``
        columns.

    Raises
    ------
    TypeError
        If ``src`` is not a ``Spectrum`` or ``brutto_dict`` is not a dict.
    ValueError
        If any element range is not a ``(min, max)`` pair.

    Warns
    -----
    UserWarning
        If no formula could be assigned to any peak.
    """
    if rel_error < 0:
        rel_error = abs(rel_error)
        warnings.warn("Relative error is negative")

    if mass_min is not None and mass_max is not None and mass_min > mass_max:
        mass_min, mass_max = mass_max, mass_min
        warnings.warn("Mass_max is less than mass_min")

    if not isinstance(src, Spectrum):
        raise TypeError(f'Некорректный формат файла {src}')

    if brutto_dict is None:
        brutto_dict = DEFAULT_BRUTTO_DICT
    elif not isinstance(brutto_dict, dict):
        raise TypeError("brutto_dict должен быть dict с диапазонами по элементам")

    for el, bounds in brutto_dict.items():
        if not (isinstance(bounds, (tuple, list)) and len(bounds) == 2):
            raise ValueError(f"Для элемента {el!r} ожидается (min, max), получено {bounds!r}")

    src = src.assign(
        brutto_dict=brutto_dict,
        rel_error=rel_error,
        sign=sign,
        mass_min=mass_min,
        mass_max=mass_max,
    )

    src = _ensure_brutto_from_element_columns(src)

    assign_col = src.table["assign"]
    if assign_col.dtype != bool:
        try:
            src.table["assign"] = src.table["assign"].astype(bool)
            assign_col = src.table["assign"]
        except Exception as e:
            raise TypeError(
                f"Ожидается булевый столбец 'assign', получен dtype={src.table['assign'].dtype}"
            ) from e

    n_assigned = int(assign_col.sum())
    if n_assigned == 0:
        warnings.warn("Ни одной брутто-формулы не назначено (assign == False для всех пиков)")

    return src


def assign_formulas(
    src,
    mode: str = "nomspectra",
    rel_error_ppm: float = 1.0,
    mass_min: float | None = None,
    mass_max: float | None = None,
    formulas=None,
    brutto_dict=None,
    sign: str = "-",
    search_config: FormulaSearchConfig | None = None,
    brutto_generation_mode: str = "nom_like",
    ion_mode: str = "[M-H]-",
    **kwargs,
):
    """Dispatch brutto-formula assignment to the selected backend.

    Parameters
    ----------
    src : nomspectra.spectrum.Spectrum
        Spectrum to annotate.
    mode : {"nomspectra", "simple", "simple_from_molecules"}, optional
        Assignment backend. Default ``"nomspectra"``.
    rel_error_ppm : float, optional
        Mass tolerance (ppm). Default 1.0.
    mass_min, mass_max : float or None, optional
        Optional mass window.
    formulas : sequence, optional
        Explicit formula list; required for ``"simple_from_molecules"``.
    brutto_dict : dict, optional
        Per-element ranges for the NOMspectra backend.
    sign : {'-', '+'}, optional
        Ionization sign for the NOMspectra backend. Default ``'-'``.
    search_config : FormulaSearchConfig or None, optional
        Configuration for the ``"simple"`` backend.
    brutto_generation_mode : {"nom_like", "soft"}, optional
        Candidate-generation mode for the ``"simple"`` backend.
    ion_mode : str, optional
        Ionization mode for the ``"simple"`` backend. Default ``"[M-H]-"``.
    **kwargs
        Extra arguments forwarded to the NOMspectra backend.

    Returns
    -------
    nomspectra.spectrum.Spectrum
        Annotated spectrum.

    Raises
    ------
    ValueError
        If ``mode`` is unknown, or ``formulas`` is missing for
        ``"simple_from_molecules"``.
    NotImplementedError
        For ``mode="simple_from_molecules"`` (not yet implemented).
    """
    kwargs.pop("rel_error", None)
    kwargs.pop("sign", None)
    kwargs.pop("mass_min", None)
    kwargs.pop("mass_max", None)
    kwargs.pop("brutto_dict", None)

    if mode == "simple":
        return assign_formulas_simple(
            src,
            rel_error_ppm=rel_error_ppm,
            mass_min=mass_min,
            mass_max=mass_max,
            search_config=search_config,
            brutto_generation_mode=brutto_generation_mode,
            ion_mode=ion_mode,
        )

    if mode == "simple_from_molecules":
        if formulas is None:
            raise ValueError("Для mode='simple_from_molecules' нужно передать список formulas")
        raise NotImplementedError("mode='simple_from_molecules' пока не реализован")

    if mode == "nomspectra":
        src = src.assign(
            brutto_dict=brutto_dict,
            rel_error=rel_error_ppm,
            sign=sign,
            mass_min=mass_min,
            mass_max=mass_max,
            **kwargs,
        )
        src = _ensure_brutto_from_element_columns(src)
        return src

    raise ValueError(f"Неизвестный режим assign: {mode}")

# ===========================================================================
# Поиск серий
# ===========================================================================

def _find_peak(mz_array, target_mz, ppm_tol):
    """Find the peak in ``mz_array`` closest to ``target_mz`` within tolerance.

    Parameters
    ----------
    mz_array : array-like of float
        Candidate m/z values to search.
    target_mz : float
        Target m/z to match.
    ppm_tol : float
        Maximum allowed deviation (ppm).

    Returns
    -------
    int or None
        Index of the closest peak within ``ppm_tol``, or ``None`` if none
        falls within tolerance.
    """
    mz = pd.Series(mz_array)
    diffs_ppm = (mz - target_mz).abs() / target_mz * 1e6
    matched = diffs_ppm[diffs_ppm <= ppm_tol]
    if matched.empty:
        return None
    return int(matched.idxmin())


def find_series(
    src,
    deriv,
    delta,
    ppm_tol=5.0,
    max_groups=20,
    allow_gaps=True,
    min_series_length=1,
):
    """Detect homologous derivatization series in a labelled spectrum.

    For each assigned source peak ``m_0``, searches the derivatized spectrum
    for the chain ``m_0 + 1*delta, m_0 + 2*delta, ..., m_0 + n*delta``. The
    number of steps found equals the number of reactive functional groups
    (-COOH for ``DELTA_CD3``, -OH for ``DELTA_CD3CO``).

    Parameters
    ----------
    src : nomspectra.spectrum.Spectrum
        Source spectrum with assigned formulas (needs ``brutto``, ``mass``,
        ``assign`` columns).
    deriv : nomspectra.spectrum.Spectrum
        Derivatized-sample spectrum (needs ``mass``, ``intensity`` columns).
    delta : float
        Expected m/z shift per functional group (Da), e.g. ``DELTA_CD3``.
    ppm_tol : float, optional
        Mass-match tolerance (ppm). Must be > 0. Default 5.0.
    max_groups : int, optional
        Maximum number of functional groups (series steps) to probe.
        Default 20.
    allow_gaps : bool, optional
        If ``True`` (recommended), keep searching past a missing step;
        if ``False``, stop the series at the first gap. Default ``True``.
    min_series_length : int, optional
        Minimum series length required to emit a record. Default 1.

    Returns
    -------
    pandas.DataFrame
        One row per detected series with columns: ``mass_src``, ``brutto``,
        ``n_groups`` (length by last found step), ``steps_found`` (1-based
        list), ``missing`` (skipped steps inside the series), ``series_mz``
        (m/z per step 1..n_groups, ``None`` for a gap).

    Raises
    ------
    ValueError
        If ``ppm_tol <= 0``, if ``max_groups``/``min_series_length`` are
        below 1, or if required columns are missing from ``src``/``deriv``.

    Notes
    -----
    The series length is the last *found* step (1-based): observing steps
    1, 2, 3, 5 yields ``n_groups = 4`` recorded as length 5 with step 4
    listed under ``missing``.
    """

    if ppm_tol <= 0:
        raise ValueError(f"ppm_tol должно быть > 0, получено {ppm_tol}")
    if max_groups < 1 or min_series_length < 1:
        raise ValueError(
            f"max_groups ({max_groups}) и min_series_length ({min_series_length}) "
            "должны быть >= 1"
        )
    required_src = ['brutto', 'mass', 'assign']
    missing_src = [c for c in required_src if c not in src.table.columns]
    if missing_src:
        raise ValueError(f"В src не хватает столбца {missing_src}")
    required_deriv = ['mass', 'intensity']
    missing_deriv = [c for c in required_deriv if c not in deriv.table.columns]
    if missing_deriv:
        raise ValueError(
            f"В deriv.table отсутствуют колонки {missing_deriv}. "
            "Файл дериватизированного спектра некорректен."
        )

    mz_deriv = deriv.table['mass'].values
    records  = []

    for _, row in src.table.iterrows():
        if not row.get('assign', False):
            continue

        m0          = row['mass']
        found_steps = []
        series_mz   = []

        for step in range(1, max_groups + 1):
            target = m0 + step * delta
            idx = _find_peak(mz_deriv, target, ppm_tol)

            if idx is not None:
                found_steps.append(step)
                series_mz.append(float(mz_deriv[idx]))
            else:
                series_mz.append(None)
                if not allow_gaps and found_steps:
                    series_mz = series_mz[:step]
                    break

        if not found_steps:
            n_groups      = 0
            missing_steps = []
            trimmed       = []
        else:
            n_groups      = max(found_steps)
            all_steps     = set(range(1, n_groups + 1))
            missing_steps = sorted(all_steps - set(found_steps))
            trimmed       = series_mz[:n_groups]

        if n_groups >= min_series_length:
            records.append({
                'mass_src':    m0,
                'brutto':      row.get('brutto', ''),
                'n_groups':    n_groups,
                'steps_found': found_steps,
                'missing':     missing_steps,
                'series_mz':   trimmed,
            })

        if not found_steps:
            continue

    return pd.DataFrame(
        records,
        columns=['mass_src', 'brutto', 'n_groups', 'steps_found', 'missing', 'series_mz'],
    )


# ===========================================================================
# Сборка итоговой таблицы
# ===========================================================================

def build_result_table(src, df_dmet, df_dacet):
    """Assemble the final -COOH / -OH count table per brutto formula.

    Parameters
    ----------
    src : nomspectra.spectrum.Spectrum
        Source spectrum with assigned formulas.
    df_dmet : pandas.DataFrame
        ``find_series`` output for deuteromethylation (CD3 series,
        ``delta = DELTA_CD3``); its ``n_groups`` becomes ``N_COOH``.
    df_dacet : pandas.DataFrame
        ``find_series`` output for deuteroacylation (CD3CO series,
        ``delta = DELTA_CD3CO``); its ``n_groups`` becomes ``N_OH``.

    Returns
    -------
    pandas.DataFrame
        Columns ``mass``, ``intensity``, ``brutto``, ``N_COOH``,
        ``N_OH_total``, ``N_OH``, ``missing_dmet``, ``missing_dacet``,
        sorted by mass. Peaks without a series get a count of 0.

    Notes
    -----
    Source and series peaks are joined on m/z rounded to 4 decimals.
    """
    base = (
        src.table
        .loc[src.table.get('assign', pd.Series(False, index=src.table.index)) == True]
        [['mass', 'intensity', 'brutto']]
        .copy()
        .reset_index(drop=True)
    )
    base['mass_key'] = base['mass'].round(4)

    def _enrich(df, prefix):
        if df.empty:
            return pd.DataFrame(columns=['mass_key', f'n_{prefix}', f'missing_{prefix}'])
        tmp = df[['mass_src', 'n_groups', 'missing']].copy()
        tmp['mass_key'] = tmp['mass_src'].round(4)
        return tmp.rename(columns={
            'n_groups': f'n_{prefix}',
            'missing':  f'missing_{prefix}',
        })[['mass_key', f'n_{prefix}', f'missing_{prefix}']]

    result = (
        base
        .merge(_enrich(df_dmet,  'dmet'),  on='mass_key', how='left')
        .merge(_enrich(df_dacet, 'dacet'), on='mass_key', how='left')
    )

    result['n_dmet']  = result['n_dmet'].fillna(0).astype(int)
    result['n_dacet'] = result['n_dacet'].fillna(0).astype(int)
    result['N_COOH']     = result['n_dmet']
    result['N_OH_total'] = result['n_dacet']
    result['N_OH'] = result['n_dacet']

    return result[[
        'mass', 'intensity', 'brutto',
        'N_COOH', 'N_OH_total', 'N_OH',
        'missing_dmet', 'missing_dacet',
    ]].sort_values('mass').reset_index(drop=True)


# ===========================================================================
# Визуализация серий с пропущенными пиками
# ===========================================================================

def visualize_series(
    src,
    deriv,
    df_series,
    delta,
    label="series",
    max_rows=15,
    figsize_per_row=(12, 1.4),
    ppm_tol=5.0,
    save_path=None,
):
    """Plot detected series, highlighting missing (gap) peaks.

    For each compound a ladder of expected peaks is drawn: blue = source
    peak ``m_0``, green = found series peak, dashed red = missing expected
    peak.

    Parameters
    ----------
    src : nomspectra.spectrum.Spectrum
        Source spectrum.
    deriv : nomspectra.spectrum.Spectrum
        Derivatized-sample spectrum.
    df_series : pandas.DataFrame
        ``find_series`` output to visualize.
    delta : float
        Series step (Da).
    label : str, optional
        Title label. Default ``"series"``.
    max_rows : int, optional
        Maximum number of compounds to display. Default 15.
    figsize_per_row : tuple of (float, float), optional
        Per-row ``(width, height)`` in inches. Default ``(12, 1.4)``.
    ppm_tol : float, optional
        Search tolerance (ppm). Default 5.0.
    save_path : str or None, optional
        If given, the figure is saved to this path; otherwise it is shown.

    Returns
    -------
    None
        Only rows containing gaps are plotted; if none exist the function
        returns after printing a message.
    """
    if df_series.empty:
        print(f"[{label}] Серии не найдены.")
        return

    has_missing = df_series[df_series['missing'].apply(len) > 0]
    display_df  = has_missing.head(max_rows)

    if display_df.empty:
        print(f"[{label}] Пропущенных пиков в сериях нет.")
        return

    n_rows = len(display_df)
    fig, axes = plt.subplots(
        n_rows, 1,
        figsize=(figsize_per_row[0], figsize_per_row[1] * n_rows + 1.5),
        squeeze=False,
    )
    fig.suptitle(
        f"Серии {label} с пропущенными пиками "
        f"(delta_m = {delta:.5f} Da, допуск {ppm_tol} ppm)",
        fontsize=11, fontweight="bold",
    )

    mz_src    = src.table['mass'].values
    int_src   = src.table['intensity'].values
    mz_deriv  = deriv.table['mass'].values
    int_deriv = deriv.table['intensity'].values

    for ax_idx, (_, row) in enumerate(display_df.iterrows()):
        ax       = axes[ax_idx][0]
        m0       = row['mass_src']
        n_groups = row['n_groups']
        missing  = set(row['missing'])
        series   = row['series_mz']

        idx_s = _find_peak(mz_src, m0, ppm_tol * 10)
        i0    = float(int_src[idx_s]) if idx_s is not None else 1.0

        max_i = i0
        for mz_step in series:
            if mz_step is not None:
                idx_d = _find_peak(mz_deriv, mz_step, ppm_tol * 2)
                if idx_d is not None:
                    max_i = max(max_i, float(int_deriv[idx_d]))

        bar_w = delta * 0.08
        ax.bar(m0, i0, width=bar_w, color='steelblue', alpha=0.85)

        for step, mz_step in enumerate(series, start=1):
            expected = m0 + step * delta
            if step in missing or mz_step is None:
                ax.axvline(x=expected, color='crimson',
                           linestyle='--', linewidth=1.0, alpha=0.75)
                ax.text(expected, max_i * 0.55, f"n={step}",
                        color='crimson', fontsize=7, ha='center', va='bottom')
            else:
                idx_d = _find_peak(mz_deriv, float(mz_step), ppm_tol * 2)
                i_step = float(int_deriv[idx_d]) if idx_d is not None else max_i * 0.1
                ax.bar(mz_step, i_step, width=bar_w, color='forestgreen', alpha=0.8)
                ax.text(mz_step, i_step + max_i * 0.02, f"n={step}",
                        color='darkgreen', fontsize=7, ha='center', va='bottom')

        ax.set_xlim(m0 - delta * 0.5, m0 + (n_groups + 1) * delta)
        ax.set_ylim(0, max_i * 1.25)
        ax.set_ylabel('I', fontsize=8)
        ax.set_title(
            f"{row['brutto']}   m/z={m0:.4f}   "
            f"серия 1..{n_groups}   пропущено: {sorted(missing)}",
            fontsize=9,
        )
        ax.tick_params(labelsize=7)

    fig.legend(
        handles=[
            mpatches.Patch(color='steelblue',   label='Исходный пик'),
            mpatches.Patch(color='forestgreen', label='Найденный пик серии'),
            mpatches.Patch(color='crimson',     label='Пропущенный пик (ожидаемая позиция)'),
        ],
        loc='lower center', ncol=3, fontsize=9, frameon=True,
        bbox_to_anchor=(0.5, 0),
    )
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[{label}] График сохранён: {save_path}")
    else:
        plt.show()
    plt.close(fig)
