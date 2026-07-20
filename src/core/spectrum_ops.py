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
from src.core.van_krevelen import NOM_REGIONS
from src.core.molecule import parse_formula
from src.configs import CHEM, PIPELINE
from typing import Literal, Sequence

# ---------------------------------------------------------------------------
# Константы (единый источник — src/configs/chemistry.json)
# ---------------------------------------------------------------------------
DELTA_CD3 = CHEM.derivatization_shifts[
    "delta_cd3"
]  # Da: сдвиг m/z при замене COOH -> COOCD3
DELTA_CD3CO = CHEM.derivatization_shifts[
    "delta_cd3co"
]  # Da: сдвиг m/z при замене OH  -> OCOCD3

# ===========================================================================
# Загрузка спектров
# ===========================================================================

logger = logging.getLogger(__name__)
# Monoisotopic masses of the elements handled by the [M-H]- assignment
# (single source of truth: chemistry.json -> monoisotopic_masses).
ATOMIC_MASS = {el: CHEM.monoisotopic_masses[el] for el in CHEM.atomic_mass_elements}

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
    max_hc: float = _FORMULA_SEARCH["max_hc"]  # H/C <= 3
    max_oc: float = _FORMULA_SEARCH["max_oc"]  # O/C <= 1.2
    max_nc: float = _FORMULA_SEARCH["max_nc"]  # N/C <= 1.0
    max_dbe: float = _FORMULA_SEARCH["max_dbe"]  # DBE <= 30
    min_c: int = _FORMULA_SEARCH["min_c"]  # minimum carbons

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
        DBE (rings plus pi-bonds), computed as ``max(0, 1 + C - H/2 + N/2)``.
        Negative values are clamped to zero (DBE < 0 is chemically meaningless).
    """
    c = counts.get("C", 0)
    h = counts.get("H", 0)
    n = counts.get("N", 0)
    return max(0.0, 1 + c - h / 2.0 + n / 2.0)

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

# -- CSV column name mapper (IMP-11) -------------------------------------------
# Единый маппинг имён колонок CSV → mass / intensity, используется
# load_spectrum() и app.py
CSV_COLUMN_MAPPER = {
    "m/z": "mass",
    "M/Z": "mass",
    "mz": "mass",
    "mass": "mass",
    "Intensity": "intensity",
    "I": "intensity",
    "int": "intensity",
    "Int": "intensity",
}

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

    _default_mapper = CSV_COLUMN_MAPPER.copy()
    if mapper:
        _default_mapper.update(mapper)

    df = df.rename(columns=_default_mapper)

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
    mode: str = "soft",
) -> list[tuple[str, float]]:
    """Enumerate candidate CHON formulas within a neutral-mass window.

    Uses precomputed mass ranges per element (C→H→O→N) to avoid redundant
    iterations and guarantee that every feasible formula is generated
    regardless of loop order.

    Parameters
    ----------
    mass_min, mass_max : float
        Neutral-mass window (Da). A small margin (+/-1 %) is added at the
        edges to tolerate rounding.
    cfg : FormulaSearchConfig
        Element ranges.
    mode : {"soft", "nom_like"}, optional
        Ignored — kept for backward compatibility. Default ``"soft"``.

    Returns
    -------
    list of tuple of (str, float)
        Pairs of ``(formula_string, exact_neutral_mass)``.
    """
    eps = 1e-9  # защита от округления в ceil/floor
    mass_min_abs = mass_min * 0.99
    mass_max_abs = mass_max * 1.01

    c_min, c_max = cfg.ranges["C"]
    h_min, h_max = cfg.ranges["H"]
    o_min, o_max = cfg.ranges.get("O", (0, 0))
    n_min, n_max = cfg.ranges.get("N", (0, 0))

    M_C = ATOMIC_MASS["C"]
    M_H = ATOMIC_MASS.get("H", 0.0)
    M_O = ATOMIC_MASS.get("O", 0.0)
    M_N = ATOMIC_MASS.get("N", 0.0)
    M_O_max = o_max * M_O
    M_N_max = n_max * M_N
    M_extra = M_O_max + M_N_max  # макс. добавка гетероатомов

    result: list[tuple[str, float]] = []

    for c in range(c_min, c_max + 1):
        base_C = c * M_C
        if base_C > mass_max_abs:
            break

        # ---------- допустимый диапазон H для этого C ----------
        # Даже с макс. O+N масса должна достичь mass_min_abs
        h_lo = max(
            h_min,
            math.ceil((mass_min_abs - base_C - M_extra) / M_H - eps),
        )
        # Без O+N масса не должна превысить mass_max_abs
        h_hi = min(
            h_max,
            math.floor((mass_max_abs - base_C) / M_H + eps),
        )
        if h_lo > h_hi:
            continue

        for h in range(h_lo, h_hi + 1):
            base_CH = base_C + h * M_H

            # ---------- допустимый диапазон O ----------
            o_lo = max(
                o_min,
                (
                    math.ceil((mass_min_abs - base_CH - M_N_max) / M_O - eps)
                    if M_O > 0
                    else 0
                ),
            )
            o_hi = min(
                o_max,
                math.floor((mass_max_abs - base_CH) / M_O + eps) if M_O > 0 else 0,
            )
            if o_lo > o_hi:
                continue

            for o in range(o_lo, o_hi + 1):
                base_CHO = base_CH + o * M_O

                # ---------- допустимый диапазон N ----------
                n_lo = max(
                    n_min,
                    math.ceil((mass_min_abs - base_CHO) / M_N - eps) if M_N > 0 else 0,
                )
                n_hi = min(
                    n_max,
                    math.floor((mass_max_abs - base_CHO) / M_N + eps) if M_N > 0 else 0,
                )
                for n in range(n_lo, n_hi + 1):
                    mass = base_CHO + n * M_N

                    # ── Классические химические правила (жёсткие) ──────
                    # LEWIS: сумма валентностей должна быть чётной
                    #   4C + H + 3N + 2O ≡ H + N (mod 2)
                    if (h + n) % 2 != 0:
                        continue
                    # SENIOR: необходимое условие существования связного графа
                    #   H ≤ 2C + N + 2  (эквивалентно DBE ≥ 0)
                    if h > 2 * c + n + 2:
                        continue

                    # строим строку формулы
                    parts: list[str] = []
                    counts = {"C": c, "H": h}
                    if o > 0:
                        counts["O"] = o
                    if n > 0:
                        counts["N"] = n
                    for el in cfg.elements:
                        val = counts.get(el, 0)
                        if val <= 0:
                            continue
                        parts.append(el if val == 1 else f"{el}{val}")
                    result.append(("".join(parts), mass))

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

    # отрицательный режим [M-H]- : вычитаем массу протона (не атома H)
    if ion_mode in ("[m-h]-", "m-h", "mh-"):
        return neutral_mass - CHEM.proton_mass

    # положительный режим [M]+ : вычитаем только массу электрона
    if ion_mode in ("[m]+", "m+", "[m+]"):
        return neutral_mass - CHEM.electron_mass

    # положительный режим [M+H]+ : добавляем массу протона
    if ion_mode in ("[m+h]+", "m+h", "mh+"):
        return neutral_mass + CHEM.proton_mass

    # можно добавить другие аддукты позже
    raise ValueError(f"Unknown ion_mode: {ion_mode}")

# ── NOM-приоритизация ────────────────────────────────────────────────────────

# Центры NOM-областей (усреднённые вершины) для расчёта расстояния
_NOM_REGION_CENTERS: list[tuple[float, float]] = [
    (
        sum(v[0] for v in r["vertices"]) / len(r["vertices"]),
        sum(v[1] for v in r["vertices"]) / len(r["vertices"]),
    )
    for r in NOM_REGIONS
]

# ── Изотопный фильтр ¹³C (опциональный, формула Бейнона) ──────────────────

# Относительные распространённости тяжёлых изотопов (в %):
# ¹³C: 1.1%, ²H: 0.015%, ¹⁷O: 0.04%, ¹⁵N: 0.37%
_BEYNON_COEFFS = {"C": 1.1, "H": 0.015, "O": 0.04, "N": 0.37}

# Порог расхождения M+1/M для штрафа: если |реальное − теоретическое| / теоретическое > 20%
_ISOTOPE_TOLERANCE = 0.20

# Штраф к score при несовпадении изотопного паттерна (средний уровень)
_ISOTOPE_PENALTY = 2.0

# Масса ¹³C − ¹²C (Da)
_DELTA_M1 = 1.00335


def _beynon_m1_ratio(counts: dict[str, int]) -> float:
    """Теоретическое соотношение (M+1)/M по формуле Бейнона.

    Parameters
    ----------
    counts : dict of {str: int}
        Атомные количества (C, H, O, N, ...).

    Returns
    -------
    float
        (M+1)/M как доля (не проценты), e.g. 0.078 для C₇H₆O₂.
    """
    total = 0.0
    for el, coeff in _BEYNON_COEFFS.items():
        total += counts.get(el, 0) * coeff
    return total / 100.0


def _measure_m1_ratio(
    mass: float,
    original_spec,
    ppm_tol: float = 5.0,
) -> float | None:
    """Измерить реальное отношение M+1/M в исходном (pre-denoise) спектре.

    Ищет пик на массе mass + 1.00335 Да в пределах ppm_tol.
    Возвращает отношение интенсивностей или None, если пик не найден.

    Parameters
    ----------
    mass : float
        Масса моноизотопного пика (m/z).
    original_spec : Spectrum
        Исходный спектр до шумоподавления.
    ppm_tol : float
        Допуск поиска в ppm. По умолчанию 5.0.

    Returns
    -------
    float or None
        M1_intensity / M_intensity, или None если M+1 не найден.
    """
    mass_m1 = mass + _DELTA_M1
    tol_da = mass_m1 * ppm_tol * 1e-6

    masses = original_spec.table["mass"].values
    intensities = original_spec.table["intensity"].values

    diffs = np.abs(masses - mass_m1)
    mask = diffs <= tol_da
    if not mask.any():
        return None

    # Найти исходный пик — ближайший по массе
    diffs_orig = np.abs(masses - mass)
    mask_orig = diffs_orig <= (mass * ppm_tol * 1e-6)
    if not mask_orig.any():
        return None

    idx_orig = np.argmin(diffs_orig)
    idx_m1 = np.argmin(diffs[mask])
    m1_indices = np.where(mask)[0]
    idx_m1 = m1_indices[idx_m1]

    intensity_orig = float(intensities[idx_orig])
    intensity_m1 = float(intensities[idx_m1])

    if intensity_orig <= 0:
        return None

    return intensity_m1 / intensity_orig


def _nom_distance(hc: float, oc: float) -> float:
    """Минимальное евклидово расстояние от (O/C, H/C) до центра NOM-области."""
    if hc <= 0:
        return 10.0  # заведомо большой штраф для не-NOM
    best = min(math.hypot(oc - cx, hc - cy) for cx, cy in _NOM_REGION_CENTERS)
    return best

def assign_formulas(
    src,
    rel_error_ppm: float = 1.0,
    mass_min: float | None = None,
    mass_max: float | None = None,
    search_config: FormulaSearchConfig | None = None,
    brutto_generation_mode: str = "nom_like",
    ion_mode: str = CHEM.default_ion_mode,
    nom_weight: float = 1.0,
    isotope_filter: bool = False,
    original=None,
    **kwargs,
):
    # Игнорируем устаревшие параметры для обратной совместимости
    kwargs.pop("mode", None)
    kwargs.pop("nom_prioritize", None)
    kwargs.pop("brutto_dict", None)
    kwargs.pop("sign", None)
    kwargs.pop("rel_error", None)
    kwargs.pop("formulas", None)

    """Assign brutto formulas by brute-force CHON enumeration.

    Generates candidate CHON formulas over the mass window, converts them to
    m/z according to ``ion_mode``, and picks the most NOM-plausible formula
    among all candidates within ``rel_error_ppm``.

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
    nom_weight : float, optional
        Weight for the NOM-distance term in the composite score:
        ``score = nom_weight * nom_distance + penalties``. Default 1.0.

    Returns
    -------
    nomspectra.spectrum.Spectrum
        The same spectrum with ``table["brutto"]`` (formula str or None),
        ``table["assign"]`` (bool), and ``table["all_candidates"]`` (list of str).

    Notes
    -----
    Ppm deviation is used ONLY to define the candidate set (admission window);
    within the window, ranking is by NOM chemical plausibility, not by |ppm|.
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

    # Корректируем массовое окно для генерации нейтральных кандидатов
    # На входе — m/z наблюдаемых пиков (ионные массы), а генератор
    # работает в нейтральных массах. Сдвигаем окно на массу носителя заряда.
    ion_mode_lower = ion_mode.lower() if ion_mode else ""
    if ion_mode_lower in ("[m-h]-", "m-h", "mh-"):
        gen_min = mass_min_local + CHEM.proton_mass
        gen_max = mass_max_local + CHEM.proton_mass
    elif ion_mode_lower in ("[m]+", "m+", "[m+]"):
        gen_min = mass_min_local + CHEM.electron_mass
        gen_max = mass_max_local + CHEM.electron_mass
    elif ion_mode_lower in ("[m+h]+", "m+h", "mh+"):
        gen_min = mass_min_local - CHEM.proton_mass
        gen_max = mass_max_local - CHEM.proton_mass
    else:
        gen_min, gen_max = mass_min_local, mass_max_local

    # Генерируем кандидатов (нейтральные массы)
    candidates = _generate_candidate_formulas(
        mass_min=gen_min,
        mass_max=gen_max,
        cfg=search_config,
        mode=brutto_generation_mode,
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
    # Новая колонка: все формулы-кандидаты в пределах ppm-окна (фича #2)
    table["all_candidates"] = None

    for idx, row in table.iterrows():
        mass_obs = float(row["mass"])

        # считаем ppm-разницу по ИОННЫМ массам
        ppm = (cand_masses_ion - mass_obs) / mass_obs * 1e6
        abs_ppm = np.abs(ppm)

        mask = abs_ppm <= rel_error_ppm
        if not mask.any():
            continue

        global_indices = np.where(mask)[0]

        # NOM-приоритизация: выбираем лучшую формулу по хим. правдоподобию
        # (ppm внутри окна НЕ учитывается как критерий — только как допуск)
        best_local: int | None = None
        best_score = float("inf")

        # Изотопный фильтр: измерить реальное M+1/M один раз для пика
        m1_real: float | None = None
        if isotope_filter and original is not None:
            try:
                m1_real = _measure_m1_ratio(mass_obs, original)
            except Exception:
                m1_real = None

        for li in global_indices:
            formula_str = cand_formulas[li]
            try:
                counts = parse_formula(formula_str)
            except Exception:
                continue
            c_val = counts.get("C", 0)
            if c_val <= 0:
                continue
            hc = counts.get("H", 0) / c_val
            oc = counts.get("O", 0) / c_val
            nc = counts.get("N", 0) / c_val
            ndist = _nom_distance(hc, oc)
            dbe = dbe_from_counts(counts)
            # Штраф за DBE > 20 (выше верхней границы типичного NOM)
            dbe_pen = (dbe - 20) * 0.5 if dbe > 20 else 0.0
            # Штраф за высокий N/C (N > 30% от C редко для NOM)
            nc_pen = nc * 2.0 if nc > 0.3 else 0.0
            # Штраф за высокий абсолютный N при низком O
            # (N>3 и O/N<0.5 — химически нехарактерно для NOM)
            n_abs = counts.get("N", 0)
            o_abs = counts.get("O", 0)
            if n_abs > 3 and (o_abs == 0 or o_abs / n_abs < 0.5):
                n_abs_pen = (n_abs - 3) * 2.0
            else:
                n_abs_pen = 0.0
            # Изотопный фильтр ¹³C: штраф при расхождении M+1/M > 20%
            iso_pen = 0.0
            if m1_real is not None and m1_real > 0:
                m1_theor = _beynon_m1_ratio(counts)
                if m1_theor > 0:
                    dev = abs(m1_real - m1_theor) / m1_theor
                    if dev > _ISOTOPE_TOLERANCE:
                        iso_pen = _ISOTOPE_PENALTY
            # score = nom_weight * nom_dist + dbe_pen + nc_pen + n_abs_pen + iso_pen
            # (ppm НЕ входит — в пределах окна все кандидаты равноправны по массе)
            score = nom_weight * ndist + dbe_pen + nc_pen + n_abs_pen + iso_pen
            if score < best_score:
                best_score = score
                best_local = li
        if best_local is None:
            # fallback: если ни один кандидат не прошёл парсинг — берём первый по ppm
            sorted_order = np.argsort(abs_ppm[mask])
            chosen_global = int(global_indices[sorted_order[0]])
        else:
            chosen_global = int(best_local)

        # Сортируем кандидатов по возрастанию |ppm| (для all_candidates)
        sorted_order = np.argsort(abs_ppm[mask])
        sorted_global = global_indices[sorted_order]

        # Сохраняем все формулы-кандидаты (упорядочены по ppm)
        all_candidates_list = [str(cand_formulas[i]) for i in sorted_global]
        table.at[idx, "all_candidates"] = all_candidates_list

        # Лучшая формула (по выбранному критерию: NOM-приоритет или минимальный ppm)
        best_formula = str(cand_formulas[chosen_global])
        table.at[idx, "brutto"] = best_formula
        table.at[idx, "assign"] = True

    src.table = table
    return src

def assign_formulas_nomspectra(
    src,
    *,
    brutto_dict=None,
    rel_error=0.5,
    sign="-",
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
        raise TypeError(f"Некорректный формат файла {src}")

    if brutto_dict is None:
        brutto_dict = DEFAULT_BRUTTO_DICT
    elif not isinstance(brutto_dict, dict):
        raise TypeError("brutto_dict должен быть dict с диапазонами по элементам")

    for el, bounds in brutto_dict.items():
        if not (isinstance(bounds, (tuple, list)) and len(bounds) == 2):
            raise ValueError(
                f"Для элемента {el!r} ожидается (min, max), получено {bounds!r}"
            )

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
        warnings.warn(
            "Ни одной брутто-формулы не назначено (assign == False для всех пиков)"
        )

    return src

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
    max_consecutive_misses=3,
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
    max_consecutive_misses : int, optional
        Stop probing the series after this many consecutive missed (gap)
        steps.  Avoids wasteful loops for molecules with few functional
        groups when ``allow_gaps=True``.  Must be >= 1.  Default 3.

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
        If ``ppm_tol <= 0``, if ``max_groups``/``min_series_length``/
        ``max_consecutive_misses`` are below 1, or if required columns
        are missing from ``src``/``deriv``.

    Notes
    -----
    The series length is the last *found* step (1-based): observing steps
    1, 2, 3, 5 yields ``n_groups = 4`` recorded as length 5 with step 4
    listed under ``missing``.
    """

    if ppm_tol <= 0:
        raise ValueError(f"ppm_tol должно быть > 0, получено {ppm_tol}")
    if max_groups < 1 or min_series_length < 1 or max_consecutive_misses < 1:
        raise ValueError(
            f"max_groups ({max_groups}), min_series_length ({min_series_length}) "
            f"и max_consecutive_misses ({max_consecutive_misses}) "
            "должны быть >= 1"
        )
    required_src = ["brutto", "mass", "assign"]
    missing_src = [c for c in required_src if c not in src.table.columns]
    if missing_src:
        raise ValueError(f"В src не хватает столбца {missing_src}")
    required_deriv = ["mass", "intensity"]
    missing_deriv = [c for c in required_deriv if c not in deriv.table.columns]
    if missing_deriv:
        raise ValueError(
            f"В deriv.table отсутствуют колонки {missing_deriv}. "
            "Файл дериватизированного спектра некорректен."
        )

    mz_deriv = deriv.table["mass"].values
    records = []

    for _, row in src.table.iterrows():
        if not row.get("assign", False):
            continue

        m0_obs = row["mass"]
        brutto = row.get("brutto", "")
        # Compute theoretical m/z from assigned brutto formula (eliminates source mass error)
        try:
            counts = parse_formula(str(brutto))
            exact_neutral = exact_mass_from_counts(counts)
            m0_theor = _neutral_to_ion_mass(exact_neutral, CHEM.default_ion_mode)
            # Sanity check: if theoretical diverges >1000 ppm from observed,
            # formula is inconsistent with mass → fall back to observed
            if abs(m0_theor - m0_obs) / m0_obs > 0.001:
                m0_theor = m0_obs
        except Exception:
            m0_theor = m0_obs  # fallback to observed mass
        found_steps = []
        series_mz = []
        consecutive_misses = 0

        for step in range(1, max_groups + 1):
            target = m0_theor + step * delta
            idx = _find_peak(mz_deriv, target, ppm_tol)

            if idx is not None:
                found_steps.append(step)
                series_mz.append(float(mz_deriv[idx]))
                consecutive_misses = 0
            else:
                series_mz.append(None)
                consecutive_misses += 1
                if consecutive_misses >= max_consecutive_misses:
                    break
                if not allow_gaps and found_steps:
                    break

        if not found_steps:
            n_groups = 0
            missing_steps = []
            trimmed = []
        else:
            n_groups = max(found_steps)
            all_steps = set(range(1, n_groups + 1))
            missing_steps = sorted(all_steps - set(found_steps))
            trimmed = series_mz[:n_groups]

        if n_groups >= min_series_length:
            records.append(
                {
                    "mass_src": m0_obs,
                    "brutto": row.get("brutto", ""),
                    "n_groups": n_groups,
                    "steps_found": found_steps,
                    "missing": missing_steps,
                    "series_mz": trimmed,
                }
            )

        if not found_steps:
            continue

    return pd.DataFrame(
        records,
        columns=[
            "mass_src",
            "brutto",
            "n_groups",
            "steps_found",
            "missing",
            "series_mz",
        ],
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
        src.table.loc[
            src.table.get("assign", pd.Series(False, index=src.table.index)) == True
        ][["mass", "intensity", "brutto", "all_candidates"]]
        .copy()
        .reset_index(drop=True)
    )
    base["mass_key"] = base["mass"].round(4)

    def _enrich(df, prefix):
        if df.empty:
            return pd.DataFrame(
                columns=["mass_key", f"n_{prefix}", f"missing_{prefix}"]
            )
        tmp = df[["mass_src", "n_groups", "missing"]].copy()
        tmp["mass_key"] = tmp["mass_src"].round(4)
        return tmp.rename(
            columns={
                "n_groups": f"n_{prefix}",
                "missing": f"missing_{prefix}",
            }
        )[["mass_key", f"n_{prefix}", f"missing_{prefix}"]]

    result = base.merge(_enrich(df_dmet, "dmet"), on="mass_key", how="left").merge(
        _enrich(df_dacet, "dacet"), on="mass_key", how="left"
    )

    result["n_dmet"] = result["n_dmet"].fillna(0).astype(int)
    result["n_dacet"] = result["n_dacet"].fillna(0).astype(int)
    result["N_COOH"] = result["n_dmet"]
    result["N_OH"] = result["n_dacet"]

    return (
        result[
            [
                "mass",
                "intensity",
                "brutto",
                "all_candidates",
                "N_COOH",
                "N_OH",
                "missing_dmet",
                "missing_dacet",
            ]
        ]
        .sort_values("mass")
        .reset_index(drop=True)
    )

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
        logger.info("[%s] Серии не найдены.", label)
        return

    has_missing = df_series[df_series["missing"].apply(len) > 0]
    display_df = has_missing.head(max_rows)

    if display_df.empty:
        logger.info("[%s] Пропущенных пиков в сериях нет.", label)
        return

    n_rows = len(display_df)
    fig, axes = plt.subplots(
        n_rows,
        1,
        figsize=(figsize_per_row[0], figsize_per_row[1] * n_rows + 1.5),
        squeeze=False,
    )
    fig.suptitle(
        f"Серии {label} с пропущенными пиками "
        f"(delta_m = {delta:.5f} Da, допуск {ppm_tol} ppm)",
        fontsize=11,
        fontweight="bold",
    )

    mz_src = src.table["mass"].values
    int_src = src.table["intensity"].values
    mz_deriv = deriv.table["mass"].values
    int_deriv = deriv.table["intensity"].values

    for ax_idx, (_, row) in enumerate(display_df.iterrows()):
        ax = axes[ax_idx][0]
        m0 = row["mass_src"]
        n_groups = row["n_groups"]
        missing = set(row["missing"])
        series = row["series_mz"]

        idx_s = _find_peak(mz_src, m0, ppm_tol * 10)
        i0 = float(int_src[idx_s]) if idx_s is not None else 1.0

        max_i = i0
        for mz_step in series:
            if mz_step is not None:
                idx_d = _find_peak(mz_deriv, mz_step, ppm_tol * 2)
                if idx_d is not None:
                    max_i = max(max_i, float(int_deriv[idx_d]))

        bar_w = delta * 0.08
        ax.bar(m0, i0, width=bar_w, color="steelblue", alpha=0.85)

        for step, mz_step in enumerate(series, start=1):
            expected = m0 + step * delta
            if step in missing or mz_step is None:
                ax.axvline(
                    x=expected,
                    color="crimson",
                    linestyle="--",
                    linewidth=1.0,
                    alpha=0.75,
                )
                ax.text(
                    expected,
                    max_i * 0.55,
                    f"n={step}",
                    color="crimson",
                    fontsize=7,
                    ha="center",
                    va="bottom",
                )
            else:
                idx_d = _find_peak(mz_deriv, float(mz_step), ppm_tol * 2)
                i_step = float(int_deriv[idx_d]) if idx_d is not None else max_i * 0.1
                ax.bar(mz_step, i_step, width=bar_w, color="forestgreen", alpha=0.8)
                ax.text(
                    mz_step,
                    i_step + max_i * 0.02,
                    f"n={step}",
                    color="darkgreen",
                    fontsize=7,
                    ha="center",
                    va="bottom",
                )

        ax.set_xlim(m0 - delta * 0.5, m0 + (n_groups + 1) * delta)
        ax.set_ylim(0, max_i * 1.25)
        ax.set_ylabel("I", fontsize=8)
        ax.set_title(
            f"{row['brutto']}   m/z={m0:.4f}   "
            f"серия 1..{n_groups}   пропущено: {sorted(missing)}",
            fontsize=9,
        )
        ax.tick_params(labelsize=7)

    fig.legend(
        handles=[
            mpatches.Patch(color="steelblue", label="Исходный пик"),
            mpatches.Patch(color="forestgreen", label="Найденный пик серии"),
            mpatches.Patch(
                color="crimson", label="Пропущенный пик (ожидаемая позиция)"
            ),
        ],
        loc="lower center",
        ncol=3,
        fontsize=9,
        frameon=True,
        bbox_to_anchor=(0.5, 0),
    )
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("[%s] График сохранён: %s", label, save_path)
    else:
        plt.show()
    plt.close(fig)
