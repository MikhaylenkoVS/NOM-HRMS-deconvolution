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
from typing import Literal, Sequence

# ---------------------------------------------------------------------------
# Константы
# ---------------------------------------------------------------------------
DELTA_CD3   = 17.03448   # Da: сдвиг m/z при замене COOH -> COOCD3
DELTA_CD3CO = 45.02939   # Da: сдвиг m/z при замене OH  -> OCOCD3


# ===========================================================================
# Загрузка спектров
# ===========================================================================

logger = logging.getLogger(__name__)
ATOMIC_MASS = {
    "H": 1.00782503223,
    "C": 12.0,
    "N": 14.00307400443,
    "O": 15.99491461957
}


@dataclass
class FormulaSearchConfig:
    elements: tuple[str, ...] = ("C", "H", "O", "N")
    ranges: dict[str, tuple[int, int]] | None = None
    # Простые фильтры (можно менять под задачу)
    max_hc: float = 3.0        # H/C <= 3
    max_oc: float = 1.2        # O/C <= 1.2
    max_nc: float = 1.0        # N/C <= 1.0
    max_dbe: float = 30.0      # DBE <= 30
    min_c: int = 1             # минимум углеродов

    def __post_init__(self):
        if self.ranges is None:
            # дефолтные диапазоны, подстрой под свои данные
            self.ranges = {
                "C": (1, 50),
                "H": (4, 100),
                "O": (0, 20),
                "N": (0, 6),
            }
        for el in self.elements:
            if el not in self.ranges:
                raise ValueError(f"Для элемента {el!r} не задан диапазон в ranges")


def exact_mass_from_counts(counts: dict[str, int]) -> float:
    mass = 0.0
    for elem, n in counts.items():
        if n <= 0:
            continue
        mass += ATOMIC_MASS[elem] * n
    return mass


def dbe_from_counts(counts: dict[str, int]) -> float:
    """Простейшая DBE для CHON: DBE = 1 + C - H/2 + N/2."""
    c = counts.get("C", 0)
    h = counts.get("H", 0)
    n = counts.get("N", 0)
    return 1 + c - h / 2.0 + n / 2.0

def _row_to_brutto(row, element_order=None):
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
    mass_min=200.0,
    mass_max=700.0,
    metadata=None,
):
    """Загрузка спектра из CSV.

    Генерирует ValueError/KeyError при проблемах.
    GUI-слой решает, как эти ошибки показывать пользователю.
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
    """
    Удалить шум из спектра (обёртка вокруг Spectrum.noise_filter()).

    Приоритет параметров:
        1. intensity  — жёсткий абсолютный порог;
        2. quantile   — нижний квантиль (0-1);
        3. force      — авто-детекция уровня шума, множитель (по умолчанию 1.5).
    """
    return spec.noise_filter(force=force, intensity=intensity, quantile=quantile)


# ===========================================================================
# ЭТАП 2b: Назначение брутто-формул
# ===========================================================================

DEFAULT_BRUTTO_DICT = {
    'C': (0, 50),
    'H': (0, 100),
    'O': (0, 25),
    'N': (0, 10),
}

def _generate_candidate_formulas(
    mass_min: float,
    mass_max: float,
    cfg: FormulaSearchConfig,
    mode: str = "nom_like",
) -> list[tuple[str, float]]:
    """
    Генерирует список (formula_str, exact_mass) в окне [mass_min, mass_max]
    с небольшим запасом по краям.

    mode:
      - "nom_like": применять все химические фильтры (H/C, O/C, N/C, DBE, min C),
                    поведение, близкое к NOMspectra.
      - "soft": только диапазоны элементов и окно по массе, без химических фильтров.
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
    """
    Перевод нейтральной массы в m/z для заданного типа иона.
    Пока реализуем только несколько базовых вариантов.
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
    """
    Простое назначение формул без эмпирического списка:
    - генерирует CHON-формулы в заданном окне масс (нейтральные массы),
    - переводит их в m/z в соответствии с ion_mode,
    - подбирает лучшую формулу по минимальному ppm-отклонению в пределах rel_error_ppm.

    Заполняет src.table["brutto"] и src.table["assign"].
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
    """
    Гарантирует, что после назначения формул в src.table есть:
    - assign
    - brutto

    Если brutto отсутствует, но есть элементные столбцы (C, H, O, N, ...),
    восстанавливает brutto из них. Все прочие столбцы сохраняются.
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
    """
    Назначить брутто-формулы пикам исходного спектра через nomspectra.

    После назначения гарантируются столбцы:
        assign     — булево назначение
        brutto     — строковая формула (если её можно восстановить)
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
    """
    Найти индекс ближайшего пика в mz_array к target_mz с точностью ppm_tol.
    Возвращает int или None.
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
    """
    Найти серии дейтериационных пиков.

    Для каждого назначенного пика m_0 ищет цепочку:
        m_0 + 1*delta,  m_0 + 2*delta,  ...,  m_0 + n*delta
    в дериватизированном спектре.

    Правило определения длины серии
    ---------------------------------
    Длина серии = последний НАЙДЕННЫЙ шаг (1-based).
    Если между первым и последним шагами есть пропуски,
    они фиксируются в поле missing.
    Логика: "видим 1,2,3,5 -> считаем серию длиной 5".

    Параметры
    ---------
    src : Spectrum
        Исходный спектр с назначенными формулами.
    deriv : Spectrum
        Спектр дериватизированного образца.
    delta : float
        Ожидаемый сдвиг m/z на одну функциональную группу (Da).
    ppm_tol : float
        Допустимая погрешность совпадения масс (ppm).
    max_groups : int
        Максимально возможное число функциональных групп на молекулу.
    allow_gaps : bool
        True  — продолжать поиск при пропуске (рекомендуется).
        False — обрывать серию на первом пропуске.
    min_series_length : int
        Минимальная длина серии для включения в вывод.

    Возвращает
    ----------
    DataFrame:
        mass_src    — m/z пика в исходном спектре
        brutto      — назначенная брутто-формула
        n_groups    — длина серии (по последнему найденному шагу)
        steps_found — список найденных шагов (1-based)
        missing     — список пропущенных шагов ВНУТРИ серии
        series_mz   — список m/z для шагов 1..n_groups (None = пропуск)
    """
    print("DEBUG _find_peak function:", _find_peak)
    print("DEBUG delta:", delta, "ppm_tol:", ppm_tol)

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
            if step == 1:
                print("TRY", m0, target, _find_peak(mz_deriv, target, ppm_tol))

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
    """
    Собрать итоговую таблицу с числом -COOH и -OH для каждой брутто-формулы.

    Логика:
        N_COOH     = n_groups из df_dmet  (серия CD3,   delta = 17.034 Da)
        N_OH_total = n_groups из df_dacet (серия CD3CO, delta = 45.029 Da)
        N_OH       = N_OH_total - N_COOH  (чистые спиртовые ОН)

    Параметры
    ---------
    src : Spectrum
    df_dmet : DataFrame — результат find_series() для дейтерометилирования.
    df_dacet : DataFrame — результат find_series() для дейтероацилирования.

    Возвращает
    ----------
    DataFrame: mass, intensity, brutto, N_COOH, N_OH_total, N_OH,
               missing_dmet, missing_dacet
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

    impossible = result[result['N_OH_total'] < result['N_COOH']]
    if not impossible.empty:
        warnings.warn(
            f"{len(impossible)} пик(ов): N_OH_total < N_COOH. "
            "Возможна ошибка назначения серий или частичная дериватизация."
        )

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
    """
    Визуализировать серии с пропущенными пиками.

    Для каждого соединения строится лесенка ожидаемых пиков:
        синий      — исходный пик (m_0)
        зелёный    — найденный пик серии
        красный -- — пропущенный ожидаемый пик

    Параметры
    ---------
    src : Spectrum
    deriv : Spectrum
    df_series : DataFrame — результат find_series().
    delta : float — шаг серии (Da).
    label : str — подпись в заголовке.
    max_rows : int — максимальное число соединений для отображения.
    figsize_per_row : tuple — (ширина, высота) одной строки.
    ppm_tol : float — допуск поиска (ppm).
    save_path : str, optional — путь для сохранения рисунка.
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
