import pandas as pd
import logging
from nomspectra.spectrum import Spectrum
import warnings
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ---------------------------------------------------------------------------
# Константы
# ---------------------------------------------------------------------------
DELTA_CD3   = 17.03448   # Da: сдвиг m/z при замене COOH -> COOCD3
DELTA_CD3CO = 45.02939   # Da: сдвиг m/z при замене OH  -> OCOCD3

# ===========================================================================
# Загрузка спектров
# ===========================================================================

logger = logging.getLogger(__name__)

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
    'C': (4, 50),
    'H': (4, 100),
    'O': (0, 25),
    'N': (0, 2),
}


def assign_formulas(
    src,
    *,
    brutto_dict=None,
    rel_error=0.5,
    sign='-',
    mass_min=None,
    mass_max=None,
):
    """
    Назначить брутто-формулы пикам исходного спектра.

    После назначения добавляются столбцы:
        brutto     — строковая формула (например 'C15H24O12')
        calc_mass  — теоретическая масса

    Параметры
    ---------
    src : Spectrum
    brutto_dict : dict, optional
        Элементы и диапазоны их количеств. По умолчанию DEFAULT_BRUTTO_DICT.
    rel_error : float
        Допустимая погрешность назначения (ppm). Типично 0.5 ppm для Orbitrap.
    sign : {'-', '+', '0'}
        Режим ионизации.
    mass_min, mass_max : float, optional
        Ограничение диапазона масс при назначении.
    """
    if rel_error < 0:
        rel_error = abs(rel_error)
        warnings.warn("Relative error is negative")
    if mass_min > mass_max:
        (mass_min, mass_max) = (mass_max, mass_min)
        warnings.warn("Mass_max is less than mass_min")

    if not isinstance(src, Spectrum): raise TypeError(f'Некорректный формат файла {src}')
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
    if "assign" not in src.table.columns:
        raise RuntimeError(
            "После вызова src.assign в таблице src.table нет колонки 'assign'"
        )

    assign_col = src.table["assign"]
    if assign_col.dtype != bool:
        raise TypeError(f"Ожидается булевый столбец 'assign', получен dtype={assign_col.dtype}")

    n_assigned = int(assign_col.sum())
    if n_assigned > 0:
        src = src.calc_mass()
        src = src.brutto()
    elif n_assigned == 0: warnings.warn("Ни одной брутто-формулы не назначено (assign == False для всех пиков)")
    else:
        if hasattr(src, "calc_mass"):
            src.calc_mass()
        if hasattr(src, "brutto"):
            src.brutto()

    return src


# ===========================================================================
# Поиск серий
# ===========================================================================

def _find_peak(mz_array, target_mz, ppm_tol):
    """
    Найти индекс ближайшего пика в отсортированном mz_array к target_mz.
    Возвращает int или None.
    """
    idx = np.searchsorted(mz_array, target_mz)
    best = None
    best_err = ppm_tol
    for i in (idx - 1, idx):
        if 0 <= i < len(mz_array):
            err_ppm = abs(mz_array[i] - target_mz) / target_mz * 1e6
            if err_ppm < best_err:
                best_err = err_ppm
                best = i
    return best


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

    return pd.DataFrame(records)


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
