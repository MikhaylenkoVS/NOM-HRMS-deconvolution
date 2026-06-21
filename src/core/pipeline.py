from __future__ import annotations

"""
src/core/pipeline.py
====================
Полный пайплайн: загрузка → денойс → assign → find_series → итог.

Режимы запуска
--------------
* Обычный вызов run_pipeline(...)  – возвращает PipelineRunResult, GUI может работать как раньше.
* test_mode=True                   – прогоняет set_01..set_05, выводит подробную статистику
                                     и возвращает список TestSetResult. GUI не запускается.

Запуск из командной строки для тест-режима:
    python -m src.core.pipeline --test
    python -m src.core.pipeline --test --sets-root data/test_sets
"""

import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from src.core.molecule import parse_formula
import pandas as pd

# ---------------------------------------------------------------------------
# Импорт зависимостей из spectrum_ops
# ---------------------------------------------------------------------------
try:
    from src.core.spectrum_ops import (
        Spectrum,
        load_spectrum,
        denoise,
        assign_formulas,
        find_series,
        build_result_table,
        visualize_series,
        DELTA_CD3,
        DELTA_CD3CO,
    )
    _IMPORT_ERROR: Optional[str] = None
except Exception as _e:
    _IMPORT_ERROR = str(_e)
    print(f"[PIPELINE] CRITICAL: не удалось импортировать spectrum_ops: {_e}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Вспомогательные утилиты
# ---------------------------------------------------------------------------

def _debug(msg: str) -> None:
    print(f"[DEBUG] {msg}", flush=True)


def _ppm_error(observed: float, theoretical: float) -> float:
    if theoretical == 0:
        return float("inf")
    return abs(observed - theoretical) / theoretical * 1e6


def _normalize_brutto(value) -> Optional[str]:
    """Нормализует брутто-формулу: убирает пробелы, переводит в верхний регистр."""
    import re
    if pd.isna(value):
        return None
    s = str(value).strip()
    if not s:
        return None
    # Канонический порядок: C H N O S P ...
    try:
        tokens = re.findall(r"([A-Z][a-z]?)(\d*)", s)
        counts: dict[str, int] = {}
        for elem, numstr in tokens:
            if not elem:
                continue
            n = int(numstr) if numstr else 1
            counts[elem] = counts.get(elem, 0) + n
        # Убираем нули
        counts = {k: v for k, v in counts.items() if v > 0}
        # Сортировка: C, H, потом остальные по алфавиту
        def sort_key(e: str) -> tuple:
            order = {"C": 0, "H": 1}
            return (order.get(e, 2), e)
        parts = []
        for elem in sorted(counts.keys(), key=sort_key):
            cnt = counts[elem]
            parts.append(elem if cnt == 1 else f"{elem}{cnt}")
        return "".join(parts)
    except Exception:
        return s.upper()


def _subtract_one_h(brutto: str) -> str:
    if not brutto:
        return brutto
    counts = parse_formula(brutto)
    h = counts.get('H', 0)
    if h <= 1:
        return brutto
    counts['H'] = h - 1
    # rebuild using Hill notation (or simple loop)
    # You can use the helper from molecule.py or write a small rebuild:
    parts = []
    for el in ['C', 'H'] + sorted(k for k in counts if k not in ('C', 'H')):
        if el in counts and counts[el] > 0:
            parts.append(el if counts[el] == 1 else f"{el}{counts[el]}")
    return "".join(parts)


def _match_row_by_mass(
    table: pd.DataFrame,
    mass_obs: float,
    ppm_tol: float,
    mass_col: str = "mass",
    require_assigned: bool = False,
) -> Optional[pd.Series]:
    """Ищет строку в таблице по массе с допуском ppm_tol.
    Возвращает ближайшую строку или None.
    """
    if table is None or table.empty:
        return None
    if mass_col not in table.columns:
        _debug(f"  _match_row_by_mass: колонка '{mass_col}' не найдена, доступны {list(table.columns)}")
        return None
    work = table.copy()
    work["_ppm"] = work[mass_col].astype(float).apply(
        lambda x: _ppm_error(float(x), float(mass_obs))
    )
    work = work.loc[work["_ppm"] <= ppm_tol].copy()
    if require_assigned:
        if "assign" not in work.columns:
            _debug("  _match_row_by_mass: require_assigned=True, но колонки 'assign' нет")
            return None
        work = work.loc[work["assign"] == True].copy()  # noqa: E712
    if work.empty:
        return None
    return work.sort_values("_ppm").iloc[0]


# ---------------------------------------------------------------------------
# Датаклассы статистики
# ---------------------------------------------------------------------------

@dataclass
class SeriesStats:
    rows: int = 0
    max_groups: int = 0
    missing_total: int = 0


@dataclass
class PipelineStats:
    src_loaded: int = 0
    dmet_loaded: int = 0
    dacet_loaded: int = 0

    src_denoised: int = 0
    dmet_denoised: int = 0
    dacet_denoised: int = 0

    assigned_count: int = 0
    assigned_ratio: float = 0.0

    dmet: SeriesStats = field(default_factory=SeriesStats)
    dacet: SeriesStats = field(default_factory=SeriesStats)

    result_rows: int = 0
    result_n_cooh_gt0: int = 0
    result_n_oh_gt0: int = 0


@dataclass
class PipelineRunResult:
    table: pd.DataFrame
    stats: PipelineStats
    messages: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Датаклассы тест-режима
# ---------------------------------------------------------------------------

@dataclass
class TestSetResult:
    set_name: str
    total_signals: int = 0
    denoised_kept: int = 0
    assigned_ok: int = 0

    # Серии
    dmet_found: int = 0
    dmet_matched: int = 0
    dmet_wrong: int = 0
    dacet_found: int = 0
    dacet_matched: int = 0
    dacet_wrong: int = 0

    # Ошибки
    errors: list[str] = field(default_factory=list)

    # Финальная таблица
    result_table: Optional[pd.DataFrame] = None
    assigned_only: Optional[pd.DataFrame] = None

    @property
    def denoise_recall(self) -> float:
        return self.denoised_kept / self.total_signals if self.total_signals else 0.0

    @property
    def assign_recall(self) -> float:
        return self.assigned_ok / self.total_signals if self.total_signals else 0.0


# ---------------------------------------------------------------------------
# Основной пайплайн
# ---------------------------------------------------------------------------

def run_pipeline(
    src_path=None,
    dmet_path=None,
    dacet_path=None,
    *,
    # Загрузка
    sep=",",
    load_mass_min: float = 0.0,
    load_mass_max: float = 1000.0,
    # Шумоподавление
    noise_force=10,
    noise_intensity=100,
    noise_quantile=None,
    # Назначение формул
    brutto_dict=None,
    rel_error: float = 1.0,
    sign: str = "-",
    assign_mass_min: float = 0,
    assign_mass_max: float = 1000,
    # Поиск серий
    ppm_tol: float = 5.0,
    max_groups: int = 20,
    allow_gaps: bool = True,
    # Визуализация
    visualize: bool = True,
    save_dmet=None,
    save_dacet=None,
    # Выходной файл
    output_csv=None,
    # Тест-режим
    test_mode: bool = False,
    test_sets_root=None,
):
    """
    Полный пайплайн определения числа -COOH и -OH групп.

    Если test_mode=True – игнорирует src_path/dmet_path/dacet_path,
    прогоняет set_01..set_05 из test_sets_root и возвращает список TestSetResult.
    GUI при этом не запускается.
    """

    # -----------------------------------------------------------------------
    # Проверка импорта
    # -----------------------------------------------------------------------
    if _IMPORT_ERROR:
        msg = f"[PIPELINE] Импорт spectrum_ops не удался: {_IMPORT_ERROR}"
        print(msg, file=sys.stderr)
        if not test_mode:
            stats = PipelineStats()
            return PipelineRunResult(table=pd.DataFrame(), stats=stats, messages=[msg])

    # -----------------------------------------------------------------------
    # ТЕСТ-РЕЖИМ
    # -----------------------------------------------------------------------
    if test_mode:
        return _run_test_mode(
            test_sets_root=test_sets_root,
            noise_force=noise_force,
            noise_intensity=noise_intensity,
            noise_quantile=noise_quantile,
            assign_mass_min=assign_mass_min,
            assign_mass_max=assign_mass_max,
            rel_error=rel_error,
            sign=sign,
            ppm_tol=ppm_tol,
            max_groups=max_groups,
            allow_gaps=allow_gaps,
        )

    # -----------------------------------------------------------------------
    # Валидация путей
    # -----------------------------------------------------------------------
    messages: list[str] = []
    for label, path in [("src", src_path), ("dmet", dmet_path), ("dacet", dacet_path)]:
        if path is None:
            messages.append(f"[PIPELINE] ОШИБКА: путь '{label}' не задан")
        elif not Path(path).exists():
            messages.append(f"[PIPELINE] ОШИБКА: файл не найден: {path}")
    if messages:
        for m in messages:
            print(m, file=sys.stderr)
        return PipelineRunResult(table=pd.DataFrame(), stats=PipelineStats(), messages=messages)

    stats = PipelineStats()

    # -----------------------------------------------------------------------
    # ШАГ 1: Загрузка спектров
    # -----------------------------------------------------------------------
    print("=" * 60)
    print("ШАГ 1: Загрузка спектров")
    print("=" * 60)
    _debug(f"src_path  = {src_path}")
    _debug(f"dmet_path = {dmet_path}")
    _debug(f"dacet_path= {dacet_path}")
    _debug(f"load_mass_min={load_mass_min}, load_mass_max={load_mass_max}")

    _mapper = {"mass": "mass", "intensity": "intensity"}
    try:
        src = load_spectrum(src_path, mapper=_mapper, sep=sep,
                            mass_min=load_mass_min, mass_max=load_mass_max,
                            metadata={"name": "src"})
        stats.src_loaded = len(src.table) if hasattr(src, "table") else 0
        _debug(f"src загружен: {stats.src_loaded} пиков")
    except Exception as e:
        msg = f"[PIPELINE] ОШИБКА загрузки src: {e}\n{traceback.format_exc()}"
        print(msg, file=sys.stderr)
        return PipelineRunResult(table=pd.DataFrame(), stats=stats, messages=[msg])

    try:
        dmet = load_spectrum(dmet_path, mapper=_mapper, sep=sep,
                             mass_min=load_mass_min, mass_max=load_mass_max,
                             metadata={"name": "dmet"})
        stats.dmet_loaded = len(dmet.table) if hasattr(dmet, "table") else 0
        _debug(f"dmet загружен: {stats.dmet_loaded} пиков")
    except Exception as e:
        msg = f"[PIPELINE] ОШИБКА загрузки dmet: {e}\n{traceback.format_exc()}"
        print(msg, file=sys.stderr)
        return PipelineRunResult(table=pd.DataFrame(), stats=stats, messages=[msg])

    try:
        dacet = load_spectrum(dacet_path, mapper=_mapper, sep=sep,
                              mass_min=load_mass_min, mass_max=load_mass_max,
                              metadata={"name": "dacet"})
        stats.dacet_loaded = len(dacet.table) if hasattr(dacet, "table") else 0
        _debug(f"dacet загружен: {stats.dacet_loaded} пиков")
    except Exception as e:
        msg = f"[PIPELINE] ОШИБКА загрузки dacet: {e}\n{traceback.format_exc()}"
        print(msg, file=sys.stderr)
        return PipelineRunResult(table=pd.DataFrame(), stats=stats, messages=[msg])

    print(f"  Загружено пиков:  src={stats.src_loaded},  dmet={stats.dmet_loaded},  dacet={stats.dacet_loaded}")

    # -----------------------------------------------------------------------
    # ШАГ 2a: Шумоподавление
    # -----------------------------------------------------------------------
    print()
    print("=" * 60)
    print("ШАГ 2a: Шумоподавление")
    print("=" * 60)
    _debug(f"noise_force={noise_force}, noise_intensity={noise_intensity}, noise_quantile={noise_quantile}")

    try:
        src = denoise(src, force=noise_force, intensity=noise_intensity, quantile=noise_quantile)
        stats.src_denoised = len(src.table) if hasattr(src, "table") else 0
        _debug(f"src после денойса: {stats.src_denoised} пиков")
    except Exception as e:
        msg = f"[PIPELINE] ОШИБКА денойса src: {e}\n{traceback.format_exc()}"
        print(msg, file=sys.stderr)
        messages.append(msg)

    try:
        dmet = denoise(dmet, force=noise_force, intensity=noise_intensity, quantile=noise_quantile)
        stats.dmet_denoised = len(dmet.table) if hasattr(dmet, "table") else 0
        _debug(f"dmet после денойса: {stats.dmet_denoised} пиков")
    except Exception as e:
        msg = f"[PIPELINE] ОШИБКА денойса dmet: {e}\n{traceback.format_exc()}"
        print(msg, file=sys.stderr)
        messages.append(msg)

    try:
        dacet = denoise(dacet, force=noise_force, intensity=noise_intensity, quantile=noise_quantile)
        stats.dacet_denoised = len(dacet.table) if hasattr(dacet, "table") else 0
        _debug(f"dacet после денойса: {stats.dacet_denoised} пиков")
    except Exception as e:
        msg = f"[PIPELINE] ОШИБКА денойса dacet: {e}\n{traceback.format_exc()}"
        print(msg, file=sys.stderr)
        messages.append(msg)

    print(f"  После шумоподавления: src={stats.src_denoised},  dmet={stats.dmet_denoised},  dacet={stats.dacet_denoised}")

    # -----------------------------------------------------------------------
    # ШАГ 2b: Назначение брутто-формул
    # -----------------------------------------------------------------------
    print()
    print("=" * 60)
    print("ШАГ 2b: Назначение брутто-формул исходному спектру")
    print("=" * 60)
    _debug(f"assign_formulas: mode=simple, rel_error={rel_error}, sign={sign}, "
           f"mass_min={assign_mass_min}, mass_max={assign_mass_max}")
    _debug(f"brutto_dict={'default' if brutto_dict is None else brutto_dict}")

    try:
        src = assign_formulas(
            src,
            mode="simple",
            brutto_dict=brutto_dict,
            rel_error_ppm=rel_error,
            sign=sign,
            mass_min=assign_mass_min,
            mass_max=assign_mass_max,
        )
        n_assigned = int(src.table.get("assign", pd.Series(dtype=bool)).sum())
        stats.assigned_count = n_assigned
        stats.assigned_ratio = n_assigned / stats.src_denoised if stats.src_denoised else 0.0
        _debug(f"assign_formulas результат: {n_assigned}/{stats.src_denoised} пиков назначено "
               f"({stats.assigned_ratio:.1%})")
        _debug(f"Колонки src.table после assign: {list(src.table.columns)}")
        # Превью первых 5 назначенных
        assigned_mask = src.table.get("assign", pd.Series(False, index=src.table.index))
        assigned_preview = src.table.loc[assigned_mask == True].head(5)  # noqa: E712
        if not assigned_preview.empty:
            _debug(f"Первые назначенные пики:\n{assigned_preview.to_string(index=False)}")
        else:
            _debug("ВНИМАНИЕ: назначенных пиков нет!")
    except Exception as e:
        msg = f"[PIPELINE] ОШИБКА assign_formulas: {e}\n{traceback.format_exc()}"
        print(msg, file=sys.stderr)
        messages.append(msg)
        n_assigned = 0

    print(f"  Назначено формул: {n_assigned} из {stats.src_denoised} пиков")

    # Применяем subtract_one_h к назначенным брутто
    _debug("Применяем subtract_one_h к brutto в src.table")
    try:
        if "brutto" in src.table.columns and "assign" in src.table.columns:
            mask = src.table["assign"] == True  # noqa: E712
            before = src.table.loc[mask, "brutto"].head(3).tolist()
            src.table.loc[mask, "brutto"] = src.table.loc[mask, "brutto"].apply(_subtract_one_h)
            after = src.table.loc[mask, "brutto"].head(3).tolist()
            _debug(f"subtract_one_h: до={before}, после={after}")
        else:
            _debug(f"ВНИМАНИЕ: нет колонок 'brutto'/'assign' в src.table, колонки: {list(src.table.columns)}")
    except Exception as e:
        msg = f"[PIPELINE] ОШИБКА subtract_one_h: {e}\n{traceback.format_exc()}"
        print(msg, file=sys.stderr)
        messages.append(msg)

    # Строим копию только с назначенными пиками
    try:
        assigned_only_table = src.table.loc[src.table["assign"] == True].reset_index(drop=True).copy()  # noqa: E712
        _debug(f"assigned_only: {len(assigned_only_table)} строк, колонки: {list(assigned_only_table.columns)}")
        if assigned_only_table.empty:
            _debug("КРИТИЧНО: assigned_only пуст – find_series вернёт пустой результат!")
    except Exception as e:
        _debug(f"ОШИБКА при создании assigned_only: {e}")
        assigned_only_table = pd.DataFrame()

    # -----------------------------------------------------------------------
    # ШАГ 3: Серии CD3 (N_COOH)
    # -----------------------------------------------------------------------
    print()
    print("=" * 60)
    print("ШАГ 3: Серии дейтерометилирования (-> N_COOH)")
    print("=" * 60)
    _debug(f"find_series: delta={DELTA_CD3:.5f}, ppm_tol={ppm_tol}, max_groups={max_groups}, allow_gaps={allow_gaps}")

    df_dmet = pd.DataFrame()
    try:
        df_dmet = find_series(
            src, dmet,
            delta=DELTA_CD3,
            ppm_tol=ppm_tol,
            max_groups=max_groups,
            allow_gaps=allow_gaps,
        )
        stats.dmet.rows = len(df_dmet)
        if not df_dmet.empty:
            stats.dmet.max_groups = int(df_dmet["n_groups"].max()) if "n_groups" in df_dmet.columns else 0
            if "missing" in df_dmet.columns:
                stats.dmet.missing_total = int(df_dmet["missing"].apply(len).sum())
        _debug(f"find_series(dmet): {len(df_dmet)} строк, колонки={list(df_dmet.columns) if not df_dmet.empty else '[]'}")
        if not df_dmet.empty:
            _debug(f"  max_groups={stats.dmet.max_groups}, missing_total={stats.dmet.missing_total}")
            _debug(f"Превью df_dmet:\n{df_dmet.head(3).to_string(index=False)}")
    except Exception as e:
        msg = f"[PIPELINE] ОШИБКА find_series(dmet): {e}\n{traceback.format_exc()}"
        print(msg, file=sys.stderr)
        messages.append(msg)

    print(f"  Соединений с сериями CD3: {len(df_dmet)}")
    if not df_dmet.empty:
        print(f"  Макс. N_COOH = {stats.dmet.max_groups}")
        if stats.dmet.missing_total:
            print(f"  ВНИМАНИЕ: Внутренних пропусков в сериях: {stats.dmet.missing_total}")

    # -----------------------------------------------------------------------
    # ШАГ 4: Серии CD3CO (N_OH)
    # -----------------------------------------------------------------------
    print()
    print("=" * 60)
    print("ШАГ 4: Серии дейтероацилирования (-> N_OH_total)")
    print("=" * 60)
    _debug(f"find_series: delta={DELTA_CD3CO:.5f}, ppm_tol={ppm_tol}, max_groups={max_groups}, allow_gaps={allow_gaps}")

    df_dacet = pd.DataFrame()
    try:
        df_dacet = find_series(
            src, dacet,
            delta=DELTA_CD3CO,
            ppm_tol=ppm_tol,
            max_groups=max_groups,
            allow_gaps=allow_gaps,
        )
        stats.dacet.rows = len(df_dacet)
        if not df_dacet.empty:
            stats.dacet.max_groups = int(df_dacet["n_groups"].max()) if "n_groups" in df_dacet.columns else 0
            if "missing" in df_dacet.columns:
                stats.dacet.missing_total = int(df_dacet["missing"].apply(len).sum())
        _debug(f"find_series(dacet): {len(df_dacet)} строк")
        if not df_dacet.empty:
            _debug(f"  max_groups={stats.dacet.max_groups}, missing_total={stats.dacet.missing_total}")
    except Exception as e:
        msg = f"[PIPELINE] ОШИБКА find_series(dacet): {e}\n{traceback.format_exc()}"
        print(msg, file=sys.stderr)
        messages.append(msg)

    print(f"  Соединений с сериями CD3CO: {len(df_dacet)}")
    if not df_dacet.empty:
        print(f"  Макс. N_OH_total = {stats.dacet.max_groups}")
        if stats.dacet.missing_total:
            print(f"  ВНИМАНИЕ: Внутренних пропусков в сериях: {stats.dacet.missing_total}")

    # -----------------------------------------------------------------------
    # ШАГ 5: Итоговая таблица
    # -----------------------------------------------------------------------
    print()
    print("=" * 60)
    print("ШАГ 5: Итоговая таблица N_COOH / N_OH")
    print("=" * 60)
    result = pd.DataFrame()
    try:
        result = build_result_table(src, df_dmet, df_dacet)
        stats.result_rows = len(result)
        if not result.empty:
            if "N_COOH" in result.columns:
                stats.result_n_cooh_gt0 = int((result["N_COOH"] > 0).sum())
            if "N_OH" in result.columns:
                stats.result_n_oh_gt0 = int((result["N_OH"] > 0).sum())
        _debug(f"build_result_table: {stats.result_rows} строк")
        _debug(f"Колонки результата: {list(result.columns)}")
        if not result.empty:
            _debug(f"Превью результата:\n{result.head(5).to_string(index=False)}")
    except Exception as e:
        msg = f"[PIPELINE] ОШИБКА build_result_table: {e}\n{traceback.format_exc()}"
        print(msg, file=sys.stderr)
        messages.append(msg)

    print(f"  Строк в таблице: {stats.result_rows}")
    if not result.empty:
        print(f"  Соединений с N_COOH > 0: {stats.result_n_cooh_gt0}")
        print(f"  Соединений с N_OH   > 0: {stats.result_n_oh_gt0}")

    # -----------------------------------------------------------------------
    # ШАГ 6: Визуализация
    # -----------------------------------------------------------------------
    if visualize:
        print()
        print("=" * 60)
        print("ШАГ 6: Визуализация пропущенных пиков")
        print("=" * 60)
        try:
            visualize_series(src, dmet, df_dmet,
                             delta=DELTA_CD3, label="дейтерометилирования",
                             ppm_tol=ppm_tol, save_path=save_dmet)
        except Exception as e:
            msg = f"[PIPELINE] ОШИБКА visualize dmet: {e}"
            print(msg, file=sys.stderr)
            messages.append(msg)
        try:
            visualize_series(src, dacet, df_dacet,
                             delta=DELTA_CD3CO, label="дейтероацилирования",
                             ppm_tol=ppm_tol, save_path=save_dacet)
        except Exception as e:
            msg = f"[PIPELINE] ОШИБКА visualize dacet: {e}"
            print(msg, file=sys.stderr)
            messages.append(msg)

    # -----------------------------------------------------------------------
    # Сохранение CSV
    # -----------------------------------------------------------------------
    if output_csv and not result.empty:
        try:
            result.to_csv(output_csv, index=False, sep=";", encoding="utf-8-sig")
            print(f"\nИтоговая таблица сохранена: {output_csv}")
            _debug(f"CSV сохранён в {output_csv}, строк={len(result)}")
        except Exception as e:
            msg = f"[PIPELINE] ОШИБКА сохранения CSV: {e}"
            print(msg, file=sys.stderr)
            messages.append(msg)

    return PipelineRunResult(table=result, stats=stats, messages=messages)


# ---------------------------------------------------------------------------
# ТЕСТ-РЕЖИМ
# ---------------------------------------------------------------------------

#: Параметры, которые используют интеграционные тесты
_TEST_DENOISE_KWARGS = dict(force=10.0, intensity=100, quantile=None)
_TEST_ASSIGN_KWARGS = dict(mode="simple", rel_error_ppm=0.5, mass_min=None, mass_max=None)
_TEST_SERIES_KWARGS = dict(ppm_tol=0.5, max_groups=20, allow_gaps=True, min_series_length=1)
_TEST_MATCH_PPM = 0.5
_DERIV_SPECS = [
    ("deutermethylated.csv", DELTA_CD3 if not _IMPORT_ERROR else 17.03448, "deutermethylated"),
    ("deuteroacylated.csv", DELTA_CD3CO if not _IMPORT_ERROR else 45.02939, "deuteroacylated"),
]


def _run_test_mode(
    test_sets_root=None,
    noise_force=10.0,
    noise_intensity=100,
    noise_quantile=None,
    assign_mass_min=None,
    assign_mass_max=None,
    rel_error=0.5,
    sign="-",
    ppm_tol=0.5,
    max_groups=20,
    allow_gaps=True,
) -> list[TestSetResult]:
    """Прогоняет пайплайн по каждому set_0N и выводит подробную статистику."""

    # Resolve roots
    if test_sets_root is None:
        # Автоопределение: ищем data/testsets относительно текущего файла
        candidate = Path(__file__).resolve().parents[2] / "data" / "test_sets"
        if candidate.exists():
            test_sets_root = candidate
        else:
            # fallback: текущая рабочая директория
            test_sets_root = Path.cwd() / "data" / "test_sets"
    test_sets_root = Path(test_sets_root)

    print("=" * 70)
    print("ТЕСТ-РЕЖИМ pipeline.py")
    print(f"  test_sets_root = {test_sets_root}")
    print(f"  exists         = {test_sets_root.exists()}")
    print("=" * 70)

    if not test_sets_root.exists():
        print(f"[TEST] ОШИБКА: директория тест-сетов не найдена: {test_sets_root}", file=sys.stderr)
        return []

    test_sets = sorted(p for p in test_sets_root.glob("set_0*") if p.is_dir())
    if not test_sets:
        print(f"[TEST] ОШИБКА: не найдено ни одного set* в {test_sets_root}", file=sys.stderr)
        return []

    print(f"  Найдено сетов: {len(test_sets)} → {[p.name for p in test_sets]}")
    print()

    results: list[TestSetResult] = []
    all_errors: list[str] = []

    for set_dir in test_sets:
        res = _run_single_test_set(
            set_dir=set_dir,
            noise_force=noise_force,
            noise_intensity=noise_intensity,
            noise_quantile=noise_quantile,
            assign_mass_min=assign_mass_min,
            assign_mass_max=assign_mass_max,
            rel_error=rel_error,
            sign=sign,
            ppm_tol=ppm_tol,
            max_groups=max_groups,
            allow_gaps=allow_gaps,
        )
        results.append(res)
        if res.errors:
            all_errors.extend([f"[{res.set_name}] {e}" for e in res.errors])

    # -----------------------------------------------------------------------
    # Итоговая сводная статистика
    # -----------------------------------------------------------------------
    print()
    print("=" * 70)
    print("ИТОГОВАЯ СВОДНАЯ СТАТИСТИКА (тест-режим)")
    print("=" * 70)
    header = (
        f"{'Set':<8}"
        f"{'Signals':>8}"
        f"{'Denoised':>10}"
        f"{'D-Rec%':>8}"
        f"{'Assigned':>10}"
        f"{'A-Rec%':>8}"
        f"{'DmetFnd':>9}"
        f"{'DmetOk':>8}"
        f"{'DacetFnd':>10}"
        f"{'DacetOk':>9}"
        f"{'Errors':>7}"
    )
    print(header)
    print("-" * 97)
    for r in results:
        dr = r.denoise_recall * 100
        ar = r.assign_recall * 100
        errs = len(r.errors)
        print(
            f"{r.set_name:<8}"
            f"{r.total_signals:>8}"
            f"{r.denoised_kept:>10}"
            f"{dr:>7.1f}%"
            f"{r.assigned_ok:>10}"
            f"{ar:>7.1f}%"
            f"{r.dmet_found:>9}"
            f"{r.dmet_matched:>8}"
            f"{r.dacet_found:>10}"
            f"{r.dacet_matched:>9}"
            f"{errs:>7}"
        )
    print("-" * 97)

    if all_errors:
        print()
        print(f"НАКОПЛЕННЫЕ ОШИБКИ ({len(all_errors)} шт.):")
        for e in all_errors:
            print(f"  {e}")

    # Проверки на пороги
    print()
    print("ПРОВЕРКА ПОРОГОВ:")
    MIN_DENOISE_RECALL = 0.90
    MIN_ASSIGN_RECALL = 0.75
    MAX_WRONG_RATIO = 0.20
    any_fail = False
    for r in results:
        if r.denoise_recall < MIN_DENOISE_RECALL:
            print(f"  FAIL denoise  {r.set_name}: {r.denoise_recall:.3f} < {MIN_DENOISE_RECALL}")
            any_fail = True
        if r.assign_recall < MIN_ASSIGN_RECALL:
            print(f"  FAIL assign   {r.set_name}: {r.assign_recall:.3f} < {MIN_ASSIGN_RECALL}")
            any_fail = True
        total = r.total_signals
        dmet_wrong_ratio = r.dmet_wrong / total if total else 0
        dacet_wrong_ratio = r.dacet_wrong / total if total else 0
        if dmet_wrong_ratio > MAX_WRONG_RATIO:
            print(f"  FAIL dmet_wrong {r.set_name}: {dmet_wrong_ratio:.3f} > {MAX_WRONG_RATIO}")
            any_fail = True
        if dacet_wrong_ratio > MAX_WRONG_RATIO:
            print(f"  FAIL dacet_wrong {r.set_name}: {dacet_wrong_ratio:.3f} > {MAX_WRONG_RATIO}")
            any_fail = True
    if not any_fail:
        print("  Все пороги пройдены ✓")

    return results


def _run_single_test_set(
    set_dir: Path,
    noise_force,
    noise_intensity,
    noise_quantile,
    assign_mass_min,
    assign_mass_max,
    rel_error,
    sign,
    ppm_tol,
    max_groups,
    allow_gaps,
) -> TestSetResult:
    """Прогоняет полный пайплайн для одного set_dir. Не бросает исключений наружу."""

    res = TestSetResult(set_name=set_dir.name)
    sep_line = "─" * 60

    print(sep_line)
    print(f"  SET: {set_dir.name}  ({set_dir})")
    print(sep_line)

    # ── загрузка annotations ──────────────────────────────────────────────
    ann_path = set_dir / "annotations.csv"
    molecules_path = set_dir / "molecules.csv"
    original_path = set_dir / "original.csv"
    dmet_path = set_dir / "deutermethylated.csv"
    dacet_path = set_dir / "deuteroacylated.csv"

    # Проверяем наличие файлов
    for fpath, label in [
        (ann_path, "annotations.csv"),
        (molecules_path, "molecules.csv"),
        (original_path, "original.csv"),
        (dmet_path, "deutermethylated.csv"),
        (dacet_path, "deuteroacylated.csv"),
    ]:
        if not fpath.exists():
            msg = f"файл не найден: {fpath}"
            print(f"  [ERROR] {msg}")
            res.errors.append(msg)

    if res.errors:
        print(f"  Пропускаем {set_dir.name} из-за отсутствующих файлов")
        return res

    # ── читаем annotations ───────────────────────────────────────────────
    try:
        ann = pd.read_csv(ann_path)
        _debug(f"{set_dir.name} annotations: {len(ann)} строк, колонки={list(ann.columns)}")
        required_cols = {"spectrum_type", "is_signal", "mass_obs", "compound_number", "formula"}
        missing_cols = required_cols - set(ann.columns)
        if missing_cols:
            msg = f"annotations.csv: отсутствуют колонки {sorted(missing_cols)}"
            print(f"  [ERROR] {msg}")
            res.errors.append(msg)
            return res
        ann_orig = ann[(ann["spectrum_type"] == "original") & (ann["is_signal"] == True)].copy()  # noqa: E712
        _debug(f"{set_dir.name} ann_orig (original+is_signal): {len(ann_orig)} строк")
        res.total_signals = len(ann_orig)
    except Exception as e:
        msg = f"ошибка чтения annotations: {e}"
        print(f"  [ERROR] {msg}")
        res.errors.append(msg)
        return res

    # ── читаем molecules ─────────────────────────────────────────────────
    molecules = pd.DataFrame()
    try:
        molecules = pd.read_csv(molecules_path)
        _debug(f"{set_dir.name} molecules: {len(molecules)} строк, колонки={list(molecules.columns)}")
    except Exception as e:
        msg = f"ошибка чтения molecules: {e}"
        print(f"  [WARN] {msg}")
        res.errors.append(msg)

    # ── загружаем спектры ────────────────────────────────────────────────
    try:
        src = load_spectrum(original_path, mass_min=100, mass_max=1000)
        _debug(f"{set_dir.name} original loaded: {len(src.table)} строк")
    except Exception as e:
        msg = f"load_spectrum original: {e}\n{traceback.format_exc()}"
        print(f"  [ERROR] {msg}")
        res.errors.append(msg)
        return res

    try:
        dmet_sp = load_spectrum(dmet_path, mass_min=100, mass_max=2000)
        _debug(f"{set_dir.name} dmet loaded: {len(dmet_sp.table)} строк")
    except Exception as e:
        msg = f"load_spectrum dmet: {e}\n{traceback.format_exc()}"
        print(f"  [ERROR] {msg}")
        res.errors.append(msg)
        dmet_sp = None

    try:
        dacet_sp = load_spectrum(dacet_path, mass_min=100, mass_max=2000)
        _debug(f"{set_dir.name} dacet loaded: {len(dacet_sp.table)} строк")
    except Exception as e:
        msg = f"load_spectrum dacet: {e}\n{traceback.format_exc()}"
        print(f"  [ERROR] {msg}")
        res.errors.append(msg)
        dacet_sp = None

    # ── денойс ──────────────────────────────────────────────────────────
    try:
        src_d = denoise(src, force=noise_force, intensity=noise_intensity, quantile=noise_quantile)
        _debug(f"{set_dir.name} denoised: {len(src_d.table)} строк (было {len(src.table)})")
    except Exception as e:
        msg = f"denoise: {e}\n{traceback.format_exc()}"
        print(f"  [ERROR] {msg}")
        res.errors.append(msg)
        src_d = src

    # Проверяем, сколько сигналов из annotations сохранилось после денойса
    denoised_kept = 0
    denoise_missing = []
    for _, row in ann_orig.iterrows():
        mass_obs = float(row["mass_obs"])
        match = _match_row_by_mass(src_d.table, mass_obs, ppm_tol=_TEST_MATCH_PPM, require_assigned=False)
        if match is None:
            denoise_missing.append({"mass_obs": mass_obs, "compound_number": row.get("compound_number")})
        else:
            denoised_kept += 1
    res.denoised_kept = denoised_kept
    denoise_recall = denoised_kept / res.total_signals if res.total_signals else 0.0
    _debug(f"{set_dir.name} denoise recall: {denoised_kept}/{res.total_signals} = {denoise_recall:.3f}")
    if denoise_missing:
        _debug(f"{set_dir.name} denoise missing (первые 3): {denoise_missing[:3]}")

    # ── assign_formulas ─────────────────────────────────────────────────
    try:
        src_a = assign_formulas(
            src_d,
            mode="simple",
            rel_error_ppm=rel_error,
            mass_min=assign_mass_min,
            mass_max=assign_mass_max,
        )
        _debug(f"{set_dir.name} assign_formulas: колонки={list(src_a.table.columns)}")
        if "assign" not in src_a.table.columns:
            msg = "assign_formulas не создала колонку 'assign'"
            print(f"  [ERROR] {msg}")
            res.errors.append(msg)
            return res
        if "brutto" not in src_a.table.columns:
            msg = "assign_formulas не создала колонку 'brutto'"
            print(f"  [ERROR] {msg}")
            res.errors.append(msg)
            return res
    except Exception as e:
        msg = f"assign_formulas: {e}\n{traceback.format_exc()}"
        print(f"  [ERROR] {msg}")
        res.errors.append(msg)
        return res

    # Применяем subtract_one_h
    try:
        assigned_table = src_a.table.copy()
        mask_assigned = assigned_table["assign"] == True  # noqa: E712
        assigned_table.loc[mask_assigned, "brutto"] = (
            assigned_table.loc[mask_assigned, "brutto"].apply(_subtract_one_h)
        )
        src_a.table = assigned_table
        _debug(f"{set_dir.name} subtract_one_h применён к {mask_assigned} строкам")
    except Exception as e:
        msg = f"subtract_one_h: {e}"
        print(f"  [WARN] {msg}")
        res.errors.append(msg)

    # assigned_only: только назначенные
    try:
        assigned_only = src_a.table.loc[src_a.table["assign"] == True].reset_index(drop=True).copy()  # noqa: E712
        _debug(f"{set_dir.name} assigned_only: {len(assigned_only)} строк")
        if assigned_only.empty:
            msg = "assigned_only пуст – find_series не найдёт серий!"
            print(f"  [WARN] {msg}")
            res.errors.append(msg)
    except Exception as e:
        msg = f"assigned_only: {e}"
        print(f"  [ERROR] {msg}")
        res.errors.append(msg)
        assigned_only = pd.DataFrame()

    res.assigned_only = assigned_only

    # Считаем assign_recall по annotations
    assigned_ok = 0
    assign_missing = []
    wrong_brutto = []
    for _, row in ann_orig.iterrows():
        mass_obs = float(row["mass_obs"])
        formula_true = _normalize_brutto(str(row["formula"]))
        match = _match_row_by_mass(src_a.table, mass_obs, ppm_tol=_TEST_MATCH_PPM, require_assigned=True)
        if match is None:
            assign_missing.append({"mass_obs": mass_obs, "compound_number": row.get("compound_number")})
            continue
        brutto_found = _normalize_brutto(match.get("brutto"))
        if brutto_found != formula_true:
            wrong_brutto.append({
                "mass_obs": mass_obs,
                "compound_number": row.get("compound_number"),
                "expected": formula_true,
                "actual": brutto_found,
            })
            continue
        assigned_ok += 1

    res.assigned_ok = assigned_ok
    assign_recall = assigned_ok / res.total_signals if res.total_signals else 0.0
    _debug(f"{set_dir.name} assign recall: {assigned_ok}/{res.total_signals} = {assign_recall:.3f}")
    if assign_missing:
        _debug(f"{set_dir.name} assign missing (первые 3): {assign_missing[:3]}")
    if wrong_brutto:
        _debug(f"{set_dir.name} wrong brutto (первые 3): {wrong_brutto[:3]}")

    print(f"  denoise recall: {denoised_kept}/{res.total_signals} = {denoise_recall:.1%}")
    print(f"  assign  recall: {assigned_ok}/{res.total_signals} = {assign_recall:.1%}")

    # ── Создаём Spectrum для assigned_only ───────────────────────────────
    # Нужен для find_series (передаём весь src_a, а не только assigned)
    # find_series берёт только назначенные пики по логике внутри себя
    # Но если у нас есть отдельный объект с assigned_only – используем его

    # Оборачиваем assigned_only в Spectrum
    try:
        src_assigned_only_sp = src_a.copy()
        src_assigned_only_sp.table = assigned_only
    except Exception as e:
        _debug(f"{set_dir.name} не удалось создать assigned_only Spectrum: {e}, используем src_a")
        src_assigned_only_sp = src_a

    # ── find_series: dmet ────────────────────────────────────────────────
    df_dmet_res = pd.DataFrame()
    if dmet_sp is not None and not assigned_only.empty:
        try:
            df_dmet_res = find_series(
                src_assigned_only_sp, dmet_sp,
                delta=DELTA_CD3,
                ppm_tol=ppm_tol,
                max_groups=max_groups,
                allow_gaps=allow_gaps,
                min_series_length=1,
            )
            res.dmet_found = len(df_dmet_res)
            _debug(f"{set_dir.name} find_series(dmet): {len(df_dmet_res)} строк")
            _debug(f"  Колонки: {list(df_dmet_res.columns) if not df_dmet_res.empty else '[]'}")
            if not df_dmet_res.empty:
                _debug(f"Превью df_dmet:\n{df_dmet_res.head(3).to_string(index=False)}")
        except Exception as e:
            msg = f"find_series(dmet): {e}\n{traceback.format_exc()}"
            print(f"  [ERROR] {msg}")
            res.errors.append(msg)
    else:
        if dmet_sp is None:
            _debug(f"{set_dir.name} dmet_sp=None, пропускаем find_series(dmet)")
        if assigned_only.empty:
            _debug(f"{set_dir.name} assigned_only пуст, пропускаем find_series(dmet)")

    # ── find_series: dacet ───────────────────────────────────────────────
    df_dacet_res = pd.DataFrame()
    if dacet_sp is not None and not assigned_only.empty:
        try:
            df_dacet_res = find_series(
                src_assigned_only_sp, dacet_sp,
                delta=DELTA_CD3CO,
                ppm_tol=ppm_tol,
                max_groups=max_groups,
                allow_gaps=allow_gaps,
                min_series_length=1,
            )
            res.dacet_found = len(df_dacet_res)
            _debug(f"{set_dir.name} find_series(dacet): {len(df_dacet_res)} строк")
        except Exception as e:
            msg = f"find_series(dacet): {e}\n{traceback.format_exc()}"
            print(f"  [ERROR] {msg}")
            res.errors.append(msg)

    # ── Сверка серий с annotations ───────────────────────────────────────
    for deriv_file, delta, deriv_label, sp_result, res_found_attr, res_matched_attr, res_wrong_attr in [
        ("deutermethylated.csv", DELTA_CD3, "dmet", df_dmet_res, "dmet_found", "dmet_matched", "dmet_wrong"),
        ("deuteroacylated.csv", DELTA_CD3CO, "dacet", df_dacet_res, "dacet_found", "dacet_matched", "dacet_wrong"),
    ]:
        if sp_result.empty:
            _debug(f"{set_dir.name} {deriv_label}: результат пустой, сверка невозможна")
            continue

        # Проверяем обязательные колонки
        expected_cols = {"mass_src", "brutto", "n_groups", "steps_found", "missing", "series_mz"}
        actual_cols = set(sp_result.columns)
        missing_result_cols = expected_cols - actual_cols
        if missing_result_cols:
            msg = f"{deriv_label} result: отсутствуют колонки {sorted(missing_result_cols)}, есть {sorted(actual_cols)}"
            print(f"  [WARN] {msg}")
            res.errors.append(msg)

        matched_series = 0
        wrong_length = []
        missing_series = []

        for _, ann_row in ann_orig.iterrows():
            mass_obs = float(ann_row["mass_obs"])
            compound_num = int(ann_row["compound_number"])

            # Ожидаемая длина серии из molecules.csv
            expected_len = None
            if not molecules.empty and "compound_number" in molecules.columns:
                mol_match = molecules.loc[molecules["compound_number"] == compound_num]
                if not mol_match.empty:
                    mol_row = mol_match.iloc[0]
                    if deriv_file == "deutermethylated.csv" and "carboxyl_count" in mol_row:
                        expected_len = int(mol_row["carboxyl_count"])
                    elif deriv_file == "deuteroacylated.csv" and "hydroxyl_count" in mol_row:
                        expected_len = int(mol_row["hydroxyl_count"])

            # Ищем строку в результате
            if "mass_src" not in sp_result.columns:
                continue
            diff = (sp_result["mass_src"] - mass_obs).abs()
            tol_da = mass_obs * _TEST_MATCH_PPM * 1e-6
            candidates = sp_result.loc[diff <= tol_da]
            if candidates.empty:
                missing_series.append({"mass_obs": mass_obs, "compound_number": compound_num, "expected_len": expected_len})
                continue

            matched_series += 1
            result_row = candidates.iloc[0]

            if expected_len is not None and "n_groups" in result_row:
                actual_len = int(result_row["n_groups"])
                if actual_len != expected_len:
                    wrong_length.append({
                        "mass_obs": mass_obs,
                        "compound_number": compound_num,
                        "expected": expected_len,
                        "actual": actual_len,
                    })

        setattr(res, res_matched_attr, matched_series)
        wrong_count = len(missing_series) + len(wrong_length)
        setattr(res, res_wrong_attr, wrong_count)

        wrong_ratio = wrong_count / res.total_signals if res.total_signals else 0.0
        _debug(
            f"{set_dir.name} {deriv_label}: "
            f"matched={matched_series}/{res.total_signals}, "
            f"missing={len(missing_series)}, wrong_len={len(wrong_length)}, "
            f"wrong_ratio={wrong_ratio:.3f}"
        )
        if missing_series:
            _debug(f"  missing_series (первые 3): {missing_series[:3]}")
        if wrong_length:
            _debug(f"  wrong_length (первые 3): {wrong_length[:3]}")
        print(
            f"  {deriv_label}: found={getattr(res, res_found_attr)}, "
            f"matched={matched_series}/{res.total_signals}, "
            f"wrong={wrong_count} ({wrong_ratio:.1%})"
        )

    # ── итоговая таблица для сета ────────────────────────────────────────
    result_table = pd.DataFrame()
    try:
        result_table = build_result_table(src_a, df_dmet_res, df_dacet_res)
        res.result_table = result_table
        _debug(f"{set_dir.name} build_result_table: {len(result_table)} строк")
    except Exception as e:
        msg = f"build_result_table: {e}"
        print(f"  [WARN] {msg}")
        res.errors.append(msg)

    # Итог по сету
    print(f"  ИТОГ {set_dir.name}: "
          f"denoise={denoise_recall:.1%}, assign={assign_recall:.1%}, "
          f"errors={len(res.errors)}")
    if res.errors:
        print(f"  ОШИБКИ ({len(res.errors)}):")
        for err in res.errors:
            short = err[:200].replace("\n", " | ")
            print(f"    • {short}")
    print()

    return res


# ---------------------------------------------------------------------------
# Точка входа командной строки
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="pipeline.py – анализ масс-спектров гуминовых веществ")
    parser.add_argument("--test", action="store_true", help="Запустить тест-режим по set_01..set_05")
    parser.add_argument("--sets-root", type=str, default=None, help="Путь к директории с тест-сетами")
    args = parser.parse_args()

    if args.test:
        run_pipeline(test_mode=True, test_sets_root=args.sets_root)
    else:
        print("Используйте --test для запуска тест-режима, или импортируйте run_pipeline() из кода.")