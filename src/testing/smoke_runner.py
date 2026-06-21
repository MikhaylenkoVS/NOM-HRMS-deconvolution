# ============================================================
# src/testing/smoke_runner.py
# ============================================================
import os
import json
import time
import logging
import traceback
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

from src.testing.report_models import (
    CompoundExportResult, SetSmokeResult, SmokeSuiteResult
)
from src.testing.artifact_export import (
    plot_three_spectra, plot_series_grid, plot_histogram
)
from src.testing.structure_export import export_structures_for_compound

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Конфигурация
# ----------------------------------------------------------------------
PIPELINE_PARAMS = {
    'load_mass_min': 0.0,
    'load_mass_max': 1000.0,
    'noise_force': 10,
    'noise_intensity': 100,
    'rel_error': 0.5,
    'sign': '-',
    'assign_mass_min': 0,
    'assign_mass_max': 1000,
    'ppm_tol': 0.5,
    'max_groups': 20,
    'allow_gaps': True,
    'visualize': False,
}

# ----------------------------------------------------------------------
def run_smoke_suite(data_root: Path, output_root: Path) -> SmokeSuiteResult:
    """
    Прогоняет полный smoke-тест по всем set_01..set_05.
    Возвращает SmokeSuiteResult с итогами.
    """
    suite = SmokeSuiteResult()
    suite.started_at = time.strftime('%Y-%m-%d %H:%M:%S')
    set_dirs = sorted([d for d in data_root.glob("set_*") if d.is_dir()])
    logger.info(f"Found {len(set_dirs)} test sets: {[d.name for d in set_dirs]}")

    for set_dir in set_dirs:
        set_name = set_dir.name
        set_out = output_root / set_name
        set_out.mkdir(parents=True, exist_ok=True)
        try:
            set_result = _run_one_set(set_dir, set_out)
        except Exception as e:
            logger.error(f"Unhandled error for {set_name}: {e}")
            tb = traceback.format_exc()
            set_result = SetSmokeResult(
                set_name=set_name,
                error=tb,
                artifacts_dir=set_out,
            )
        suite.sets.append(set_result)

    suite.overall_success = all(s.success for s in suite.sets)
    suite.finished_at = time.strftime('%Y-%m-%d %H:%M:%S')
    _export_summary(suite, output_root)
    return suite

def _run_one_set(set_dir: Path, output_dir: Path) -> SetSmokeResult:
    from src.core.pipeline import run_pipeline
    set_name = set_dir.name
    result = SetSmokeResult(set_name=set_name, artifacts_dir=output_dir)
    logger.info(f"--- Processing {set_name} ---")

    # Пути к входным файлам
    orig_csv = set_dir / "original.csv"
    dmet_csv = set_dir / "deutermethylated.csv"
    dacet_csv = set_dir / "deuteroacylated.csv"

    # Проверка существования
    for path, label in [(orig_csv, "original"), (dmet_csv, "dmet"), (dacet_csv, "dacet")]:
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")

    # 1. Прогон пайплайна
    try:
        pipe_result = run_pipeline(
            src_path=str(orig_csv),
            dmet_path=str(dmet_csv),
            dacet_path=str(dacet_csv),
            output_csv=None,   # сохраним сами
            **PIPELINE_PARAMS,
        )
        result.pipeline_success = True
        # Сохраняем таблицу
        table = pipe_result.table
        result_table_path = output_dir / "result_table.csv"
        table.to_csv(result_table_path, index=False, sep=';')
        result.result_table_path = result_table_path
    except Exception as e:
        logger.error(f"Pipeline failed for {set_name}: {e}")
        result.pipeline_success = False
        result.error = traceback.format_exc()
        result.success = False
        return result

    # 2. Графики
    try:
        spectra_png = output_dir / "spectra.png"
        plot_three_spectra(str(orig_csv), str(dmet_csv), str(dacet_csv), spectra_png,
                           title=f"Spectra for {set_name}")
        result.spectra_plot_path = spectra_png
    except Exception as e:
        logger.warning(f"Failed to plot spectra for {set_name}: {e}")

    # Для серий нужно получить df_dmet/df_dacet и deriv-спектры.
    # Пайплайн их не возвращает напрямую, но мы можем вытащить из его внутренностей.
    # Чтобы не менять API, выполним find_series повторно с теми же параметрами.
    try:
        from src.core.spectrum_ops import load_spectrum, denoise, find_series, DELTA_CD3, DELTA_CD3CO
        # Загружаем и шумим
        src_sp = load_spectrum(str(orig_csv), mass_min=0, mass_max=1000)
        src_sp = denoise(src_sp, force=10, intensity=100)
        dmet_sp = load_spectrum(str(dmet_csv), mass_min=0, mass_max=1000)
        dmet_sp = denoise(dmet_sp, force=10, intensity=100)
        dacet_sp = load_spectrum(str(dacet_csv), mass_min=0, mass_max=1000)
        dacet_sp = denoise(dacet_sp, force=10, intensity=100)

        # Назначаем формулы (используем тот же вызов, что и пайплайн)
        from src.core.spectrum_ops import assign_formulas_simple
        src_sp = assign_formulas_simple(src_sp, rel_error_ppm=0.5, ion_mode='[M-H]-')

        df_dmet = find_series(src_sp, dmet_sp, delta=DELTA_CD3, ppm_tol=0.5, max_groups=20, allow_gaps=True)
        df_dacet = find_series(src_sp, dacet_sp, delta=DELTA_CD3CO, ppm_tol=0.5, max_groups=20, allow_gaps=True)

        if not df_dmet.empty:
            series_dmet_png = output_dir / "series_dmet.png"
            plot_series_grid(df_dmet, dmet_sp.table['mass'].values, DELTA_CD3, 0.5, "dmet", series_dmet_png)
            result.series_dmet_path = series_dmet_png
        if not df_dacet.empty:
            series_dacet_png = output_dir / "series_dacet.png"
            plot_series_grid(df_dacet, dacet_sp.table['mass'].values, DELTA_CD3CO, 0.5, "dacet", series_dacet_png)
            result.series_dacet_path = series_dacet_png
    except Exception as e:
        logger.warning(f"Failed to plot series for {set_name}: {e}")

    # 3. Гистограммы
    try:
        hist_cooh = output_dir / "hist_n_cooh.png"
        plot_histogram(table, 'N_COOH', hist_cooh, f"{set_name} N_COOH distribution")
        result.hist_cooh_path = hist_cooh
        hist_oh = output_dir / "hist_n_oh.png"
        plot_histogram(table, 'N_OH', hist_oh, f"{set_name} N_OH distribution")
        result.hist_oh_path = hist_oh
    except Exception as e:
        logger.warning(f"Failed to plot histograms for {set_name}: {e}")

    # 4. Структуры для каждого соединения
    compounds_dir = output_dir / "compounds"
    compounds_dir.mkdir(exist_ok=True)
    for idx, row in table.iterrows():
        compound_idx = int(idx) + 1
        cpd_dir = compounds_dir / f"compound_{compound_idx}"
        cpd_result = CompoundExportResult(
            compound_index=compound_idx,
            mass=row['mass'],
            brutto=str(row['brutto']),
            n_cooh=int(row['N_COOH']),
            n_oh=int(row['N_OH']),
            structures_found=0,
        )
        try:
            paths = export_structures_for_compound(
                brutto=cpd_result.brutto,
                n_cooh=cpd_result.n_cooh,
                n_oh=cpd_result.n_oh,
                output_dir=cpd_dir,
            )
            cpd_result.structures_found = len([p for p in paths if p.suffix == '.mol'])
            cpd_result.structure_paths = paths
        except Exception as e:
            cpd_result.error = str(e)
            logger.warning(f"Structure export failed for {cpd_result.brutto}: {e}")
        result.compound_results.append(cpd_result)

    result.success = True
    return result

def _export_summary(suite: SmokeSuiteResult, output_root: Path):
    """Сохраняет summary.json и summary.csv."""
    summary = {
        "started_at": suite.started_at,
        "finished_at": suite.finished_at,
        "overall_success": suite.overall_success,
        "sets": []
    }
    rows = []
    for s in suite.sets:
        set_sum = {
            "set_name": s.set_name,
            "success": s.success,
            "pipeline_success": s.pipeline_success,
            "error": s.error,
            "num_compounds": len(s.compound_results),
            "compounds_with_structures": sum(1 for c in s.compound_results if c.structures_found > 0),
        }
        summary["sets"].append(set_sum)
        rows.append(set_sum)

    # JSON
    json_path = output_root / "summary.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # CSV
    csv_path = output_root / "summary.csv"
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    logger.info(f"Summary written to {json_path} and {csv_path}")