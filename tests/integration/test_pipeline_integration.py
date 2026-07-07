# /test_pipeline_integration.py
from pathlib import Path

import pandas as pd
import pytest

from src.core.spectrum_ops import (
    Spectrum,
    load_spectrum,
    denoise,
    assign_formulas,
    find_series,
)
from tests.unit.test_assign_formulas import normalize_brutto

from src.configs import CHEM, PIPELINE, PATHS

# ── Единый источник истины: src/configs/{chemistry,pipeline,paths}.json ──

# Параметры из pipeline.json -> test_mode
_TEST_CFG = PIPELINE.test_mode
REL_ERROR_PPM = _TEST_CFG["assign"]["rel_error_ppm"]
MATCH_PPM = _TEST_CFG["match_ppm"]
ASSIGN_MATCH_PPM = REL_ERROR_PPM  # тот же допуск для assign

# Сдвиги дериватизации из chemistry.json
DELTA_DEUTEROMETHYLATED = CHEM.derivatization_shifts["delta_cd3"]
DELTA_DEUTEROACYLATED = CHEM.derivatization_shifts["delta_cd3co"]

# Пути из paths.json
PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEST_SETS_ROOT = PROJECT_ROOT / PATHS.test_sets_dir
TEST_SETS = sorted([p for p in TEST_SETS_ROOT.glob("set_*") if p.is_dir()])

# Пороги из pipeline.json -> thresholds
MIN_DENOISE_RECALL = PIPELINE.thresholds["min_denoise_recall"]
MIN_ASSIGN_RECALL = PIPELINE.thresholds["min_assign_recall"]
MAX_SERIES_PROBLEM_RATIO = PIPELINE.thresholds["max_wrong_ratio"]

DEBUG_PREVIEW_ROWS = 5

DENOISE_KWARGS = dict(_TEST_CFG["denoise"])
ASSIGN_KWARGS = {
    "mode": _TEST_CFG["assign"]["mode"],
    "rel_error_ppm": ASSIGN_MATCH_PPM,
    "mass_min": 0,  # явно — см. TODO ниже
    "mass_max": 1000,
}
# TODO: Разобрать зависимость assign_formulas от диапазона масс.
#  Сейчас интеграционный тест зелёный только при явном указании:
#      ASSIGN_KWARGS["mass_min"] = 0
#      ASSIGN_KWARGS["mass_max"] = 1000
#  Без этих границ assign_formulas на denoised-спектрах сильно
#  недоназначал формулы (assign_recall ~ 30–50%), хотя на raw original
#  даёт 0.77–0.90.

# Имена файлов спектров из paths.json
_SF = PATHS.spectrum_files
DERIV_SPECS = [
    (_SF["deutermethylated"], DELTA_DEUTEROMETHYLATED, "deutermethylated"),
    (_SF["deuteroacylated"], DELTA_DEUTEROACYLATED, "deuteroacylated"),
]


def _debug(msg: str) -> None:
    print(f"[DEBUG] {msg}")


def _build_pipeline_log(
    set_name: str,
    deriv_name: str,
    total_signals: int,
    denoised_kept: int,
    assigned_ok: int,
    result_count: int,
    matched_series: int,
    wrong_count: int,
) -> str:
    denoise_cov = denoised_kept / total_signals if total_signals else 0.0
    assign_cov = assigned_ok / total_signals if total_signals else 0.0
    series_cov = matched_series / total_signals if total_signals else 0.0
    wrong_ratio = wrong_count / total_signals if total_signals else 0.0

    return (
        f"{set_name:<8} | "
        f"{deriv_name:<18} | "
        f"signals={total_signals:>3} | "
        f"denoise={denoised_kept:>3} ({denoise_cov:>6.1%}) | "
        f"assign={assigned_ok:>3} ({assign_cov:>6.1%}) | "
        f"series={result_count:>3} | "
        f"matched={matched_series:>3} ({series_cov:>6.1%}) | "
        f"wrong={wrong_count:>3} ({wrong_ratio:>6.1%})"
    )


def _print_pipeline_log(
    set_name: str,
    deriv_name: str,
    total_signals: int,
    denoised_kept: int,
    assigned_ok: int,
    result_count: int,
    matched_series: int,
    wrong_count: int,
) -> None:
    print(
        "\n"
        + _build_pipeline_log(
            set_name=set_name,
            deriv_name=deriv_name,
            total_signals=total_signals,
            denoised_kept=denoised_kept,
            assigned_ok=assigned_ok,
            result_count=result_count,
            matched_series=matched_series,
            wrong_count=wrong_count,
        )
    )


def _debug_candidates(label: str, df: pd.DataFrame, mass_obs: float, ppm: float = 0.5):
    diff_ppm = (df["mass"] - mass_obs) / mass_obs * 1e6
    cand = df.loc[diff_ppm.abs() <= ppm + 1e-6].copy()
    if cand.empty:
        _debug(f"{label}: no candidates for mass={mass_obs:.6f}")
        return

    cand["ppm_err"] = diff_ppm.loc[cand.index].abs()
    print(
        cand[["mass", "assign", "brutto", "ppm_err"]]
        .sort_values("ppm_err")
        .head(10)
        .to_string(index=False)
    )


def _ppm_error(observed: float, theoretical: float) -> float:
    return abs(observed - theoretical) / theoretical * 1e6


def _load_molecules_map(set_dir: Path) -> pd.DataFrame:
    molecules = pd.read_csv(set_dir / "molecules.csv")

    required_cols = {"compound_number", "carboxyl_count", "hydroxyl_count"}
    missing = required_cols - set(molecules.columns)
    assert (
        not missing
    ), f"{set_dir.name}/molecules.csv: отсутствуют колонки {sorted(missing)}"

    return molecules


def _expected_series_length(
    deriv_filename: str,
    molecule_row: pd.Series,
) -> int:
    if deriv_filename == "deutermethylated.csv":
        return int(molecule_row["carboxyl_count"])
    if deriv_filename == "deuteroacylated.csv":
        return int(molecule_row["hydroxyl_count"])
    raise AssertionError(f"Неизвестный тип дериватизации: {deriv_filename}")


def _match_table_row_by_mass(
    table: pd.DataFrame,
    mass_obs: float,
    ppm_tol: float,
    mass_col: str = "mass",
    require_assigned: bool = False,
):
    if table.empty:
        return None
    if mass_col not in table.columns:
        return None

    work = table.copy()
    work["_ppm"] = (
        work[mass_col]
        .astype(float)
        .apply(lambda x: _ppm_error(float(x), float(mass_obs)))
    )
    work = work.loc[work["_ppm"] <= ppm_tol].copy()

    if require_assigned:
        if "assign" not in work.columns:
            return None
        work = work.loc[work["assign"] == True].copy()

    if work.empty:
        return None

    return work.sort_values("_ppm").iloc[0]


def _match_result_row_by_mass(
    result: pd.DataFrame,
    mass_src: float,
    ppm_tol: float,
):
    if result.empty:
        return None

    diffs_ppm = result["mass_src"].apply(
        lambda x: _ppm_error(float(x), float(mass_src))
    )
    matched = result.loc[diffs_ppm <= ppm_tol]
    if matched.empty:
        return None

    return matched.sort_values("mass_src").iloc[0]


def _preview_table(df: pd.DataFrame, name: str) -> None:
    _debug(f"{name}: shape={df.shape}, columns={list(df.columns)}")
    if not df.empty:
        print(df.head(DEBUG_PREVIEW_ROWS).to_string(index=False))


@pytest.mark.parametrize("set_dir", TEST_SETS, ids=lambda p: p.name)
def test_pipeline_denoise_assign_find_series_on_existing_sets(set_dir: Path):
    assert TEST_SETS_ROOT.exists(), f"Не найдена папка test_sets: {TEST_SETS_ROOT}"

    _debug(f"=== START SET {set_dir.name} ===")

    ann = pd.read_csv(set_dir / "annotations.csv")
    molecules = _load_molecules_map(set_dir)

    required_ann_cols = {
        "spectrum_type",
        "is_signal",
        "mass_obs",
        "compound_number",
        "formula",
    }
    missing_ann_cols = required_ann_cols - set(ann.columns)
    assert not missing_ann_cols, (
        f"{set_dir.name}/annotations.csv: отсутствуют колонки "
        f"{sorted(missing_ann_cols)}"
    )

    ann_orig_signal = ann[
        (ann["spectrum_type"] == "original") & (ann["is_signal"] == True)
    ].copy()

    # A. assign по raw original — как в test_assign_formulas
    src_raw = load_spectrum(set_dir / "original.csv", mass_min=100, mass_max=1000)

    assigned_raw = assign_formulas(
        src_raw,
        mode="simple",
        rel_error_ppm=0.5,
        mass_min=None,
        mass_max=None,
    )

    assigned_raw_df = assigned_raw.table.copy()
    assigned_raw_only = (
        assigned_raw_df.loc[assigned_raw_df["assign"] == True]
        .reset_index(drop=True)
        .copy()
    )

    _debug(f"{set_dir.name}: raw assigned rows = {len(assigned_raw_only)}")

    # считаем assign_recall_raw так же, как в test_assign_formulas:
    assigned_ok_raw = 0
    for _, ann_row in ann_orig_signal.iterrows():
        mass_obs = float(ann_row["mass_obs"])
        formula_true = normalize_brutto(str(ann_row["formula"]))

        diff_ppm = (assigned_raw_df["mass"] - mass_obs) / mass_obs * 1e6
        candidates = assigned_raw_df[diff_ppm.abs() <= 0.5 + 1e-6]
        if candidates.empty:
            continue

        cand_norm = candidates["brutto"].apply(normalize_brutto)
        if any(cand_norm == formula_true):
            assigned_ok_raw += 1

    assign_recall_raw = assigned_ok_raw / len(ann_orig_signal)
    _debug(
        f"{set_dir.name}: assign_recall_raw={assigned_ok_raw}/{len(ann_orig_signal)} "
        f"({assign_recall_raw:.3f})"
    )

    assert (
        not ann_orig_signal.empty
    ), f"{set_dir.name}: в annotations.csv нет signal-пиков original"

    _debug(f"{set_dir.name}: annotations rows={len(ann)}")
    _debug(f"{set_dir.name}: original signal rows={len(ann_orig_signal)}")
    _preview_table(
        ann_orig_signal[["mass_obs", "compound_number", "formula"]],
        f"{set_dir.name} ann_orig_signal",
    )

    src = load_spectrum(set_dir / "original.csv", mass_min=100, mass_max=1000)
    assert isinstance(
        src, Spectrum
    ), f"{set_dir.name}: load_spectrum(original) не вернул Spectrum"
    assert not src.table.empty, f"{set_dir.name}: original.csv после load_spectrum пуст"

    _preview_table(src.table, f"{set_dir.name} original loaded")

    denoised = denoise(src, **DENOISE_KWARGS)
    assert isinstance(
        denoised, Spectrum
    ), f"{set_dir.name}: denoise должен возвращать Spectrum"
    assert not denoised.table.empty, f"{set_dir.name}: после denoise спектр пуст"

    _preview_table(denoised.table, f"{set_dir.name} denoised")

    assigned = assign_formulas(denoised, **ASSIGN_KWARGS)
    assert isinstance(
        assigned, Spectrum
    ), f"{set_dir.name}: assign_formulas должен возвращать Spectrum"

    assigned_table = assigned.table.copy()

    assert (
        "assign" in assigned_table.columns
    ), f"{set_dir.name}: после assign_formulas нет колонки assign"
    assert (
        "brutto" in assigned_table.columns
    ), f"{set_dir.name}: после assign_formulas нет колонки brutto"

    _preview_table(assigned_table, f"{set_dir.name} assigned")

    assigned_only = assigned.copy()
    assigned_only.table = (
        assigned_table.loc[assigned_table["assign"] == True]
        .reset_index(drop=True)
        .copy()
    )

    assert (
        not assigned_only.table.empty
    ), f"{set_dir.name}: assign_formulas не назначил ни одной формулы"

    _debug(f"{set_dir.name}: raw assigned count = {len(assigned_raw_only)}")
    _debug(f"{set_dir.name}: denoised assigned count = {len(assigned_only.table)}")
    _preview_table(assigned_only.table, f"{set_dir.name} assigned_only")

    total_signals = len(ann_orig_signal)

    denoise_missing_cases = []
    assign_missing_cases = []
    wrong_brutto_cases = []

    denoised_kept = 0
    assigned_ok = 0

    for _, ann_row in ann_orig_signal.iterrows():
        mass_obs = float(ann_row["mass_obs"])
        compound_number = int(ann_row["compound_number"])
        formula_true = normalize_brutto(ann_row["formula"])

        den_row = _match_table_row_by_mass(
            denoised.table,
            mass_obs,
            ppm_tol=MATCH_PPM,
            mass_col="mass",
            require_assigned=False,
        )
        if den_row is None:
            denoise_missing_cases.append(
                {
                    "mass_obs": mass_obs,
                    "compound_number": compound_number,
                }
            )
            continue

        denoised_kept += 1

        ass_row = _match_table_row_by_mass(
            assigned_table,
            mass_obs,
            ppm_tol=ASSIGN_MATCH_PPM,
            mass_col="mass",
            require_assigned=True,
        )
        if ass_row is None:
            _debug(f"{set_dir.name}: MISS mass={mass_obs:.6f}, formula={formula_true}")
            _debug_candidates("RAW assigned_only", assigned_raw_only, mass_obs, 0.5)
            _debug_candidates("DENOISED assigned_table", assigned_table, mass_obs, 0.5)
            _debug_candidates("DENOISED raw_table", denoised.table, mass_obs, 0.5)
            assign_missing_cases.append(
                {
                    "mass_obs": mass_obs,
                    "compound_number": compound_number,
                }
            )
            continue

        formula_true = normalize_brutto(str(ann_row["formula"]))
        brutto_found = normalize_brutto(ass_row["brutto"])
        if brutto_found != formula_true:
            wrong_brutto_cases.append(
                {
                    "mass_obs": mass_obs,
                    "compound_number": compound_number,
                    "expected": formula_true,
                    "actual": brutto_found,
                }
            )
            continue

        assigned_ok += 1

    denoise_recall = denoised_kept / total_signals if total_signals else 0.0
    assign_recall = assigned_ok / total_signals if total_signals else 0.0

    _debug(
        f"{set_dir.name}: denoised_kept={denoised_kept}/{total_signals} ({denoise_recall:.1%})"
    )
    _debug(
        f"{set_dir.name}: assigned_ok={assigned_ok}/{total_signals} ({assign_recall:.1%})"
    )

    if denoise_missing_cases:
        _debug(f"{set_dir.name}: first denoise misses: {denoise_missing_cases[:3]}")
    if assign_missing_cases:
        _debug(f"{set_dir.name}: first assign misses: {assign_missing_cases[:3]}")
    if wrong_brutto_cases:
        _debug(f"{set_dir.name}: first wrong brutto cases: {wrong_brutto_cases[:3]}")

    assert denoise_recall >= MIN_DENOISE_RECALL, (
        f"{set_dir.name}: denoise потерял слишком много signal-пиков "
        f"original: {denoised_kept}/{total_signals} ({denoise_recall:.1%}). "
        f"Примеры: {denoise_missing_cases[:3] if denoise_missing_cases else 'нет'}"
    )

    assert assign_recall >= MIN_ASSIGN_RECALL, (
        f"{set_dir.name}: после assign_formulas слишком мало корректно "
        f"назначенных signal-пиков: {assigned_ok}/{total_signals} ({assign_recall:.1%}). "
        f"Примеры пропусков: {assign_missing_cases[:3] if assign_missing_cases else 'нет'}. "
        f"Примеры неправильных формул: {wrong_brutto_cases[:3] if wrong_brutto_cases else 'нет'}"
    )

    for deriv_filename, delta, deriv_label in DERIV_SPECS:
        _debug(f"{set_dir.name}: START deriv={deriv_filename}, delta={delta}")

        deriv = load_spectrum(set_dir / deriv_filename, mass_min=100, mass_max=2000)
        assert isinstance(
            deriv, Spectrum
        ), f"{set_dir.name}/{deriv_filename}: load_spectrum не вернул Spectrum"
        assert (
            not deriv.table.empty
        ), f"{set_dir.name}/{deriv_filename}: deriv спектр пуст"

        _preview_table(deriv.table, f"{set_dir.name} {deriv_label} loaded")

        result = find_series(
            src=assigned_only,
            deriv=deriv,
            delta=delta,
            ppm_tol=REL_ERROR_PPM,
            max_groups=20,
            allow_gaps=True,
            min_series_length=1,
        )

        assert (
            not result.empty
        ), f"{set_dir.name}/{deriv_filename}: find_series не нашел ни одной серии"

        expected_columns = [
            "mass_src",
            "brutto",
            "n_groups",
            "steps_found",
            "missing",
            "series_mz",
        ]
        assert list(result.columns) == expected_columns, (
            f"{set_dir.name}/{deriv_filename}: неожиданные колонки результата: "
            f"{list(result.columns)}"
        )

        _preview_table(result, f"{set_dir.name} {deriv_label} result")

        matched_series = 0
        wrong_length_cases = []
        missing_series_cases = []
        total_with_series = 0

        for _, ann_row in ann_orig_signal.iterrows():
            mass_obs = float(ann_row["mass_obs"])
            compound_number = int(ann_row["compound_number"])

            molecule_match = molecules.loc[
                molecules["compound_number"] == compound_number
            ]
            assert not molecule_match.empty, (
                f"{set_dir.name}: compound_number={compound_number} из annotations.csv "
                f"не найден в molecules.csv"
            )
            assert len(molecule_match) == 1, (
                f"{set_dir.name}: compound_number={compound_number} не уникален "
                f"в molecules.csv"
            )

            molecule_row = molecule_match.iloc[0]
            expected_len = _expected_series_length(deriv_filename, molecule_row)

            # Если целевых групп нет (expected_len = 0), серия не ожидается
            if expected_len == 0:
                continue

            total_with_series += 1

            row = _match_result_row_by_mass(result, mass_obs, MATCH_PPM)
            if row is None:
                missing_series_cases.append(
                    {
                        "mass_obs": mass_obs,
                        "compound_number": compound_number,
                        "expected_len": expected_len,
                    }
                )
                continue

            matched_series += 1

            assert isinstance(
                row["steps_found"], list
            ), f"{set_dir.name}/{deriv_filename}: steps_found должен быть list"
            assert isinstance(
                row["missing"], list
            ), f"{set_dir.name}/{deriv_filename}: missing должен быть list"
            assert isinstance(
                row["series_mz"], list
            ), f"{set_dir.name}/{deriv_filename}: series_mz должен быть list"

            if int(row["n_groups"]) != expected_len:
                wrong_length_cases.append(
                    {
                        "mass_obs": mass_obs,
                        "compound_number": compound_number,
                        "expected_len": expected_len,
                        "actual_len": int(row["n_groups"]),
                        "steps_found": row["steps_found"],
                        "missing": row["missing"],
                    }
                )

        wrong_count = len(missing_series_cases) + len(wrong_length_cases)
        _total_series = total_with_series if total_with_series else total_signals
        wrong_ratio = wrong_count / _total_series if _total_series else 0.0

        _debug(
            f"{set_dir.name}/{deriv_label}: matched_series={matched_series}/{total_with_series}, "
            f"wrong_count={wrong_count}/{_total_series} ({wrong_ratio:.1%})"
        )

        if missing_series_cases:
            _debug(
                f"{set_dir.name}/{deriv_label}: first missing_series_cases: "
                f"{missing_series_cases[:3]}"
            )
        if wrong_length_cases:
            _debug(
                f"{set_dir.name}/{deriv_label}: first wrong_length_cases: "
                f"{wrong_length_cases[:3]}"
            )

        assert wrong_ratio <= MAX_SERIES_PROBLEM_RATIO, (
            f"{set_dir.name}/{deriv_filename}: слишком много пиков с неправильной длиной "
            f"серии или без серии: {wrong_count}/{total_signals} ({wrong_ratio:.1%}). "
            f"NO_SERIES examples={missing_series_cases[:3] if missing_series_cases else 'нет'}; "
            f"BAD_LEN examples={wrong_length_cases[:3] if wrong_length_cases else 'нет'}"
        )

        _print_pipeline_log(
            set_name=set_dir.name,
            deriv_name=deriv_label,
            total_signals=total_signals,
            denoised_kept=denoised_kept,
            assigned_ok=assigned_ok,
            result_count=len(result),
            matched_series=matched_series,
            wrong_count=wrong_count,
        )

    _debug(f"=== END SET {set_dir.name} ===")
