from pathlib import Path

import pandas as pd
import pytest
from src.core.spectrum_ops import load_spectrum, find_series, Spectrum

REL_ERROR_PPM = 1.2
MATCH_PPM = 1.2
ASSIGN_MATCH_PPM = 0.1

DELTA_DEUTEROMETHYLATED = 17.03448
DELTA_DEUTEROACYLATED = 45.02939

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_SETS_ROOT = PROJECT_ROOT / "data" / "test_sets"
TEST_SETS = sorted([p for p in TEST_SETS_ROOT.glob("set_*") if p.is_dir()])


def _build_find_series_log(
    set_name: str,
    deriv_name: str,
    total_signals: int,
    src_count: int,
    result_count: int,
    matched_count: int,
) -> str:
    coverage = matched_count / total_signals if total_signals else 0.0
    return (
        f"{set_name:<8} | "
        f"{deriv_name:<18} | "
        f"signals={total_signals:>3} | "
        f"src={src_count:>3} | "
        f"series={result_count:>3} | "
        f"matched={matched_count:>3} | "
        f"coverage={coverage:>6.1%}"
    )


def _print_find_series_log(
    set_name: str,
    deriv_name: str,
    total_signals: int,
    src_count: int,
    result_count: int,
    matched_count: int,
) -> None:
    print(
        "\n"
        + _build_find_series_log(
            set_name=set_name,
            deriv_name=deriv_name,
            total_signals=total_signals,
            src_count=src_count,
            result_count=result_count,
            matched_count=matched_count,
        )
    )


def _ppm_error(observed: float, theoretical: float) -> float:
    return abs(observed - theoretical) / theoretical * 1e6


def _prepare_assigned_original(set_dir: Path, ppm_tol: float) -> Spectrum:
    src = load_spectrum(set_dir / "original.csv", mass_min=100, mass_max=1000)
    ann = pd.read_csv(set_dir / "annotations.csv")

    ann_orig_signal = ann[
        (ann["spectrum_type"] == "original")
        & (ann["is_signal"] == True)
    ].copy()

    table = src.table.copy()

    if "brutto" not in table.columns:
        table["brutto"] = None
    if "assign" not in table.columns:
        table["assign"] = False

    ann_masses = ann_orig_signal["mass_obs"].astype(float).tolist()
    ann_formulas = ann_orig_signal["formula"].tolist()

    matched_rows = set()

    for i, mz in enumerate(table["mass"].astype(float)):
        best_j = None
        best_ppm = None

        for j, ann_mz in enumerate(ann_masses):
            err = _ppm_error(mz, ann_mz)
            if err <= ppm_tol and (best_ppm is None or err < best_ppm):
                best_j = j
                best_ppm = err

        if best_j is not None:
            formula = ann_formulas[best_j]
            if pd.notna(formula) and str(formula).strip():
                table.at[i, "brutto"] = str(formula).strip()
                table.at[i, "assign"] = True
                matched_rows.add(i)

    table = table.loc[sorted(matched_rows)].reset_index(drop=True).copy()

    filtered = src.copy()
    filtered.table = table
    return filtered


def _filter_to_signal_original_peaks(
    src: Spectrum,
    annotations: pd.DataFrame,
    ppm_tol: float,
) -> Spectrum:
    ann_orig_signal = annotations[
        (annotations["spectrum_type"] == "original")
        & (annotations["is_signal"] == True)
    ].copy()

    signal_masses = ann_orig_signal["mass_obs"].astype(float).tolist()

    table = src.table.copy()
    keep_mask = [
        any(_ppm_error(float(mz), signal_mz) <= ppm_tol for signal_mz in signal_masses)
        for mz in table["mass"]
    ]

    filtered_table = table.loc[keep_mask].reset_index(drop=True)
    return Spectrum(filtered_table)


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


def _load_molecules_map(set_dir: Path) -> pd.DataFrame:
    molecules = pd.read_csv(set_dir / "molecules.csv")

    required_cols = {"compound_number", "carboxyl_count", "hydroxyl_count"}
    missing = required_cols - set(molecules.columns)
    assert not missing, (
        f"{set_dir.name}/molecules.csv: отсутствуют колонки {sorted(missing)}"
    )

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


@pytest.mark.parametrize("set_dir", TEST_SETS, ids=lambda p: p.name)
@pytest.mark.parametrize(
    "deriv_filename,delta",
    [
        ("deutermethylated.csv", DELTA_DEUTEROMETHYLATED),
        ("deuteroacylated.csv", DELTA_DEUTEROACYLATED),
    ],
    ids=["deutermethylated", "deuteroacylated"],
)
def test_find_series_on_existing_sets(
    set_dir: Path,
    deriv_filename: str,
    delta: float,
):
    assert TEST_SETS_ROOT.exists(), f"Не найдена папка test_sets: {TEST_SETS_ROOT}"

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
        (ann["spectrum_type"] == "original")
        & (ann["is_signal"] == True)
    ].copy()

    assert not ann_orig_signal.empty, (
        f"{set_dir.name}: в annotations.csv нет signal-пиков original"
    )

    src = _prepare_assigned_original(set_dir, ppm_tol=ASSIGN_MATCH_PPM)
    src = _filter_to_signal_original_peaks(src, ann, ppm_tol=MATCH_PPM)

    assert not src.table.empty, (
        f"{set_dir.name}: после фильтрации не осталось signal-пиков original"
    )

    deriv = load_spectrum(set_dir / deriv_filename, mass_min=100, mass_max=2000)

    result = find_series(
        src=src,
        deriv=deriv,
        delta=delta,
        ppm_tol=REL_ERROR_PPM,
        max_groups=20,
        allow_gaps=True,
        min_series_length=1,
    )

    assert not result.empty, (
        f"{set_dir.name}/{deriv_filename}: find_series не нашел ни одной серии"
    )

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

    matched_signals = 0
    wrong_length_cases = []
    missing_series_cases = []

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

        matched_signals += 1

        assert isinstance(row["steps_found"], list), (
            f"{set_dir.name}/{deriv_filename}: steps_found должен быть list"
        )
        assert isinstance(row["missing"], list), (
            f"{set_dir.name}/{deriv_filename}: missing должен быть list"
        )
        assert isinstance(row["series_mz"], list), (
            f"{set_dir.name}/{deriv_filename}: series_mz должен быть list"
        )

        if row["n_groups"] != expected_len:
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

    total_signals = len(ann_orig_signal)
    src_count = len(src.table)
    result_count = len(result)

    wrong_count = len(wrong_length_cases) + len(missing_series_cases)
    wrong_ratio = wrong_count / total_signals if total_signals else 0.0

    examples = []

    for case in missing_series_cases[:3]:
        examples.append(
            f"NO_SERIES mass={case['mass_obs']:.6f}, "
            f"compound_number={case['compound_number']}, "
            f"expected_len={case['expected_len']}"
        )

    for case in wrong_length_cases[:3]:
        examples.append(
            f"BAD_LEN mass={case['mass_obs']:.6f}, "
            f"compound_number={case['compound_number']}, "
            f"expected={case['expected_len']}, actual={case['actual_len']}, "
            f"steps_found={case['steps_found']}, missing={case['missing']}"
        )

    MAX_PROBLEM_RATIO = 0.07
    assert wrong_ratio <= MAX_PROBLEM_RATIO, (
        f"{set_dir.name}/{deriv_filename}: слишком много пиков с неправильной длиной серии "
        f"или без серии: {wrong_count}/{total_signals} ({wrong_ratio:.1%}). "
        f"Примеры: {' | '.join(examples) if examples else 'нет'}"
    )

    _print_find_series_log(
        set_name=set_dir.name,
        deriv_name=deriv_filename.removesuffix(".csv"),
        total_signals=total_signals,
        src_count=src_count,
        result_count=result_count,
        matched_count=matched_signals,
    )

