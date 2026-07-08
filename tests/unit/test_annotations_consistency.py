import csv
from math import isclose
from pathlib import Path

from src.configs import CHEM, PATHS
from tests.conftest import PROJECT_ROOT, TEST_SETS_ROOT

DATA_ROOT = PROJECT_ROOT / PATHS.data_dir


def _get_set_dir(set_id: str) -> Path:
    return TEST_SETS_ROOT / set_id


def _load_spectrum_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _to_float(value: str | None) -> float:
    return float(value) if value not in (None, "") else 0.0


def test_annotations_have_matching_peaks_in_spectra_and_ppm_limits():
    """Для каждой записи в annotations.csv должен существовать пик
    в соответствующем CSV-спектре, а mass_error_ppm должен быть согласован
    с mass_obs и mass_theor и не выходить за пределы config['ppm_error']['max_abs'].
    """

    set_id = "set_01"
    set_dir = _get_set_dir(set_id)

    # Порог ppm соответствует default-конфигу генератора (0.5 ppm)
    max_abs_ppm = 0.5

    # Загружаем annotations.csv
    ann_path = set_dir / PATHS.spectrum_files["annotations"]
    assert ann_path.exists(), "annotations.csv должен существовать"

    with ann_path.open("r", encoding="utf-8", newline="") as f:
        ann_reader = csv.DictReader(f)
        annotations = list(ann_reader)

    # Загружаем спектры
    spectra_files = {
        "original": set_dir / PATHS.spectrum_files["original"],
        "deutermethylated": set_dir / PATHS.spectrum_files["deutermethylated"],
        "deuteroacylated": set_dir / PATHS.spectrum_files["deuteroacylated"],
    }

    spectra_data: dict[str, list[dict]] = {}
    for spectrum_type, path in spectra_files.items():
        assert path.exists(), f"{path.name} должен существовать"
        spectra_data[spectrum_type] = _load_spectrum_csv(path)

    # Строим индекс (mass, intensity) для каждого spectrum_type
    spectra_index: dict[str, list[tuple[float, float]]] = {}
    for spectrum_type, rows in spectra_data.items():
        pairs: list[tuple[float, float]] = []
        for row in rows:
            mass = _to_float(row.get("mass"))
            intensity = _to_float(row.get("intensity"))
            pairs.append((mass, intensity))
        spectra_index[spectrum_type] = pairs

    # Основная проверка по аннотациям
    for ann in annotations:
        spectrum_type = ann.get("spectrum_type")
        assert (
            spectrum_type in spectra_index
        ), f"Неизвестный spectrum_type в annotations: {spectrum_type}"

        mass_obs = _to_float(ann.get("mass_obs"))
        mass_theor = _to_float(ann.get("mass_theor"))
        mass_error_ppm = _to_float(ann.get("mass_error_ppm"))
        intensity = _to_float(ann.get("intensity"))

        # 1) Проверяем, что mass_obs соответствует mass в соответствующем спектре
        candidates = spectra_index[spectrum_type]

        found = False
        for mass_spec, intensity_spec in candidates:
            if isclose(mass_spec, mass_obs, rel_tol=0, abs_tol=1e-5) and isclose(
                intensity_spec, intensity, rel_tol=1e-6, abs_tol=0.1
            ):
                found = True
                break

        assert found, (
            f"Для аннотации (spectrum_type={spectrum_type}, mass_obs={mass_obs}, "
            f"intensity={intensity}) не найден соответствующий пик "
            f"в {spectrum_type}.csv"
        )

        # 2) Проверяем согласованность mass_error_ppm с mass_obs и mass_theor
        #    mass_error_ppm ~= (mass_obs - mass_theor) / mass_theor * 1e6
        if mass_theor != 0.0:
            calculated_ppm = (mass_obs - mass_theor) / mass_theor * 1e6
            # допускаем небольшой численный разброс
            assert isclose(
                calculated_ppm,
                mass_error_ppm,
                rel_tol=0,
                abs_tol=0.01,
            ), (
                f"mass_error_ppм не согласован с mass_obs и mass_theor: "
                f"mass_obs={mass_obs}, mass_theor={mass_theor}, "
                f"mass_error_ppm={mass_error_ppm}, "
                f"calculated={calculated_ppm}"
            )

        # 3) Проверяем, что |mass_error_ppm| <= max_abs_ppm из config.json
        assert abs(mass_error_ppm) <= max_abs_ppm + 1e-6, (
            f"mass_error_ppm={mass_error_ppm} выходит за пределы "
            f"max_abs_ppm={max_abs_ppm}"
        )
