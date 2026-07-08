import csv
from pathlib import Path

from src.configs import PATHS
from tests.conftest import PROJECT_ROOT, TEST_SETS_ROOT

# Тесты структурной валидности файлов тестовых наборов

DATA_ROOT = PROJECT_ROOT / PATHS.data_dir


def _get_set_dir(set_id: str) -> Path:
    return TEST_SETS_ROOT / set_id


def test_required_files_exist_in_set_01():
    """Проверка наличия базовых файлов в set_01."""

    set_dir = _get_set_dir("set_01")
    assert set_dir.exists(), "Директория set_01 должна существовать"

    expected_files = {
        PATHS.spectrum_files["molecules"],
        PATHS.spectrum_files["original"],
        PATHS.spectrum_files["deutermethylated"],
        PATHS.spectrum_files["deuteroacylated"],
        PATHS.spectrum_files["annotations"],
    }

    existing = {p.name for p in set_dir.iterdir() if p.is_file()}
    missing = expected_files - existing
    assert not missing, f"Отсутствуют ожидаемые файлы: {missing}"


def test_spectra_csv_headers_use_mass_and_intensity():
    """Проверка, что файлы спектров имеют заголовки mass,intensity."""

    set_dir = _get_set_dir("set_01")
    spectrum_files = [
        PATHS.spectrum_files["original"],
        PATHS.spectrum_files["deutermethylated"],
        PATHS.spectrum_files["deuteroacylated"],
    ]

    for filename in spectrum_files:
        path = set_dir / filename
        assert path.exists(), f"{filename} должен существовать"

        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)
        assert header is not None, f"{filename} пустой или без заголовка"
        assert header == [
            "mass",
            "intensity",
        ], f"В {filename} ожидается заголовок ['mass', 'intensity'], получено {header}"


def test_molecules_csv_header_structure():
    """Проверка заголовка molecules.csv."""

    set_dir = _get_set_dir("set_01")
    path = set_dir / PATHS.spectrum_files["molecules"]
    assert path.exists(), "molecules.csv должен существовать"

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)

    expected_header = [
        "set_id",
        "compound_id",
        "compound_number",
        "name",
        "smiles",
        "inchi",
        "formula",
        "charge",
        "mode",
        "carboxyl_count",
        "hydroxyl_count",
        "other_fg",
        "nom_like_flag",
        "comment",
    ]

    assert header == expected_header, f"Некорректный заголовок molecules.csv: {header}"


def test_annotations_csv_header_structure():
    """Проверка заголовка annotations.csv."""

    set_dir = _get_set_dir("set_01")
    path = set_dir / PATHS.spectrum_files["annotations"]
    assert path.exists(), "annotations.csv должен существовать"

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)

    expected_header = [
        "set_id",
        "spectrum_type",
        "peak_id",
        "mass_obs",
        "intensity",
        "mass_theor",
        "exact_mass",
        "mass_error_ppm",
        "compound_id",
        "compound_number",
        "formula",
        "derivatization_state",
        "adduct_type",
        "charge",
        "assignment_confidence",
        "is_signal",
    ]

    assert (
        header == expected_header
    ), f"Некорректный заголовок annotations.csv: {header}"
