import csv
import json
from pathlib import Path

# Тесты структурной валидности файлов тестовых наборов

SUBPROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = SUBPROJECT_ROOT / "data"
TEST_SETS_ROOT = DATA_ROOT / "test_sets"


def _get_set_dir(set_id: str) -> Path:
    return TEST_SETS_ROOT / set_id


def test_required_files_exist_in_set_01():
    """Проверка наличия базовых файлов в set_01."""

    set_dir = _get_set_dir("set_01")
    assert set_dir.exists(), "Директория set_01 должна существовать"

    expected_files = {
        "config.json",
        "molecules.csv",
        "original.csv",
        "deutermethylated.csv",
        "deuteroacylated.csv",
        "annotations.csv",
    }

    existing = {p.name for p in set_dir.iterdir() if p.is_file()}
    missing = expected_files - existing
    assert not missing, f"Отсутствуют ожидаемые файлы: {missing}"


def test_config_json_has_basic_keys():
    """Проверка, что config.json имеет базовые ключи и типы."""

    set_dir = _get_set_dir("set_01")
    config_path = set_dir / "config.json"
    assert config_path.exists(), "config.json должен существовать"

    data = json.loads(config_path.read_text(encoding="utf-8"))

    # обязательные верхнеуровневые ключи
    for key in ["set_id", "mass_range", "ppm_error", "noise", "derivatization"]:
        assert key in data, f"В config.json отсутствует ключ: {key}"

    # базовые проверки вложенных структур
    assert isinstance(data["mass_range"], dict), "mass_range должен быть dict"
    assert "min" in data["mass_range"] and "max" in data["mass_range"], \
        "mass_range должен содержать поля 'min' и 'max'"

    assert isinstance(data["ppm_error"], dict), "ppm_error должен быть dict"
    for key in ["type", "mean", "std", "max_abs"]:
        assert key in data["ppm_error"], f"В ppm_error отсутствует ключ: {key}"

    assert isinstance(data["noise"], dict), "noise должен быть dict"
    for key in ["peak_count", "intensity_fraction_max"]:
        assert key in data["noise"], f"В noise отсутствует ключ: {key}"

    assert isinstance(data["derivatization"], dict), "derivatization должен быть dict"
    for key in ["deutermethyl", "deuteroacyl"]:
        assert key in data["derivatization"], f"В derivatization отсутствует ключ: {key}"


def test_spectra_csv_headers_use_mass_and_intensity():
    """Проверка, что файлы спектров имеют заголовки mass,intensity."""

    set_dir = _get_set_dir("set_01")
    spectrum_files = [
        "original.csv",
        "deutermethylated.csv",
        "deuteroacylated.csv",
    ]

    for filename in spectrum_files:
        path = set_dir / filename
        assert path.exists(), f"{filename} должен существовать"

        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)
        assert header is not None, f"{filename} пустой или без заголовка"
        assert header == ["mass", "intensity"], \
            f"В {filename} ожидается заголовок ['mass', 'intensity'], получено {header}"


def test_molecules_csv_header_structure():
    """Проверка заголовка molecules.csv."""

    set_dir = _get_set_dir("set_01")
    path = set_dir / "molecules.csv"
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

    assert header == expected_header, \
        f"Некорректный заголовок molecules.csv: {header}"


def test_annotations_csv_header_structure():
    """Проверка заголовка annotations.csv."""

    set_dir = _get_set_dir("set_01")
    path = set_dir / "annotations.csv"
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

    assert header == expected_header, \
        f"Некорректный заголовок annotations.csv: {header}"