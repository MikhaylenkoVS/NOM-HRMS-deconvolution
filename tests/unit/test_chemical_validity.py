import csv
from pathlib import Path

from src.configs import PATHS
from tests.conftest import PROJECT_ROOT, TEST_SETS_ROOT

DATA_ROOT = PROJECT_ROOT / PATHS.data_dir


def _get_set_dir(set_id: str) -> Path:
    return TEST_SETS_ROOT / set_id


def test_carboxyl_and_hydroxyl_counts_in_range_if_present():
    """Если в molecules.csv есть строки, carboxyl_count и hydroxyl_count должны быть в [0, 10].

    Ноль допускается: после исправления hydroxyl_count = hydroxyl_count - carboxyl_count
    у многих молекул свободные OH-группы отсутствуют.
    """

    set_dir = _get_set_dir("set_01")
    path = set_dir / PATHS.spectrum_files["molecules"]
    assert path.exists(), "molecules.csv должен существовать"

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    for row in rows:
        carboxyl_str = row.get("carboxyl_count")
        hydroxyl_str = row.get("hydroxyl_count")

        if carboxyl_str and carboxyl_str.strip():
            carboxyl_value = int(carboxyl_str)
            assert (
                0 <= carboxyl_value <= 10
            ), f"carboxyl_count вне диапазона [0, 10]: {carboxyl_value}"

        if hydroxyl_str and hydroxyl_str.strip():
            hydroxyl_value = int(hydroxyl_str)
            assert (
                0 <= hydroxyl_value <= 10
            ), f"hydroxyl_count вне диапазона [0, 10]: {hydroxyl_value}"
