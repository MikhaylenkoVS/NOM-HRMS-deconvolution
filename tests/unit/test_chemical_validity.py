import csv
from pathlib import Path

SUBPROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = SUBPROJECT_ROOT / "data"
TEST_SETS_ROOT = DATA_ROOT / "test_sets"


def _get_set_dir(set_id: str) -> Path:
    return TEST_SETS_ROOT / set_id


def test_carboxyl_and_hydroxyl_counts_in_range_if_present():
    """Если в molecules.csv есть строки, carboxyl_count и hydroxyl_count должны быть в [1, 10].

    На данном этапе допускается, что файл может быть пустым (только заголовок),
    в этом случае тест просто ничего не проверяет по строкам.
    """

    set_dir = _get_set_dir("set_01")
    path = set_dir / "molecules.csv"
    assert path.exists(), "molecules.csv должен существовать"

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    for row in rows:
        # допускаем, что пока поля могут быть пустыми — тогда пропускаем проверку
        carboxyl = row.get("carboxyl_count")
        hydroxyl = row.get("hydroxyl_count")

        if carboxyl:
            carboxyl_value = int(carboxyl)
            assert 1 <= carboxyl_value <= 10, \
                f"carboxyl_count вне диапазона [1, 10]: {carboxyl_value}"

        if hydroxyl:
            hydroxyl_value = int(hydroxyl)
            assert 1 <= hydroxyl_value <= 10, \
                f"hydroxyl_count вне диапазона [1, 10]: {hydroxyl_value}"