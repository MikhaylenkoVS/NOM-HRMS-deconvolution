# ============================================================
# tests/test_app_smoke.py
# ============================================================
import pytest
import logging
from pathlib import Path
import sys

# Добавляем корень проекта в путь, если запускаем не из корня
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Настройка логгера
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

from src.testing.smoke_runner import run_smoke_suite

@pytest.fixture(scope="session")
def smoke_output_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("smoke_results")

@pytest.fixture(scope="session")
def data_dir():
    return Path(__file__).resolve().parents[2] / "data" / "test_sets"

def test_smoke_all_sets(data_dir, smoke_output_dir):
    """Smoke test: прогоняет все тестовые наборы и проверяет, что все успешны."""
    suite = run_smoke_suite(data_dir, smoke_output_dir)

    # Проверки
    assert len(suite.sets) == 5, "Ожидалось 5 тестовых наборов"
    for s in suite.sets:
        assert s.pipeline_success, f"{s.set_name}: пайплайн завершился с ошибкой: {s.error}"
        assert s.success, f"{s.set_name}: общий провал: {s.error}"
        assert s.result_table_path is not None, f"{s.set_name}: result_table.csv не создан"
        assert s.result_table_path.exists(), f"{s.set_name}: result_table.csv отсутствует"
        # Не менее 1 соединения должно быть в таблице (в тестовых данных точно есть)
        import pandas as pd
        df = pd.read_csv(s.result_table_path, sep=';')
        assert len(df) > 0, f"{s.set_name}: result_table пуста"

    # Проверка общего успеха
    assert suite.overall_success, "Не все наборы прошли успешно, см. логи"