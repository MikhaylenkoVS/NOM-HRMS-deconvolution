# Генерация синтетических тест‑спектров (`src/simulations`)

Папка `src/simulations` содержит скрипты и функции для генерации синтетических
тестовых наборов масс‑спектров, используемых в `data/test_sets`.

Исходник — адаптированная копия подпроекта `generate_test_spectra/tools` с
минимально необходимыми доработками под текущий проект.

## Назначение

Основные задачи:

- прочитать конфигурации наборов (`data/test_sets/set_XX/config.json`);
- прочитать список молекул (`data/test_sets/set_XX/molecules.csv`);
- по формуле рассчитать теоретические массы;
- сгенерировать теоретические спектры (original, deuteromethyl, deuteroacyl);
- применить точные масс‑сдвиги для дериватизации (из `config.json`);
- добавить реалистичную ppm‑ошибку и шумовые пики;
- записать итоговые файлы:
  - `original.csv`,
  - `deutermethylated.csv`,
  - `deuteroacylated.csv`,
  - `annotations.csv`.

Таким образом `src/simulations` — генератор эталонных тестовых данных.

## Основные компоненты

Ключевой файл (название может отличаться, см. фактическую структуру):

- `generate_test_sets.py`:
  - `generate_all_test_sets(overwrite: bool)` — генерация всех наборов `set_01`, `set_02`, …;
  - `generate_single_test_set(set_id: str, overwrite: bool)` — генерация одного набора;
  - вспомогательные функции:
    - загрузка и создание `config.json`;
    - загрузка `molecules.csv` и расчёт масс по формуле;
    - генерация теоретических масс для разных степеней дериватизации;
    - добавление ppm‑ошибки (`apply_mass_error`) и шумовых пиков;
    - запись `*.csv` и `annotations.csv`.

## Как запустить генерацию

### Предварительные требования

- Активированное virtualenv проекта (если есть):

  ```bash
  source .venv/bin/activate     # Linux/macOS
  .venv\Scripts\activate        # Windows
  ```

- Установлены зависимости проекта (см. корневой `requirements.txt` или `pyproject.toml`).

### Запуск генерации всех наборов

Из корня проекта:

```bash
python src/simulations/generate_test_sets.py
```

(Путь может отличаться, если файл лежит глубже; в исходном проекте это
было `generate_test_spectra/tools/generate_test_sets.py` — здесь путь
адаптирован под `src/simulations`.)

В конце файла должен быть блок:

```python
def main() -> None:
    generate_all_test_sets(overwrite=True)


if __name__ == "__main__":
    main()
```

После запуска:

- для каждого `set_XX` в `data/test_sets` будут:
  - обновлены `original.csv`, `deutermethylated.csv`, `deuteroacylated.csv`, `annotations.csv`;
  - использованы текущие параметры из `set_XX/config.json`;
- структура и согласованность файлов можно проверить через `pytest tests/unit`.

### Запуск генерации одного набора

Если нужно пересоздать только один набор (например, после изменения
`set_01/config.json`):

1. Временный код (пример):

   ```python
   def main() -> None:
       generate_single_test_set("set_01", overwrite=True)
   ```

2. Запустить:

   ```bash
   python src/simulations/generate_test_sets.py
   ```

3. После пересоздания вернуть `main()` к варианту с `generate_all_test_sets`, чтобы случайно не забыть.

Альтернатива — сделать отдельный CLI (например, через `argparse`), но в текущей версии используется простой вызов в `main()`.

## Настройка параметров генерации

Все параметры задаются в `config.json` внутри каждого `set_XX`. Основные блоки:

- `mass_range` — минимальная и максимальная масса для генерации пиков;
- `ppm_error` — модель инструментальной ошибки:
  - `type` — тип распределения (`"normal"` и т.п.);
  - `mean` — среднее значение ошибки (ppm);
  - `std` — стандартное отклонение (ppm);
  - `max_abs` — максимальная величина ошибки по модулю (ppm).

- `noise` — параметры шума:
  - `peak_count` — количество шумовых пиков;
  - `intensity_fraction_max` — максимальная доля от максимальной интенсивности сигнала (шум не должен быть выше сигнальных пиков).

- `derivatization`:
  - `deutermethyl`:
    - `target_groups` — список целевых групп (например, `["COOH"]`);
    - `mass_shift_per_group` — точный масс‑сдвиг (Da) на одну группу;
    - `label` — метка (`"CD3"`);
    - `conversion_yield` — доля срабатывания дериватизации.
  - `deuteroacyl` — аналогично, но для другой группы (например, `["OH"]`) и другого mass shift.

### Важные замечания

- **Точные масс‑сдвиги**: значения `mass_shift_per_group` вычисляются один раз по химической формуле (например, через функцию `exact_mass_from_formula`) и вручную вносятся в `config.json`. Скрипты генерации больше не считают их «на лету» и используют только то, что указано в конфиге.
- **Единственное применение ppm‑ошибки**: ошибку (`apply_mass_error`) нужно применять один раз при формировании словарей `spectra`. Далее:
  - `mass` в `original/deutermethylated/deuteroacylated.csv` = `mass_obs`;
  - `annotations.csv` читает те же значения (без повторного вызова `apply_mass_error`), благодаря чему unit‑тесты корректно находят соответствие `mass_obs ↔ mass`.

## Проверка после генерации

После любого изменения `src/simulations` или `config.json` рекомендуется:

```bash
python src/simulations/generate_test_sets.py
pytest tests/unit
```

Если все тесты проходят:

- структура файлов в `data/test_sets` корректна;
- `annotations.csv` согласован с тремя спектрами и `config.json`;
- данные готовы для тестирования алгоритма дешифровки (`core.py`).

Это обеспечивает воспроизводимость и устойчивость всего конвейера генерации тестовых данных.