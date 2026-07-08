# NOM-HRMS Functional-Group Analyzer

Инструмент для расшифровки масс-спектров природного органического вещества (NOM)  
по данным HPLC-HRMS с дериватизацией дейтерометилированием и дейтероацилированием.

Определяет брутто-формулы, строит диаграммы Ван-Кревелена, находит гомологические  
серии и подсчитывает количество функциональных групп (-COOH, -OH) на молекулу.

---

## 📦 Установка (для конечного пользователя)

```bash
# Требуется Python ≥ 3.10

# Способ 1 — установка одной командой из GitHub
pip install git+https://github.com/MikhaylenkoVS/NOM-HRMS-deconvolution.git

# Способ 2 — из локальной копии репозитория
git clone https://github.com/MikhaylenkoVS/NOM-HRMS-deconvolution.git
cd NOM-HRMS-deconvolution
pip install .
```

### Опциональные компоненты

```bash
# Поддержка ThermoRAW-файлов (только Windows, требуется MSFileReader 3.1 SP4)
pip install ".[raw]"

# Инструменты разработчика
pip install ".[dev]"
```

## 🚀 Запуск

```bash
nom-analyzer          # графический интерфейс (рекомендуется)
python -m src         # альтернативный запуск GUI
```

**Системные требования:** `pip install ".[dev]"` + `pre-commit install`.

---

## 📖 Типичные сценарии использования

### Сценарий 1 — GUI-анализ de novo (рекомендуемый)

1. Запустите `nom-analyzer`
2. На вкладке **Параметры** укажите три CSV-файла:
   - **Исходный спектр** — недериватизированный образец
   - **Дейтерометилирование** — образец после CD₃-метилирования
   - **Дейтероацилирование** — образец после CD₃CO-ацилирования
3. При необходимости загрузите ThermoRAW-файлы (`.raw`) — автоматически усреднятся в CSV
4. Нажмите **▶ Запустить анализ**
5. Перейдите на вкладки **Спектры**, **Ван-Кревелен**, **Серии**, **Результаты** для визуализации

### Сценарий 2 — CSV-файлы через GUI

Подготовьте CSV со столбцами `m/z` (или `mass`) и `intensity`. Разделитель — запятая или табуляция.

Пример:
```csv
mass,intensity
100.0518,245030.0
101.0597,123410.5
...
```

Затем выполните сценарий 1.

### Сценарий 3 — Python API

```python
from src.core.spectrum_ops import load_spectrum, assign_formulas, denoise
from src.core.pipeline import run_pipeline

# Загрузить, денойзить, назначить формулы
spectrum = load_spectrum("sample.csv")
clean = denoise(spectrum)
assigned = assign_formulas(clean, mode="simple", rel_error_ppm=1.0)

# Или запустить полный конвейер
run_pipeline(
    original_csv="original.csv",
    dmet_csv="dmet.csv",
    dacet_csv="dacet.csv",
)
```

### Сценарий 4 — ThermoRAW → CSV (Windows)

```python
from src.core.raw_bridge import average_raw_to_csv
csv_path = average_raw_to_csv("sample.raw", rt_min=5.0, rt_max=25.0)
```

### Сценарий 5 — только CLI (без GUI)

```bash
python -m src.core.pipeline --help
```

---

## 🧪 Разработка

```bash
git clone https://github.com/MikhaylenkoVS/NOM-HRMS-deconvolution.git
cd NOM-HRMS-deconvolution
pip install -e ".[dev]"
pre-commit install

# Запуск тестов
pytest tests/ -q            # все тесты (~2 мин)
pytest tests/ -q -m smoke   # только быстрые smoke-тесты
```

---

## 📂 Структура проекта

```
├── src/                    # Исходный код пакета
│   ├── core/               # Ядро: pipeline, spectrum_ops, fragments, molecule
│   ├── configs/            # Конфигурации (pipeline.json, chemistry.json)
│   ├── simulations/        # Генерация синтетических тест-сетов
│   ├── structures/         # Структурные формулы (RDKit), GUI-виджеты
│   ├── testing/            # Smoke-тесты, экспорт артефактов
│   ├── ui/                 # Тема, графики, макет GUI
│   └── app.py              # Главное приложение (GUI)
├── data/
│   ├── ref_data/           # Эталонные данные
│   └── test_sets/          # Синтетические тест-наборы (set_01..set_05)
├── tests/                  # PyTest-тесты
├── docs/                   # Документация, архитектура, планы
├── external/               # Стороннее ПО (GPL-3.0)
├── pyproject.toml          # Метаданные пакета, зависимости
└── requirements.txt        # Зависимости (для разработки)
```

---

## 📄 Лицензия

Основной код — **MIT**. Сторонний код в `external/` — **GPL-3.0** (см. `external/README.txt`).

## 🔗 Ссылки

- [Репозиторий](https://github.com/MikhaylenkoVS/NOM-HRMS-deconvolution)
- [Исходная NOMspectra](https://github.com/kozelkov-ie/NOMspectra)
- [MSFileReader (ThermoRAW)](https://github.com/frallain/pymsfilereader)
