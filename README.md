# NOM-HRMS Functional-Group Analyzer

Инструмент для расшифровки масс-спектров природного органического вещества (NOM)  
по данным HPLC-HRMS с дериватизацией дейтерометилированием и дейтероацилированием.

Определяет брутто-формулы, строит диаграммы Ван-Кревелена, находит гомологические  
серии и подсчитывает количество функциональных групп (-COOH, -OH) на молекулу.

---

## 🚀 Быстрый старт (Windows)

> 💡 **Ни Python, ни Git, ни командная строка не нужны.**

1.  **Скачайте `NOM_HRMS_FGA.exe`** со страницы [Releases](https://github.com/MikhaylenkoVS/NOM-HRMS-FGA/releases)

2.  **Запустите** — двойным кликом по `NOM_HRMS_FGA.exe`

3.  Приложение откроется — можно сразу загружать спектры и анализировать.

`NOM_HRMS_FGA.exe` — это полностью автономный файл (~120 МБ), который содержит  
в себе Python, все библиотеки и графический интерфейс. Ничего дополнительно  
устанавливать не требуется.

### ⚠️ Windows SmartScreen

При первом запуске Windows может показать предупреждение  
«**Windows защитил ваш компьютер**» — это нормально для любого нового  
приложения без цифровой подписи:

1. Нажмите **«Подробнее»**
2. Нажмите **«Выполнить в любом случае»**

Приложение безопасно — исходный код открыт (GPL-3.0) и доступен в этом репозитории.

---

## 💻 Установка для разработчиков (Windows / macOS / Linux)

Если вам нужен доступ к исходному коду, API или вы работаете не на Windows.

### Windows

1.  **Установите Python** (≥ 3.10):
    - Скачайте установщик с [python.org/downloads](https://python.org/downloads)
    - **Важно:** при установке отметьте галочку **«Add Python to PATH»**
    - Проверьте: откройте `Командную строку` → `python --version`

2.  **Установите программу** одной командой:
    ```cmd
    pip install git+https://github.com/MikhaylenkoVS/NOM-HRMS-FGA.git
    ```

3.  **Запустите:**
    ```cmd
    nom-hrms-fga
    ```

### macOS

1.  **Установите Python** (≥ 3.10):
    - Скачайте установщик с [python.org/downloads](https://python.org/downloads)
    - Или через Homebrew: `brew install python@3.12`
    - Проверьте: `python3 --version`

2.  **Установите программу:**
    ```bash
    pip3 install git+https://github.com/MikhaylenkoVS/NOM-HRMS-FGA.git
    ```

3.  **Запустите:**
    ```bash
    nom-hrms-fga
    ```
    Если команда не найдена: `python3 -m src`

### Linux (Ubuntu / Debian)

1.  **Установите Python и pip:**
    ```bash
    sudo apt update
    sudo apt install python3 python3-pip python3-tk
    ```

2.  **Установите программу:**
    ```bash
    pip3 install git+https://github.com/MikhaylenkoVS/NOM-HRMS-FGA.git
    ```

3.  **Запустите:**
    ```bash
    nom-hrms-fga
    ```
    Если команда не найдена: `python3 -m src`

### Только CLI (без GUI)

```bash
git clone https://github.com/MikhaylenkoVS/NOM-HRMS-FGA.git
cd NOM-HRMS-FGA
pip install -r requirements.txt
python -m src.core.pipeline --help
```

### Опционально: поддержка ThermoRAW

Только на Windows, требуется [MSFileReader 3.1 SP4](https://thermo.flexnetoperations.com/control/thmo/search?query=MSFileReader):
```bash
pip install ".[raw]"
```

<br>

> 💡 **Проблемы с запуском?** Самая частая причина — Python не добавлен в PATH при установке. Переустановите Python с галочкой «Add Python to PATH».

---

## 📖 Типичные сценарии использования

### Сценарий 1 — GUI-анализ de novo (рекомендуемый)

1. Запустите `NOM_HRMS_FGA.exe` (Windows) или `nom-hrms-fga` (macOS/Linux)
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
    src_path="original.csv",
    dmet_path="dmet.csv",
    dacet_path="dacet.csv",
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
git clone https://github.com/MikhaylenkoVS/NOM-HRMS-FGA.git
cd NOM-HRMS-FGA
pip install -e ".[dev]"
pre-commit install

# Запуск тестов
pytest tests/ -q            # все тесты (~2 мин)
pytest tests/ -q -m unit    # только unit-тесты
pytest tests/ -q -m smoke   # только быстрые smoke-тесты
```

### Сборка `.exe`

```bash
python tools/build_exe.py          # сборка NOM_HRMS_FGA.exe (~120 MB)
python tools/build_exe.py --clean  # очистка + сборка
python tools/build_exe.py --test   # сборка + smoke-тест
```

Сборка использует [PyInstaller](https://pyinstaller.org) и конфигурацию из `tools/NOM_HRMS_FGA.spec`.
Автоматическая сборка при создании релиза — `.github/workflows/release_exe.yml`.

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
├── tests/                  # PyTest-тесты (105 шт.)
├── docs/                   # Документация, архитектура, планы
├── external/               # Стороннее ПО (GPL-3.0)
├── .github/workflows/      # CI/CD (автосборка .exe при релизе)
├── assets/                 # Иконка приложения
├── tools/                  # Сборка .exe (PyInstaller)
│   ├── build_exe.py        # Сценарий сборки
│   ├── launcher.py         # Точка входа (crash-safe)
│   └── NOM_HRMS_FGA.spec   # Конфигурация PyInstaller
├── pyproject.toml          # Метаданные пакета, зависимости
└── requirements.txt        # Зависимости (для разработки)
```

---

## 📄 Лицензия

Основной код — **GPL-3.0**. Сторонний код в `external/` — **GPL-3.0** (см. `external/README.txt`).

## 🔗 Ссылки

- [Репозиторий](https://github.com/MikhaylenkoVS/NOM-HRMS-FGA)
- [Исходная NOMspectra](https://github.com/kozelkov-ie/NOMspectra)
- [MSFileReader (ThermoRAW)](https://github.com/frallain/pymsfilereader)
