# Архитектура NOM-HRMS-FGA (v0.4.2)

## Общая схема

```
Пользователь
    │
    ├─ NOM_HRMS_FGA.exe (Windows, двойной клик)
    │      └─ tools/launcher.py → crash-safe запуск → src.app.main()
    │
    └─ nom-hrms-fga (CLI / pip install)
           └─ src/__main__.py → src.app.main()

                    ┌──────────┴──────────┐
                    │     src/app.py       │
                    │   GUI (tkinter)      │
                    │   Вкладки:           │
                    │   - Параметры        │
                    │   - Спектры          │
                    │   - Серии            │
                    │   - Результаты       │
                    │   - Van Krevelen     │
                    │   - Структуры        │
                    │   - Лог              │
                    └──────────┬──────────┘
                               │ run_pipeline()
                               ▼
               ┌───────────────────────────────┐
               │     src/core/pipeline.py      │
               │                               │
               │  Шаг 1: load_spectrum (×3)    │
               │  Шаг 2a: denoise (×3)         │
               │  Шаг 2b: assign_formulas      │
               │  Шаг 3: find_series (dmet)     │
               │  Шаг 4: find_series (dacet)    │
               │  Шаг 5: build_result_table    │
               │  Шаг 6: visualize (опционально)│
               │  Шаг 7: create_van_krevelen_plot │
               └───────────────┬───────────────┘
                               │
          ┌────────────────────┼────────────────────┐
          │                    │                    │
          ▼                    ▼                    ▼
   spectrum_ops.py       van_krevelen.py      molecule.py
   - load_spectrum       - NOM_REGIONS        - Atom
   - assign_formulas     - create_plot        - Molecule
   - find_series         - compute_data       - parse_formula
   - build_result_table                       - calculate_IHD
   - DELTA_CD3/CD3CO
          │
          ├── fragments.py (FRAGMENT_LIBRARY)
          ├── fragment_combinations.py
          ├── atoms.py (ELEMENT_DATA)
          └── rdkit_bridge.py (RDKit ↔ Molecule)
```

## Модульная структура

### `src/core/` — Вычислительное ядро

| Модуль | Назначение |
|--------|------------|
| `pipeline.py` | Главный конвейер: run_pipeline(), тест-режим, статистика |
| `spectrum_ops.py` | Загрузка, денойз, назначение формул, поиск серий, сборка таблицы |
| `van_krevelen.py` | Диаграмма Ван-Кревелена, классификация NOM-областей |
| `molecule.py` | Граф молекулы (Atom, Molecule), парсинг формул, расчёт IHD |
| `atoms.py` | Данные элементов, гибридизация атомов |
| `fragments.py` | Библиотека фрагментов и функциональных групп |
| `fragment_combinations.py` | Комбинаторный поиск молекулярных структур |
| `rdkit_bridge.py` | Конвертация Molecule → RDKit, визуализация |
| `raw_bridge.py` | Усреднение ThermoRAW → CSV (Windows, опционально) |
| `_safety.py` | Утилиты: safe(), _safe_df() для Optional типов |

### `src/configs/` — Единый источник конфигурации

| Файл | Содержание |
|------|------------|
| `chemistry.json` | Моноизотопные массы, сдвиги дериватизации, массы протона/электрона |
| `pipeline.json` | Параметры по умолчанию, диапазоны элементов, пороги тестов |
| `paths.json` | Пути к данным, имена файлов спектров |
| `loader.py` | ConfigNamespace — загрузка JSON с кешированием и атрибутным доступом |

### `src/ui/` — Графический интерфейс

| Модуль | Назначение |
|--------|------------|
| `theme.py` | Цветовая схема Catppuccin, стили ttk, стиль matplotlib |
| `plots.py` | Встраивание matplotlib-фигур в tkinter (embed_figure) |

### `src/structures/` — Визуализация молекул

| Модуль | Назначение |
|--------|------------|
| `tab.py` | StructureViewerTab — вкладка «Структуры» в GUI |
| `widgets.py` | StructureCard — карточка молекулы с RDKit-отрисовкой |
| `rdkit_utils.py` | fragment_to_rdkit, save_mol, save_png |

### `src/testing/` — Smoke-тестирование

| Модуль | Назначение |
|--------|------------|
| `smoke_runner.py` | Прогон пайплайна на всех тест-сетах, экспорт артефактов |
| `artifact_export.py` | Построение графиков: спектры, серии, гистограммы |
| `report_models.py` | Датаклассы: SetSmokeResult, SmokeSuiteResult |
| `structure_export.py` | Экспорт структур для соединений |

### `src/simulations/` — Генерация тест-сетов

| Модуль | Назначение |
|--------|------------|
| `generate_test_sets.py` | Создание синтетических спектров с изотопными метками и шумом |
| `search_pubchem_nom_like_cids.py` | Поиск NOM-подобных соединений в PubChem |
| `process_pubchem_candidates.py` | Фильтрация кандидатов PubChem |

## Система сборки `.exe`

```
tools/build_exe.py              # Сценарий: установка deps → PyInstaller → smoke-test
    │
    ├─ tools/NOM_HRMS_FGA.spec  # PyInstaller: onefile, windowed, ~120 MB
    │     ├─ tools/launcher.py  # Точка входа с crash-safe обработкой
    │     ├─ collect_all: rdkit, matplotlib, PIL, nomspectra, scipy
    │     ├─ datas: src/configs/*.json
    │     └─ excludes: PyQt5, PyQt6, jupyter, pytest, IPython, ...
    │
    └─ .github/workflows/release_exe.yml   # CI/CD: автосборка при релизе
```

## Тестирование

```
tests/
├── conftest.py              # Фикстуры: project_root, test_sets_root
├── unit/
│   ├── test_core_utils.py   # _ppm_error, _normalize_brutto, parse_formula, ...
│   ├── test_assign_formulas.py     # assign_formulas по всем test_sets
│   ├── test_denoise.py             # Шумоподавление: recall сигналов и шума
│   ├── test_find_series.py         # Поиск серий + max_consecutive_misses
│   ├── test_van_krevelen.py        # Диаграмма Ван-Кревелена
│   ├── test_nom_prioritize.py      # NOM-приоритизация формул
│   ├── test_app_fallback.py        # Консистентность сигнатур embed_figure
│   ├── test_annotations_consistency.py  # Валидность аннотаций
│   ├── test_chemical_validity.py   # Химическая валидность данных
│   └── test_structural_validity.py # Структура CSV-файлов
└── integration/
    ├── test_pipeline_integration.py # Полный конвейер: denoise→assign→series
    └── test_app_smoke.py            # Сквозной smoke-тест через smoke_runner
```

Всего: **105 тестов** (129 с учётом параметризованных вариантов). Запуск: `pytest tests/ -q` (~2 мин).

## Поток данных

```
CSV-файлы (.csv) или RAW-файлы (.raw)
        │
        ▼
load_spectrum() → Spectrum (nomspectra)
   ┌────┼────┐
   ▼    ▼    ▼
  src  dmet  dacet   (исходный, дейтерометилированный, дейтероацилированный)
   │    │    │
   ▼    ▼    ▼
denoise() → отфильтрованные Spectrum
   │
   ▼
assign_formulas() → Spectrum с колонками brutto, assign, all_candidates
   │
   ├──────────────────┐
   ▼                  ▼
find_series(dmet)    find_series(dacet)
  DELTA_CD3           DELTA_CD3CO
   │                  │
   └────────┬─────────┘
            ▼
build_result_table() → DataFrame: mass, intensity, brutto, all_candidates, N_COOH, N_OH, missing_*
            │
            ├── create_van_krevelen_plot()
            └── экспорт CSV
```
