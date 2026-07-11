# Code Availability / Доступность программного кода

> Bilingual document (English / Русский) prepared for the *Code Availability*
> section of an ACS journal article.
> Двуязычный документ (English / Русский) для раздела *Code Availability*
> статьи в журнале ACS.

---

## English

**Software name:** NOM-HRMS-FGA

**Description:** Open-source Python software for deconvolution of high-resolution
mass spectra (HPLC-HRMS) of natural organic matter (NOM) / humic substances.
The pipeline combines selective chemical derivatization with isotope labelling —
deuteromethylation of carboxyl groups (–COOH → –COOCD₃, mass increment
DELTA_CD3 = 17.03448 Da per group) and deuteroacylation of hydroxyl groups
(–OH → –OCOCD₃, mass increment DELTA_CD3CO = 45.02939 Da per group) — to
count functional groups, assign molecular formulas in negative ion mode
([M–H]⁻), and enumerate candidate molecular structures.

**Repository:** https://github.com/MikhaylenkoVS/NOM-HRMS-FGA

**Version / Tag:** v0.4.0

**Archived version (DOI):** [DOI будет присвоен при архивировании релиза на Zenodo]

**License:** GPL-3.0 (see `LICENSE`). Third-party code under
`external/usrednenie_spectrov_i_hromatogramm/` is distributed under GPL-3.0 and
retains its own license.

**Programming language / Runtime:** Python ≥ 3.10.

**Installation & launch:**

*Windows (end users):*
1. Download `NOM_HRMS_FGA.exe` from [GitHub Releases](https://github.com/MikhaylenkoVS/NOM-HRMS-FGA/releases)
2. Double-click to launch — no Python, Git, or command line required.

*All platforms (developers):*
```bash
# One-command install from GitHub
pip install git+https://github.com/MikhaylenkoVS/NOM-HRMS-FGA.git

# Launch
nom-hrms-fga           # graphical interface
python -m src          # alternative
```

*Build from source:*
```bash
git clone https://github.com/MikhaylenkoVS/NOM-HRMS-FGA.git
cd NOM-HRMS-FGA
pip install -e ".[dev]"
python build_exe.py    # produces dist/NOM_HRMS_FGA.exe (~120 MB)
```

**Dependencies:** listed in `pyproject.toml` (`[project.dependencies]`) and
[`requirements.txt`](requirements.txt).
Core packages: NumPy, pandas, matplotlib, nomspectra, RDKit, Pillow.
The graphical interface additionally requires `tkinter` (a system package on
some Linux distributions, e.g. `apt install python3-tk`).

**Reproduction (minimal example):**

```bash
# 1. Install
pip install git+https://github.com/MikhaylenkoVS/NOM-HRMS-FGA.git

# 2. Run the automated test suite (129 tests)
pytest
```

**Data availability:** synthetic test datasets used for validation are bundled
with the repository under `data/test_sets/` (`set_01` … `set_05`). Each set
contains the original spectrum plus the deuteromethylated and deuteroacylated
spectra together with ground-truth annotations, allowing the full pipeline and
its results to be reproduced without external data.

---

## Русский

**Название ПО:** NOM-HRMS-FGA

**Описание:** программное обеспечение с открытым исходным кодом на языке Python
для деконволюции масс-спектров высокого разрешения (ВЭЖХ-МСВР) природного
органического вещества (ПОВ) / гуминовых веществ. Метод сочетает селективную
химическую дериватизацию с изотопной меткой — дейтерометилирование
карбоксильных групп (–COOH → –COOCD₃, приращение массы
DELTA_CD3 = 17.03448 Да на группу) и дейтероацилирование гидроксильных групп
(–OH → –OCOCD₃, приращение массы DELTA_CD3CO = 45.02939 Да на группу) — что
позволяет подсчитывать функциональные группы, назначать молекулярные формулы в
режиме отрицательных ионов ([M–H]⁻) и перечислять возможные молекулярные
структуры.

**Репозиторий:** https://github.com/MikhaylenkoVS/NOM-HRMS-FGA

**Версия / тег:** v0.4.0

**Архивная версия (DOI):** [DOI будет присвоен при архивировании релиза на Zenodo]

**Лицензия:** GPL-3.0 (см. файл `LICENSE`). Сторонний код в каталоге
`external/usrednenie_spectrov_i_hromatogramm/` распространяется под лицензией
GPL-3.0 и сохраняет собственную лицензию.

**Язык программирования / среда выполнения:** Python ≥ 3.10.

**Установка и запуск:**

*Windows (конечные пользователи):*
1. Скачайте `NOM_HRMS_FGA.exe` со страницы [GitHub Releases](https://github.com/MikhaylenkoVS/NOM-HRMS-FGA/releases)
2. Запустите двойным кликом — Python, Git и командная строка не требуются.

*Все платформы (разработчики):*
```bash
# Установка одной командой из GitHub
pip install git+https://github.com/MikhaylenkoVS/NOM-HRMS-FGA.git

# Запуск
nom-hrms-fga           # графический интерфейс
python -m src          # альтернативный способ
```

*Сборка из исходного кода:*
```bash
git clone https://github.com/MikhaylenkoVS/NOM-HRMS-FGA.git
cd NOM-HRMS-FGA
pip install -e ".[dev]"
python build_exe.py    # создаёт dist/NOM_HRMS_FGA.exe (~120 МБ)
```

**Зависимости:** перечислены в файлах `pyproject.toml` (`[project.dependencies]`)
и [`requirements.txt`](requirements.txt).
Основные пакеты: NumPy, pandas, matplotlib, nomspectra, RDKit, Pillow.
Для графического интерфейса дополнительно требуется `tkinter` (в некоторых
дистрибутивах Linux ставится отдельно, например `apt install python3-tk`).

**Воспроизведение (минимальный пример):**

```bash
# 1. Установить
pip install git+https://github.com/MikhaylenkoVS/NOM-HRMS-FGA.git

# 2. Запустить автоматический набор тестов (129 тестов)
pytest
```

**Доступность данных:** синтетические тестовые наборы, использованные для
валидации, входят в состав репозитория (`data/test_sets/`, наборы
`set_01` … `set_05`). Каждый набор содержит исходный спектр, а также
дейтерометилированный и дейтероацилированный спектры вместе с эталонной
разметкой, что позволяет воспроизвести работу всего конвейера и его результаты
без привлечения внешних данных.
