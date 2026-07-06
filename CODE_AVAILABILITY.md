# Code Availability / Доступность программного кода

> Bilingual document (English / Русский) prepared for the *Code Availability*
> section of an ACS journal article.
> Двуязычный документ (English / Русский) для раздела *Code Availability*
> статьи в журнале ACS.

---

## English

**Software name:** NOM-HRMS-deconvolution

**Description:** Open-source Python software for deconvolution of high-resolution
mass spectra (HPLC-HRMS) of natural organic matter (NOM) / humic substances.
The pipeline combines selective chemical derivatization with isotope labelling —
deuteromethylation of carboxyl groups (–COOH → –COOCD₃, mass increment
DELTA_CD3 = 17.03448 Da per group) and deuteroacylation of hydroxyl groups
(–OH → –OCOCD₃, mass increment DELTA_CD3CO = 45.02939 Da per group) — to
count functional groups, assign molecular formulas in negative ion mode
([M–H]⁻), and enumerate candidate molecular structures.

**Repository:** https://github.com/mikhaylenkovs-droid/NOM-HRMS-deconvolution

**Version / Tag:** v0.3.1

**Archived version (DOI):** [DOI будет присвоен при архивировании релиза на Zenodo]

**License:** MIT (see `LICENSE`). Third-party code under
`external/usrednenie_spectrov_i_hromatogramm/` is distributed under GPL-3.0 and
retains its own license.

**Programming language / Runtime:** Python ≥ 3.10.

**Dependencies:** listed in [`requirements.txt`](requirements.txt).
Core packages: NumPy, pandas, matplotlib, nomspectra, RDKit, Pillow, pytest.
The graphical interface additionally requires `tkinter` (a system package on
some Linux distributions, e.g. `apt install python3-tk`).

**Reproduction (minimal example):**

```bash
# 1. Create an environment and install dependencies
pip install -r requirements.txt

# 2. Run the built-in end-to-end test over the bundled synthetic sets
python -m src.core.pipeline --test

# 3. Run the automated test suite (28 tests)
pytest
```

**Data availability:** synthetic test datasets used for validation are bundled
with the repository under `data/test_sets/` (`set_01` … `set_05`). Each set
contains the original spectrum plus the deuteromethylated and deuteroacylated
spectra together with ground-truth annotations, allowing the full pipeline and
its results to be reproduced without external data.

---

## Русский

**Название ПО:** NOM-HRMS-deconvolution

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

**Репозиторий:** https://github.com/mikhaylenkovs-droid/NOM-HRMS-deconvolution

**Версия / тег:** v0.3.1

**Архивная версия (DOI):** [DOI будет присвоен при архивировании релиза на Zenodo]

**Лицензия:** MIT (см. файл `LICENSE`). Сторонний код в каталоге
`external/usrednenie_spectrov_i_hromatogramm/` распространяется под лицензией
GPL-3.0 и сохраняет собственную лицензию.

**Язык программирования / среда выполнения:** Python ≥ 3.10.

**Зависимости:** перечислены в файле [`requirements.txt`](requirements.txt).
Основные пакеты: NumPy, pandas, matplotlib, nomspectra, RDKit, Pillow, pytest.
Для графического интерфейса дополнительно требуется `tkinter` (в некоторых
дистрибутивах Linux ставится отдельно, например `apt install python3-tk`).

**Воспроизведение (минимальный пример):**

```bash
# 1. Создать окружение и установить зависимости
pip install -r requirements.txt

# 2. Запустить встроенный сквозной тест на прилагаемых синтетических наборах
python -m src.core.pipeline --test

# 3. Запустить автоматический набор тестов (28 тестов)
pytest
```

**Доступность данных:** синтетические тестовые наборы, использованные для
валидации, входят в состав репозитория (`data/test_sets/`, наборы
`set_01` … `set_05`). Каждый набор содержит исходный спектр, а также
дейтерометилированный и дейтероацилированный спектры вместе с эталонной
разметкой, что позволяет воспроизвести работу всего конвейера и его результаты
без привлечения внешних данных.
