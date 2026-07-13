# NOM-HRMS-FGA v0.5.0 — Stable release

**2026-07-13** | [Full changelog](https://github.com/MikhaylenkoVS/NOM-HRMS-FGA/compare/v0.4.2...v0.5.0)

Functional-group analysis of natural organic matter from HPLC–HRMS data — stable release with 305 tests, isotope filter, presets, CI, and full code audit.

---

## 🚀 Highlights

- **305 unit tests** (up from 123 in v0.4.2) — all 12 core modules covered
- **¹³C isotope filter** (Beynon formula) — optional checkbox in GUI, penalises formulas with mismatched M+1/M pattern
- **4 config presets** — soil, water, peat, coal; selectable from GUI dropdown
- **GitHub Actions CI** — automated pytest on Python 3.11–3.13 for every push/PR
- **Full v0.4.3 audit** — 8 bugs fixed, 16 `print()` replaced with `logging`, 37 `except` blocks audited
- **Progress bars** — RAW averaging and structure preview no longer appear frozen

---

## 🐛 Bug fixes

| # | Description |
|---|-------------|
| #74 | `_import_folder` fallback now includes `.raw` files (previously CSV-only) |
| #78 | `_auto_plot_hist` exceptions now show user-visible warning instead of silent failure |
| — | `rdkit_bridge.py`: fixed `.attachment_points` → `.get_free_attachment_points()` (highlight mode was broken) |
| — | `van_krevelen.py`: removed unused `import matplotlib` |
| — | `pipeline.py`: all `print(msg, file=sys.stderr)` → `logger.error()` (19 places) |
| — | `fragment_combinations.py`: 16 `print()` → `logging.info()`/`logging.warning()` |

---

## ✨ New features

| # | Feature |
|---|---------|
| #6 | **¹³C isotope filter** — checks theoretical vs real M+1/M ratio (Beynon formula, 20% threshold). Available as checkbox in «Filters» tab. Penalty +2.0 to scoring — does not exclude, only deprioritises. |
| #64 | **Config presets** — 4 parameter presets for typical NOM samples: soil, water, peat, coal. Select from «Files» tab dropdown, apply with one click. |
| #65 | **GitHub Actions CI** — runs `pytest tests/unit/` on push/PR (Python 3.11, 3.12, 3.13). Smoke tests run when test data is present. |
| #79 | **Progress indication** — indeterminate progress bar during RAW averaging and structure preview search. Status updates in the status bar. |
| #82 | **spectrum_ops.py tests** — 27 new tests covering `load_spectrum`, `denoise`, `assign_formulas`, `find_series`, `build_result_table`. |

---

## 📝 Documentation

| # | Description |
|---|-------------|
| #75 | `src/README.md`: removed references to non-existent `result_table.csv` / `van_krevelen.png` |
| #80 | `generate_test_sets.py`: docstring updated to reflect `pipeline.json → test_sets` |
| #81 | `src/README.md`: removed reference to non-existent `core/README.txt` |
| — | `CITATION.cff` — added for GitHub/Zenodo citation support |
| — | `docs/zenodo_metadata.md` — Zenodo deposit template |
| — | `CODE_AVAILABILITY.md` — updated test count (105 → 305), v0.5.0 release info |

---

## 🧪 Testing

| Metric | v0.4.2 | v0.5.0 |
|--------|--------|--------|
| Unit tests | 123 | **305** |
| Core modules covered | 5/12 | **12/12** |
| `print()` calls in source | ~145 | ~100 |
| `print(stderr)` | 19 | **0** |
| CI | none | GitHub Actions |

---

## 📦 Installation & usage

**Windows (end users):** download `NOM_HRMS_FGA.exe` below, double-click to run.

**Developers:**
```bash
pip install git+https://github.com/MikhaylenkoVS/NOM-HRMS-FGA.git@v0.5.0
nom-hrms-fga
```

**Build from source:**
```bash
git clone https://github.com/MikhaylenkoVS/NOM-HRMS-FGA.git
cd NOM-HRMS-FGA
git checkout v0.5.0
pip install -e ".[dev]"
pytest          # 305 tests
```

---

## 📦 Assets

- `NOM_HRMS_FGA.exe` — standalone Windows executable (~120 MB)
- `test_sets.zip` — 5 synthetic test datasets with ground-truth annotations
- `Source code` (zip) — full repository snapshot at v0.5.0
- `Source code` (tar.gz) — full repository snapshot at v0.5.0
