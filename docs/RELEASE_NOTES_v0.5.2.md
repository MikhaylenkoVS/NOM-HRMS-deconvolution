# NOM-HRMS-FGA v0.5.2 — Patch release

**2026-07-20** | [Full changelog](https://github.com/MikhaylenkoVS/NOM-HRMS-FGA/compare/v0.5.1...v0.5.2)

Bug-fix and cleanup patch: sulfur/phosphorus valences, print()→logging migration, test hygiene.

---

## 🐛 Bug fixes

| # | Description |
|---|-------------|
| #35 | **Sulfur valence** — raised from 2→6 (`valence_charged`: {0:6, 1:3, -1:1}); fixes SO₂/SO₃H fragment assembly |
| #35 | **Phosphorus valence** — raised from 3→5 (`valence_charged`: {0:5, 1:4}); fixes PO₄ fragment assembly |

---

## 🧹 Cleanup

| # | Description |
|---|-------------|
| #20 | `print()` → `logging`: **23 calls** migrated in `rdkit_bridge.py` (16), `spectrum_ops.py` (3), `app.py` (2), `van_krevelen.py` (1). `pipeline.py` (98) deferred to v0.6 (requires GUI logger redesign). |
| #48 | `random.seed(42)` moved after docstring in `generate_test_sets.py` (was shadowing the docstring) |
| #50 | `TODO` in `test_pipeline_integration.py` replaced with documented workaround comment |

---

## ✅ Verified (already fixed in earlier versions)

| # | Description |
|---|-------------|
| #31 | Bare `except:` audit — zero bare `except:` clauses found in `src/` |
| #33 | `bond_types[order-1]` — uses safe `.get(order, SINGLE)` |
| #47 | pytest markers — already registered in `pytest.ini` |

---

## 📦 Assets

- `Source code` (zip) — full repository snapshot at v0.5.2
- `Source code` (tar.gz) — full repository snapshot at v0.5.2

> **Note:** `NOM_HRMS_FGA.exe` is not rebuilt for this patch — only source changes, no binary impact.
