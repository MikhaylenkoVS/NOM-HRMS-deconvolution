# NOM-HRMS-FGA v0.5.1 — Patch release

**2026-07-20** | [Full changelog](https://github.com/MikhaylenkoVS/NOM-HRMS-FGA/compare/v0.5.0...v0.5.1)

Patch release with application icon fix, additional test coverage, and merge of the outstanding `fix/app-icon` branch.

---

## 🐛 Bug fixes

| # | Description |
|---|-------------|
| — | **Application icon** — taskbar now shows the correct spectrum icon instead of the default tkinter pen |
| — | `icon.ico` converted from fake PNG-in-ICO wrapper (1.17 MB) to proper multi-resolution ICO (113 KB): 16×16 … 256×256 |
| — | **Frozen builds** — icon resolved via `sys._MEIPASS` instead of broken `__file__`-relative path |
| — | **Taskbar icon** — `WM_SETICON` + `ICON_BIG` sent via Win32 `SendMessageW` (tkinter's `iconbitmap` only sets `ICON_SMALL`) |
| — | **Dev mode** — icon path restored: `os.path.join(os.path.dirname(__file__), '..')` to walk up from `src/` to project root |
| — | Icon loading failures are now logged via `logging.exception` instead of silently swallowed |

---

## 🧪 Testing

| Metric | v0.5.0 | v0.5.1 |
|--------|--------|--------|
| Unit tests | 315 | **332** |
| New tests | — | **+17** |
| `raw_bridge` mock coverage | error paths only | ✅ full happy-path mock |
| `pipeline._match_row_by_mass` | untested | ✅ 10 edge cases |

---

## 📦 Installation & usage

**Windows (end users):** download `NOM_HRMS_FGA.exe` below, double-click to run.

**Developers:**
```bash
pip install git+https://github.com/MikhaylenkoVS/NOM-HRMS-FGA.git@v0.5.1
nom-hrms-fga
```

---

## 📦 Assets

- `NOM_HRMS_FGA.exe` — standalone Windows executable (~123 MB)
- `Source code` (zip) — full repository snapshot at v0.5.1
- `Source code` (tar.gz) — full repository snapshot at v0.5.1
