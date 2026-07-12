# tools/ — сборка и вспомогательные утилиты

## Назначение

Сборка автономного `.exe` (Windows) через PyInstaller.

## Состав

| Файл | Назначение |
|------|------------|
| `build_exe.py` | Сценарий сборки `NOM_HRMS_FGA.exe`: установка зависимостей, PyInstaller, smoke-тест |
| `launcher.py` | Crash-safe точка входа для `.exe`: перехват ошибок, диалог диагностики |
| `NOM_HRMS_FGA.spec` | Конфигурация PyInstaller: onefile, windowed, collect_all для rdkit/matplotlib/scipy |

## Использование

```bash
python tools/build_exe.py          # сборка
python tools/build_exe.py --clean  # очистка + сборка
python tools/build_exe.py --test   # сборка + smoke-тест
```

Результат: `dist/NOM_HRMS_FGA.exe` (~120 МБ, автономный).

## Связь с общим конвейером

`launcher.py` импортирует `src.app.main()` — это единственная точка входа для
PyInstaller-сборки. Сам конвейер и GUI находятся в `src/`.
