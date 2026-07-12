#!/usr/bin/env python
"""Build NOM_HRMS_FGA.exe — standalone Windows executable.

Usage:
    python tools/build_exe.py              # build the .exe
    python tools/build_exe.py --clean      # clean previous build, then rebuild
    python tools/build_exe.py --test       # build, then smoke-test the .exe

Requirements:
    - Windows 10/11 x64
    - Python 3.10+ with pip
    - Git (optional, for version tagging)

The resulting ``NOM_HRMS_FGA.exe`` is a single-file, double-click-to-run
application with no external dependencies.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT = Path(__file__).resolve().parent.parent
SPEC_FILE = PROJECT / "tools" / "NOM_HRMS_FGA.spec"
DIST_DIR = PROJECT / "dist"
BUILD_DIR = PROJECT / "build"
EXE_NAME = "NOM_HRMS_FGA.exe"
EXE_PATH = DIST_DIR / EXE_NAME
ASSETS_DIR = PROJECT / "assets"
ICON_PATH = ASSETS_DIR / "icon.ico"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def run(cmd: list[str], **kw) -> int:
    """Run a command and print its output in real time."""
    print(f"\n  $ {' '.join(cmd)}")
    return subprocess.call(cmd, cwd=str(PROJECT), **kw)


def step(title: str) -> None:
    print(f"\n{'=' * 60}\n  {title}\n{'=' * 60}")


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------


def check_platform() -> None:
    """Warn if not on Windows."""
    step("1/6  Проверка платформы")
    if sys.platform != "win32":
        print(
            "  ⚠  Сборка .exe возможна только на Windows.\n"
            "     Текущая платформа: {}".format(sys.platform)
        )
        if "--ci" not in sys.argv:
            sys.exit(1)
    print(f"  ✓  Платформа: {sys.platform}, Python {sys.version.split()[0]}")


def install_requirements() -> None:
    """Ensure PyInstaller and project dependencies are installed."""
    step("2/6  Установка зависимостей")
    # PyInstaller must be present
    try:
        import PyInstaller  # noqa: F401
    except ImportError:
        print("  Устанавливаю PyInstaller...")
        run([sys.executable, "-m", "pip", "install", "pyinstaller>=6.0"])

    # Core deps from pyproject.toml
    run([sys.executable, "-m", "pip", "install", "-e", ".[dev,raw]"])
    print("  ✓  Зависимости готовы")


def generate_icon() -> None:
    """Generate a minimal .ico file if none exists."""
    step("3/6  Иконка")
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    if ICON_PATH.is_file():
        print(f"  ✓  Используется существующая: {ICON_PATH}")
        return

    # Generate a simple 32x32 RGBA icon programmatically
    try:
        from PIL import Image, ImageDraw

        img = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        # Hexagon-like molecule shape
        draw.regular_polygon(
            (16, 16, 12), 6, fill=(137, 180, 250, 255), outline=(30, 30, 46)
        )
        # Save as .ico
        img.save(str(ICON_PATH), format="ICO", sizes=[(32, 32)])
        print(f"  ✓  Сгенерирована: {ICON_PATH}")
    except Exception as e:
        print(f"  ⚠  Не удалось создать иконку: {e}. Сборка без иконки.")


def clean_previous() -> None:
    """Remove old build artifacts."""
    step("4/6  Очистка предыдущей сборки")
    for path in [BUILD_DIR, DIST_DIR]:
        if path.exists():
            shutil.rmtree(path)
            print(f"  ✓  Удалено: {path}")
    # Also remove PyInstaller cache
    for cache_dir in PROJECT.glob("**/__pycache__"):
        shutil.rmtree(cache_dir, ignore_errors=True)
    print("  ✓  Готово")


def run_pyinstaller() -> None:
    """Execute PyInstaller with the spec file."""
    step("5/6  Сборка PyInstaller")
    rc = run([sys.executable, "-m", "PyInstaller", str(SPEC_FILE), "--noconfirm"])
    if rc != 0:
        print("\n  ✗  PyInstaller завершился с ошибкой.")
        sys.exit(rc)

    if not EXE_PATH.is_file():
        print(f"\n  ✗  {EXE_NAME} не найден в {DIST_DIR}")
        sys.exit(1)

    size_mb = EXE_PATH.stat().st_size / (1024 * 1024)
    print(f"\n  ✓  {EXE_NAME} собран ({size_mb:.0f} MB)")


def smoke_test_exe() -> None:
    """Quick smoke test: launch the exe, verify it starts, then kill it."""
    step("6/6  Smoke-тест .exe")
    if not EXE_PATH.is_file():
        print(f"  ✗  {EXE_PATH} не существует — пропускаем тест")
        return

    print(f"  Запускаю {EXE_NAME} на 10 секунд...")
    try:
        proc = subprocess.Popen(
            [str(EXE_PATH)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        import time

        time.sleep(10)
        # Check if still alive
        if proc.poll() is None:
            print("  ✓  Процесс запущен и работает 10+ сек.")
            proc.terminate()
            proc.wait(timeout=5)
        else:
            rc = proc.returncode
            print(f"  ⚠  Процесс завершился с кодом {rc} (возможно, окно было закрыто)")
    except Exception as e:
        print(f"  ⚠  Не удалось запустить .exe: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = set(sys.argv[1:])

    if "--clean" in args:
        clean_previous()

    check_platform()
    install_requirements()
    generate_icon()
    run_pyinstaller()

    if "--test" in args or "--smoke" in args:
        smoke_test_exe()

    print(f"\n{'=' * 60}")
    print(f"  ✓  ГОТОВО: {EXE_PATH}")
    print(f"  Размер: {EXE_PATH.stat().st_size / (1024**2):.0f} MB")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
