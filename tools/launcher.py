"""Launcher entry point for PyInstaller-built NOM_HRMS_FGA.exe.

Wraps the main application with a top-level crash handler so that if the
GUI fails to start, the user sees a native error dialog rather than a
silent exit (critical for --windowed / --noconsole builds).
"""

import sys
import traceback
import os

# -- Work around a common tkinter issue in PyInstaller onefile builds:
#    the Tcl/Tk runtime may fail to locate its library path.
#    PyInstaller sets sys._MEIPASS (or a frozen attribute) to the
#    temporary extraction directory.  We ensure Tk can find it.
if getattr(sys, "frozen", False):
    # Running inside a PyInstaller bundle
    base_dir = (
        sys._MEIPASS if hasattr(sys, "_MEIPASS") else os.path.dirname(sys.executable)
    )
    # Add Tcl/Tk library paths
    tcl_path = os.path.join(base_dir, "tcl")
    tk_path = os.path.join(base_dir, "tk")
    if os.path.isdir(tcl_path):
        os.environ.setdefault("TCL_LIBRARY", tcl_path)
    if os.path.isdir(tk_path):
        os.environ.setdefault("TK_LIBRARY", tk_path)

    # Also put the bundle's library dir on PATH so that DLLs (rdkit, etc.)
    # are resolvable at runtime.
    lib_dir = os.path.join(base_dir, "library.zip")  # not a real dir for onefile
    os.environ["PATH"] = base_dir + os.pathsep + os.environ.get("PATH", "")


def _show_crash_dialog(message: str) -> None:
    """Try to display a native error dialog; fall back to stderr."""
    try:
        import ctypes

        ctypes.windll.user32.MessageBoxW(
            0, message, "NOM HRMS FGA — Ошибка запуска", 0x10
        )
    except Exception:
        try:
            import tkinter.messagebox as mb

            mb.showerror("NOM HRMS FGA — Ошибка запуска", message)
        except Exception:
            print(message, file=sys.stderr)


def main():
    try:
        from src.app import main as app_main

        app_main()
    except Exception:
        tb = traceback.format_exc()
        _show_crash_dialog(
            "Не удалось запустить приложение.\n\n"
            "Убедитесь, что у вас установлены все необходимые "
            "системные компоненты (Visual C++ Redistributable).\n\n"
            f"Техническая информация:\n{tb[-2000:]}"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
