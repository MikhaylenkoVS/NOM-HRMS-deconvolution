# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for NOM HRMS FGA — single-file Windows .exe.

Usage:
    pyinstaller tools/NOM_HRMS_FGA.spec

Result:  ``dist/NOM_HRMS_FGA.exe``  (≈200–400 MB, self-contained).
"""
from __future__ import annotations

import os
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_submodules, collect_data_files

# ---------------------------------------------------------------------------
# Project root (this spec file's directory)
# ---------------------------------------------------------------------------
_PROJECT = Path(SPECPATH)  # type: ignore[name-defined]  # noqa: F821

# ---------------------------------------------------------------------------
# Hidden imports — modules imported lazily, dynamically, or via __import__
# ---------------------------------------------------------------------------
_hidden: list[str] = [
    # tkinter & friends
    "tkinter", "tkinter.ttk", "tkinter.filedialog", "tkinter.messagebox",
    "tkinter.scrolledtext", "_tkinter",
    # matplotlib
    "matplotlib.backends.backend_tkagg",
    "matplotlib.backends._tkagg",
    # numpy
    "numpy.core._methods", "numpy.lib.format",
    # pandas
    "pandas._libs.tslibs",
    # misc
    "queue", "threading", "warnings", "logging",
    "json", "csv", "io", "ast", "re", "math",
    "ctypes",
]

# ---------------------------------------------------------------------------
# Collect-all for complex native-extension packages
# ---------------------------------------------------------------------------
_collect_packages = ["rdkit", "matplotlib", "PIL", "nomspectra", "scipy"]
_collected_datas: list[tuple[str, str]] = []
_collected_binaries: list[tuple[str, str]] = []

for _pkg in _collect_packages:
    _d, _b, _h = collect_all(_pkg)
    _collected_datas.extend(_d)
    _collected_binaries.extend(_b)
    _hidden.extend(_h)

# Collect submodules for nomspectra (may be missed by collect_all)
_hidden.extend(collect_submodules("nomspectra"))

# ---------------------------------------------------------------------------
# Data files: config JSONs
# ---------------------------------------------------------------------------
_added_datas: list[tuple[str, str]] = []
_config_dir = _PROJECT / "src" / "configs"
if _config_dir.is_dir():
    for _jf in _config_dir.glob("*.json"):
        _added_datas.append((str(_jf), os.path.join("src", "configs")))

# Also collect matplotlib's mpl-data
_added_datas.extend(collect_data_files("matplotlib"))

# Merge all datas
_all_datas: list[tuple[str, str]] = []
_all_datas.extend(_added_datas)
_all_datas.extend(_collected_datas)

# ---------------------------------------------------------------------------
# Exclusions
# ---------------------------------------------------------------------------
_excludes: list[str] = [
    "IPython", "ipykernel", "jupyter", "notebook",
    "PyQt5", "PyQt6", "PySide2", "PySide6", "wx",
    "zmq", "tornado", "sqlalchemy",
    "sympy", "openpyxl", "xlrd", "xlsxwriter",
    "lxml", "html5lib", "bs4", "sphinx",
    "pytest", "setuptools", "pip", "wheel", "Cython",
    "distutils",
]

# ---------------------------------------------------------------------------
# Icon
# ---------------------------------------------------------------------------
_icon_path = _PROJECT / "assets" / "icon.ico"
_icon = str(_icon_path) if _icon_path.is_file() else None

# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
a = Analysis(
    [str(_PROJECT / "tools" / "launcher.py")],
    pathex=[str(_PROJECT)],
    binaries=_collected_binaries,
    datas=_all_datas,
    hiddenimports=list(dict.fromkeys(_hidden)),  # deduplicate
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=_excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)

# ---------------------------------------------------------------------------
# PYZ + EXE
# ---------------------------------------------------------------------------
pyz = PYZ(a.pure, a.zipped_data, cipher=None)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="NOM_HRMS_FGA",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=True,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=_icon,
)
