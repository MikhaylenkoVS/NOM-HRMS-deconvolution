"""
app.py  —  GUI-интерфейс для пайплайна определения -COOH и -OH групп
           Запускать: python app.py
           Требует: tkinter (стандартная библиотека Python), matplotlib, pandas
"""

from __future__ import annotations
from src.core._safety import _safe_df
import ast
import io
import os
import queue
import threading
import traceback
import warnings
import sys
from pathlib import Path
from typing import Optional
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

# RDKit CoordGen: sp3-зигзаги вместо линейных цепочек
try:
    from rdkit.Chem import rdDepictor
    rdDepictor.SetPreferCoordGen(True)
except Exception:
    pass

matplotlib.use("TkAgg")
# ── Импорт UI-утилит ─────────────────────────────────────────────────────────
try:
    from src.ui.plots import embed_figure
    from src.ui.theme import (
        ACCENT,
        BG,
        FG,
        IMG_H,
        IMG_W,
        MONO,
        OK,
        PANEL,
        WARN,
        _mpl_style,
        _style,
    )
    from src.structures.tab import StructureViewerTab

    _UI_LOADED = True
    _UI_ERROR = ""
except Exception as _ui_err:
    _UI_LOADED = False
    _UI_ERROR = traceback.format_exc()
    # Fallback-константы, чтобы GUI хотя бы запустился без src.ui
    BG = "#1e1e2e"
    ACCENT = "#cba6f7"
    PANEL = "#313244"
    WARN = "#f38ba8"
    FG = "#cdd6f4"
    OK = "#a6e3a1"
    MONO = ("Consolas", 9)
    IMG_H = 400
    IMG_W = 800

    def _mpl_style():
        pass

    def _style(root):
        pass

    StructureViewerTab = None

    def embed_figure(fig, parent, toolbar=True):
        """Минимальный fallback через FigureCanvasTkAgg."""
        try:
            from matplotlib.backends.backend_tkagg import (
                FigureCanvasTkAgg,
                NavigationToolbar2Tk,
            )

            canvas = FigureCanvasTkAgg(fig, master=parent)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)
            if toolbar:
                NavigationToolbar2Tk(canvas, parent)
        except Exception:
            plt.show()


# ── Импорт пайплайна ─────────────────────────────────────────────────────────
# ВАЖНО: оставляем ТОЛЬКО один блок импорта из src.core.
# Дублирующий "from core import ..." убран — он тихо перекрывал правильный
# импорт и приводил к расхождению поведения GUI и тестов.
try:
    from src.core import (
        DELTA_CD3,
        DELTA_CD3CO,
        create_van_krevelen_plot,
        find_series,
        load_spectrum,
        run_pipeline,
        visualize_series,
    )

    CORE_LOADED = True
    _CORE_ERROR = ""
except Exception as _core_err:
    CORE_LOADED = False
    _CORE_ERROR = traceback.format_exc()
    # Fallback for names so they are not undefined at startup.
    from src.configs import CHEM

    DELTA_CD3 = CHEM.derivatization_shifts["delta_cd3"]
    DELTA_CD3CO = CHEM.derivatization_shifts["delta_cd3co"]
    run_pipeline = load_spectrum = find_series = visualize_series = None
    create_van_krevelen_plot = None

# ── Импорт raw-бриджа (опционально, только Windows + MSFileReader) ──────────
try:
    from src.core.raw_bridge import average_raw_to_csv, is_available as _raw_available

    _RAW_LOADED = True
    _RAW_ERROR = ""
except Exception as _raw_err:
    _RAW_LOADED = False
    _RAW_ERROR = str(_raw_err)
    average_raw_to_csv = None  # type: ignore[assignment]

    def _raw_available():
        return False


# ── Импорт конфигурации: единый источник дефолтов GUI ─────────────────────
from src.configs import PIPELINE as _PIPE_CFG, PATHS as _PATHS_CFG

_GUI_DEFAULTS = _PIPE_CFG.run_pipeline_defaults
_BRUTTO_DEFAULTS = _PIPE_CFG.default_brutto_dict
_FORMULA_RANGES = _PIPE_CFG.formula_search["ranges"]


# ═══════════════════════════════════════════════════════════════════════════════
#  Перехват stdout/stderr → thread-safe очередь → GUI-лог
# ═══════════════════════════════════════════════════════════════════════════════


class _QueueWriter:
    """Thread-safe stream shim that tees writes to a queue and a stream.

    Used to redirect ``sys.stdout``/``sys.stderr`` so that ``print``
    output from the pipeline (which may run in a worker thread) is
    delivered to the GUI log via a queue while still reaching the original
    stream.

    Parameters
    ----------
    q : queue.Queue
        Queue that receives ``("log", text)`` items.
    original : io.TextIOBase, optional
        Underlying stream to also forward writes to. Default ``None``.
    """

    def __init__(self, q: queue.Queue, original=None):
        self._q = q
        self._orig = original

    def write(self, data: str):
        if data:
            self._q.put(("log", data))
        if self._orig:
            try:
                self._orig.write(data)
            except Exception:
                pass

    def flush(self):
        if self._orig:
            try:
                self._orig.flush()
            except Exception:
                pass

    def fileno(self):
        if self._orig and hasattr(self._orig, "fileno"):
            return self._orig.fileno()
        raise io.UnsupportedOperation("fileno")


# ═══════════════════════════════════════════════════════════════════════════════
#  ГЛАВНОЕ ОКНО
# ═══════════════════════════════════════════════════════════════════════════════


class App(tk.Tk):
    """Main Tkinter window for the -COOH/-OH functional-group analyzer.

    Provides a tabbed interface to load the three input spectra
    (underivatized, deuteromethylated, deuteroacylated), configure and run
    the deconvolution pipeline in a background thread, and inspect the
    results as spectra plots, homologous-series diagrams, per-compound
    histograms, a results table and (optionally) candidate structures.

    Notes
    -----
    The heavy work runs on a worker thread; ``stdout``/``stderr`` are
    redirected through :class:`_QueueWriter` so pipeline messages appear in
    the GUI log without blocking the Tk event loop.
    """

    # ── init ──────────────────────────────────────────────────────────────────

    def __init__(self):
        super().__init__()
        try:
            import os as _os
            _icon = _os.path.join(_os.path.dirname(__file__), '..', 'assets', 'icon.ico')
            if _os.path.exists(_icon):
                self.iconbitmap(_icon)
        except Exception:
            pass
        self.title("NOM HRMS FGA")
        self.geometry("1200x760")
        self.configure(bg=BG)
        self.resizable(True, True)

        try:
            _style(self)
        except Exception as e:
            print(f"[WARN] _style failed: {e}")
        try:
            _mpl_style()
        except Exception as e:
            print(f"[WARN] _mpl_style failed: {e}")

        # ── данные ──
        self.result_df: Optional[pd.DataFrame] = None
        self.src_spec = None
        self.dmet_spec = None
        self.dacet_spec = None
        self.df_dmet_series = None
        self.df_dacet_series = None

        # ── очередь для потоко-безопасного логирования ──
        self._log_queue: queue.Queue = queue.Queue()

        # ── файловые переменные ──
        self.src_var = tk.StringVar()
        self.dmet_var = tk.StringVar()
        self.dacet_var = tk.StringVar()
        self.vk_color_var = tk.StringVar(value="N_COOH")

        # ── RAW-файлы (опционально, вместо CSV) ──

        # ── RT-диапазоны для усреднения RAW ──
        self.src_rt_min = tk.StringVar(value="")
        self.src_rt_max = tk.StringVar(value="")
        self.dmet_rt_min = tk.StringVar(value="")
        self.dmet_rt_max = tk.StringVar(value="")
        self.dacet_rt_min = tk.StringVar(value="")
        self.dacet_rt_max = tk.StringVar(value="")

        # ── параметры (значения из pipeline.json -> run_pipeline_defaults) ──
        self.sep_var = tk.StringVar(value=str(_GUI_DEFAULTS["sep"]))
        self.mass_min_var = tk.StringVar(value=str(_GUI_DEFAULTS["load_mass_min"]))
        self.mass_max_var = tk.StringVar(value=str(_GUI_DEFAULTS["load_mass_max"]))
        self.noise_force_var = tk.StringVar(value=str(_GUI_DEFAULTS["noise_force"]))
        self.noise_int_var = tk.StringVar(value=str(_GUI_DEFAULTS["noise_intensity"]))
        self.noise_method_var = tk.StringVar(value="intensity")
        self.noise_value_var = tk.StringVar(value=str(_GUI_DEFAULTS["noise_intensity"]))
        self.rel_error_var = tk.StringVar(value=str(_GUI_DEFAULTS["rel_error"]))
        self.sign_var = tk.StringVar(value=str(_GUI_DEFAULTS["sign"]))
        self.ppm_tol_var = tk.StringVar(value=str(_GUI_DEFAULTS["ppm_tol"]))
        self.max_groups_var = tk.StringVar(value=str(_GUI_DEFAULTS["max_groups"]))
        self.allow_gaps_var = tk.BooleanVar(value=bool(_GUI_DEFAULTS["allow_gaps"]))
        self.isotope_filter_var = tk.BooleanVar(value=False)
        self.output_csv_var = tk.StringVar(value=str(_PATHS_CFG.default_output_csv))
        # Диапазоны элементов из pipeline.json -> formula_search.ranges
        _r = _FORMULA_RANGES
        self.c_min = tk.StringVar(value=str(_r["C"][0]))
        self.c_max = tk.StringVar(value=str(_r["C"][1]))
        self.h_min = tk.StringVar(value=str(_r["H"][0]))
        self.h_max = tk.StringVar(value=str(_r["H"][1]))
        self.o_min = tk.StringVar(value=str(_r["O"][0]))
        self.o_max = tk.StringVar(value=str(_r["O"][1]))
        self.n_min = tk.StringVar(value=str(_r["N"][0]))
        self.n_max = tk.StringVar(value=str(_r["N"][1]))

        self._build_ui()

        # Опрос очереди стартует после построения UI
        self._poll_log_queue()

        # Корректное завершение при закрытии окна (крестик)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # Отложенные предупреждения об ошибках импорта
        if not CORE_LOADED:
            self._log(f"[ОШИБКА] src.core не загружен:\n{_CORE_ERROR}", color=WARN)
        if not _UI_LOADED:
            self._log(
                f"[WARN] src.ui / src.structures не загружены:\n{_UI_ERROR}", color=WARN
            )

    # ── потоко-безопасный опрос очереди ──────────────────────────────────────

    def _poll_log_queue(self):
        """Вызывается из главного потока каждые 50 мс."""
        try:
            while True:
                kind, data = self._log_queue.get_nowait()
                if kind == "log":
                    self._append_log_raw(data)
                elif kind == "success":
                    self._on_run_success_data(data)
                elif kind == "error":
                    self._on_run_error_data(data)
        except queue.Empty:
            pass
        finally:
            self._poll_id = self.after(50, self._poll_log_queue)

    def _on_close(self):
        """Корректное завершение: остановка poll, закрытие matplotlib, выход."""
        try:
            self.after_cancel(self._poll_id)
        except Exception:
            pass
        try:
            import sys as _sys
            _sys.stdout = _sys.__stdout__
            _sys.stderr = _sys.__stderr__
        except Exception:
            pass
        try:
            import matplotlib.pyplot as _plt
            _plt.close("all")
        except Exception:
            pass
        self.destroy()
        import os as _os
        _os._exit(0)

    # ── методы лога ───────────────────────────────────────────────────────────

    def _append_log_raw(self, text: str):
        """Вставка сырого текста без тега (безопасна из любого контекста)."""
        if not hasattr(self, "log_text") or self.log_text is None:
            return
        self.log_text.configure(state="normal")
        self.log_text.insert("end", text)
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _log(self, msg: str, color: str = FG):
        """Вставка строки с цветовым тегом. Только из главного потока."""
        if not hasattr(self, "log_text") or self.log_text is None:
            return
        tag = "ok" if color == OK else ("warn" if color == WARN else "info")
        self.log_text.configure(state="normal")
        self.log_text.insert("end", msg + "\n", tag)
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _clear_log(self):
        if not hasattr(self, "log_text"):
            return
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")

    def _save_log(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".txt", filetypes=[("Text", "*.txt"), ("All", "*.*")]
        )
        if path:
            content = self.log_text.get("1.0", "end")
            try:
                Path(path).write_text(content, encoding="utf-8")
                self._log(f"Лог сохранён: {path}", color=OK)
            except Exception as e:
                messagebox.showerror("Ошибка", str(e))

    # ── статус ────────────────────────────────────────────────────────────────

    def _set_status(self, msg: str):
        if hasattr(self, "status_var"):
            self.status_var.set(msg)
        self.update_idletasks()

    def _clear_frame(self, parent):
        for child in parent.winfo_children():
            child.destroy()

    # ── коллбеки воркера (всегда в главном потоке) ───────────────────────────

    def _on_run_success_data(self, payload: dict):
        self.progress.stop()
        result_df = payload.get("result")
        self.result_df = result_df
        n = len(result_df) if result_df is not None else 0
        self._set_status(f"Готово. Найдено {n} соединений.")
        self._log("✅ Анализ завершён успешно.", color=OK)
        if result_df is not None and not result_df.empty:
            self._fill_result_table(result_df)
            self._auto_plot_hist()
            # Автообновление списка соединений во вкладке Структуры
            self._refresh_structures_tab()
        else:
            self._log("[WARN] Результирующая таблица пуста.", color=WARN)

    def _on_run_error_data(self, payload: dict):
        self.progress.stop()
        tb = payload["traceback"]
        self._set_status("Ошибка! Смотри лог.")
        self._log("[ОШИБКА ВЫПОЛНЕНИЯ]\n" + tb, color=WARN)
        messagebox.showerror("Ошибка выполнения", tb[:1200])

    # ── ПОСТРОЕНИЕ ИНТЕРФЕЙСА ─────────────────────────────────────────────────

    def _build_ui(self):
        hdr = tk.Label(
            self,
            text="⚗  NOM HRMS FGA",
            bg=BG,
            fg=ACCENT,
            font=("Segoe UI", 16, "bold"),
        )
        hdr.pack(fill="x", padx=16, pady=(12, 4))
        ttk.Separator(self, orient="horizontal").pack(fill="x", padx=16, pady=2)

        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=16, pady=8)

        self.tab_params = ttk.Frame(nb)
        self.tab_spectra = ttk.Frame(nb)
        self.tab_series = ttk.Frame(nb)
        self.tab_result = ttk.Frame(nb)
        self.tab_van_krevelen = ttk.Frame(nb)
        self.tab_log = ttk.Frame(nb)

        nb.add(self.tab_params, text="⚙  Параметры")
        nb.add(self.tab_spectra, text="📈  Спектры")
        nb.add(self.tab_series, text="🔗  Серии")
        nb.add(self.tab_result, text="📊  Результаты")
        nb.add(self.tab_van_krevelen, text="🌿  Van Krevelen")
        nb.add(self.tab_log, text="📋  Лог")

        if StructureViewerTab is not None:
            try:
                self.tab_struct = StructureViewerTab(nb, app=self)
                nb.add(self.tab_struct, text="🧪  Структуры")
            except Exception as e:
                self._log_queue.put(
                    ("log", f"[WARN] StructureViewerTab init failed: {e}\n")
                )

        self._build_params_tab()
        self._build_spectra_tab()
        self._build_series_tab()
        self._build_result_tab()
        self._build_van_krevelen_tab()
        self._build_log_tab()

        self.status_var = tk.StringVar(value="Готов к работе")
        tk.Label(
            self,
            textvariable=self.status_var,
            bg=PANEL,
            fg=FG,
            font=("Segoe UI", 9),
            anchor="w",
            padx=8,
        ).pack(fill="x", side="bottom")
        self.progress = ttk.Progressbar(self, mode="indeterminate", length=200)
        self.progress.pack(fill="x", side="bottom")

    # ── ВКЛАДКА ПАРАМЕТРОВ ────────────────────────────────────────────────────

    def _build_params_tab(self):
        p = self.tab_params
        p.columnconfigure(0, weight=1)
        p.rowconfigure(0, weight=1)   # подвкладки растягиваются

        sub_nb = ttk.Notebook(p)
        sub_nb.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)

        self._build_params_files(sub_nb)
        self._build_params_processing(sub_nb)
        self._build_params_formulas(sub_nb)
        self._build_params_series(sub_nb)
        self._build_params_advanced(sub_nb)

        try:
            run_btn = ttk.Button(
                p, text="▶  Запустить анализ", style="Accent.TButton", command=self._run
            )
        except Exception:
            run_btn = ttk.Button(p, text="▶  Запустить анализ", command=self._run)
        run_btn.grid(row=1, column=0, pady=12, ipadx=20, ipady=4)

    def _build_params_files(self, nb: ttk.Notebook):
        frame = ttk.Frame(nb)
        nb.add(frame, text="📂  Файлы")
        frame.columnconfigure(0, weight=1)
        files_lf = ttk.LabelFrame(frame, text="Входные спектры")
        files_lf.grid(row=0, column=0, sticky="ew", padx=8, pady=6)
        files_lf.columnconfigure(1, weight=1)

        rt_configs = [
            (self.src_var,   self.src_rt_min,   self.src_rt_max),
            (self.dmet_var,  self.dmet_rt_min,  self.dmet_rt_max),
            (self.dacet_var, self.dacet_rt_min, self.dacet_rt_max),
        ]
        for i, (label, (spec_var, rt_min_var, rt_max_var)) in enumerate([
            ("Исходный спектр:",     rt_configs[0]),
            ("Дейтерометилирование:", rt_configs[1]),
            ("Дейтероацилирование:",  rt_configs[2]),
        ]):
            base_row = i * 2
            ttk.Label(files_lf, text=label).grid(
                row=base_row, column=0, sticky="w", padx=6, pady=4)
            ttk.Entry(files_lf, textvariable=spec_var, width=55).grid(
                row=base_row, column=1, sticky="ew", padx=4, pady=4)
            ttk.Button(files_lf, text="...",
                       command=lambda v=spec_var: self._browse(v)).grid(
                row=base_row, column=2, padx=4, pady=4)
            # RT-диапазон (под полем ввода, для .raw)
            rt_frame = ttk.Frame(files_lf)
            rt_frame.grid(row=base_row + 1, column=1, columnspan=2,
                          sticky="w", padx=4, pady=(0, 6))
            ttk.Label(rt_frame, text="RT, мин:").pack(side="left")
            ttk.Entry(rt_frame, textvariable=rt_min_var, width=5).pack(
                side="left", padx=2)
            ttk.Label(rt_frame, text="–").pack(side="left")
            ttk.Entry(rt_frame, textvariable=rt_max_var, width=5).pack(
                side="left", padx=2)
            ttk.Label(rt_frame, text="(если .raw)",
                      foreground="gray").pack(side="left", padx=4)
        out_lf = ttk.LabelFrame(frame, text="💾  Выходной файл")
        out_lf.grid(row=1, column=0, sticky="ew", padx=8, pady=6)
        out_lf.columnconfigure(0, weight=1)
        ttk.Entry(out_lf, textvariable=self.output_csv_var, width=50).grid(row=0, column=0, sticky="ew", padx=6, pady=4)
        ttk.Button(out_lf, text="...", command=lambda: self._save_browse(self.output_csv_var)).grid(row=0, column=1, padx=4, pady=4)

        # ── Пресеты ──
        preset_lf = ttk.LabelFrame(frame, text="🎯  Пресеты параметров")
        preset_lf.grid(row=2, column=0, sticky="ew", padx=8, pady=6)
        preset_lf.columnconfigure(0, weight=1)
        self.preset_var = tk.StringVar(value="")
        try:
            from src.configs.presets_loader import list_presets
            presets = list_presets()
            preset_names = [f"{p['name']}" for p in presets]
        except Exception as e:
            self._log(f"[WARN] Пресеты не загружены: {e}", color="info")
            presets = []
            preset_names = ["(пресеты недоступны)"]
        cb = ttk.Combobox(
            preset_lf, textvariable=self.preset_var,
            values=preset_names, state="readonly", width=45,
        )
        cb.pack(side="left", padx=6, pady=4, fill="x", expand=True)
        ttk.Button(
            preset_lf, text="Применить",
            command=lambda: self._apply_preset(presets),
        ).pack(side="left", padx=4, pady=4)
        self._presets_data = presets

        # Кнопка импорта целой папки
        ttk.Button(frame, text="📁 Импорт папки со спектрами",
                   command=self._import_folder).grid(
            row=3, column=0, sticky="w", padx=8, pady=(4, 2))
        self._folder_path_var = tk.StringVar()
        tk.Label(frame, textvariable=self._folder_path_var,
                 bg=BG, fg=ACCENT, font=("Segoe UI", 8), anchor="w").grid(
            row=3, column=0, sticky="ew", padx=12, pady=(0, 2))

    def _build_params_processing(self, nb: ttk.Notebook):
        frame = ttk.Frame(nb)
        nb.add(frame, text="📏  Обработка")
        frame.columnconfigure(0, weight=1)
        load_lf = ttk.LabelFrame(frame, text="Загрузка и фильтрация")
        load_lf.grid(row=0, column=0, sticky="ew", padx=8, pady=6)
        for row, (lbl, var) in enumerate([
            ("Разделитель CSV:", self.sep_var), ("m/z min:", self.mass_min_var), ("m/z max:", self.mass_max_var),
        ]):
            ttk.Label(load_lf, text=lbl).grid(row=row, column=0, sticky="w", padx=6, pady=3)
            ttk.Entry(load_lf, textvariable=var, width=12).grid(row=row, column=1, sticky="w", padx=4, pady=3)
        ttk.Label(load_lf, text="Шумоподавление:").grid(row=3, column=0, sticky="w", padx=6, pady=3)
        noise_methods = ["force", "intensity", "quantile"]
        noise_names = {"force": "Force (S/N, 1.5-3)", "intensity": "Абс. интенсивность (100)", "quantile": "Квантиль (0.01)"}
        self._noise_cb = ttk.Combobox(load_lf, textvariable=self.noise_method_var, values=[noise_names[m] for m in noise_methods], width=28, state="readonly")
        self._noise_cb.grid(row=3, column=1, sticky="w", padx=4, pady=3)
        self._noise_cb.bind("<<ComboboxSelected>>", self._on_noise_method_change)
        self._noise_cb.current(1)  # default = intensity
        ttk.Label(load_lf, text="Значение:").grid(row=4, column=0, sticky="w", padx=6, pady=3)
        ttk.Entry(load_lf, textvariable=self.noise_value_var, width=12).grid(row=4, column=1, sticky="w", padx=4, pady=3)

    def _build_params_formulas(self, nb: ttk.Notebook):
        frame = ttk.Frame(nb)
        nb.add(frame, text="🔬  Формулы")
        frame.columnconfigure(0, weight=1)
        lf = ttk.LabelFrame(frame, text="Назначение брутто-формул")
        lf.grid(row=0, column=0, sticky="ew", padx=8, pady=6)
        ttk.Label(lf, text="Знак иона:").grid(row=0, column=0, sticky="w", padx=6, pady=3)
        ttk.Combobox(lf, textvariable=self.sign_var, values=["-", "+"], width=5, state="readonly").grid(row=0, column=1, sticky="w", padx=4, pady=3)
        ttk.Label(lf, text="Погрешность (ppm):").grid(row=1, column=0, sticky="w", padx=6, pady=3)
        ttk.Entry(lf, textvariable=self.rel_error_var, width=8).grid(row=1, column=1, sticky="w", padx=4, pady=3)
        ttk.Label(lf, text="Диапазоны элементов:").grid(row=2, column=0, columnspan=2, sticky="w", padx=6, pady=(10, 2))
        for i, (sym, mn, mx) in enumerate([("C", self.c_min, self.c_max), ("H", self.h_min, self.h_max), ("O", self.o_min, self.o_max), ("N", self.n_min, self.n_max)]):
            r = 3 + i
            ttk.Label(lf, text=f"{sym}:").grid(row=r, column=0, sticky="w", padx=20, pady=2)
            ef = ttk.Frame(lf)
            ef.grid(row=r, column=1, sticky="w", padx=4, pady=2)
            ttk.Entry(ef, textvariable=mn, width=5).pack(side="left")
            ttk.Label(ef, text="-").pack(side="left", padx=2)
            ttk.Entry(ef, textvariable=mx, width=5).pack(side="left")

    def _build_params_series(self, nb: ttk.Notebook):
        frame = ttk.Frame(nb)
        nb.add(frame, text="🔍  Серии")
        frame.columnconfigure(0, weight=1)
        lf = ttk.LabelFrame(frame, text="Поиск гомологических серий")
        lf.grid(row=0, column=0, sticky="ew", padx=8, pady=6)
        ttk.Label(lf, text="Допуск поиска (ppm):").grid(row=0, column=0, sticky="w", padx=6, pady=3)
        ttk.Entry(lf, textvariable=self.ppm_tol_var, width=8).grid(row=0, column=1, sticky="w", padx=4, pady=3)
        ttk.Label(lf, text="Макс. число групп:").grid(row=1, column=0, sticky="w", padx=6, pady=3)
        ttk.Entry(lf, textvariable=self.max_groups_var, width=8).grid(row=1, column=1, sticky="w", padx=4, pady=3)
        ttk.Checkbutton(lf, text="Разрешить пропуски в сериях", variable=self.allow_gaps_var).grid(row=2, column=0, columnspan=2, sticky="w", padx=6, pady=4)

    def _build_params_advanced(self, nb: ttk.Notebook):
        frame = ttk.Frame(nb)
        nb.add(frame, text="🧪  Фильтры")
        frame.columnconfigure(0, weight=1)
        lf = ttk.LabelFrame(frame, text="Дополнительные фильтры")
        lf.grid(row=0, column=0, sticky="ew", padx=8, pady=6)
        cb = ttk.Checkbutton(
            lf,
            text="🔬 Изотопный фильтр ¹³C (формула Бейнона)",
            variable=self.isotope_filter_var,
        )
        cb.grid(row=0, column=0, sticky="w", padx=6, pady=4)
        ttk.Label(
            lf,
            text="Штрафует формулы, чей изотопный паттерн M+1/M\n"
                 "отличается от теоретического более чем на 20%.\n"
                 "Проверка — по исходному спектру до шумоподавления.",
            font=("Segoe UI", 8),
            foreground="#888",
        ).grid(row=1, column=0, sticky="w", padx=12, pady=(0, 6))

    def _build_spectra_tab(self):
        frame = self.tab_spectra
        ctrl = ttk.Frame(frame)
        ctrl.pack(fill="x", pady=4, padx=8)
        ttk.Button(ctrl, text="📈 Построить спектры", command=self._plot_spectra).pack(
            side="left", padx=4
        )
        ttk.Button(
            ctrl,
            text="🗑 Очистить",
            command=lambda: self._clear_frame(self.spectra_canvas_frame),
        ).pack(side="left", padx=4)
        self.spectra_canvas_frame = ttk.Frame(frame)
        self.spectra_canvas_frame.pack(fill="both", expand=True)

    # ── ВКЛАДКА СЕРИИ ─────────────────────────────────────────────────────────

    def _build_series_tab(self):
        frame = self.tab_series
        ctrl = ttk.Frame(frame)
        ctrl.pack(fill="x", pady=4, padx=8)
        ttk.Button(
            ctrl,
            text="🔗 Показать серии CD₃",
            command=lambda: self._plot_series("dmet"),
        ).pack(side="left", padx=4)
        ttk.Button(
            ctrl,
            text="🔗 Показать серии CD₃CO",
            command=lambda: self._plot_series("dacet"),
        ).pack(side="left", padx=4)
        ttk.Button(
            ctrl,
            text="🗑 Очистить",
            command=lambda: self._clear_frame(self.series_canvas_frame),
        ).pack(side="left", padx=4)
        self.series_canvas_frame = ttk.Frame(frame)
        self.series_canvas_frame.pack(fill="both", expand=True)

    # ── ВКЛАДКА РЕЗУЛЬТАТОВ ───────────────────────────────────────────────────

    def _build_result_tab(self):
        frame = self.tab_result
        frame.columnconfigure(0, weight=3)
        frame.columnconfigure(1, weight=1)
        frame.rowconfigure(1, weight=1)

        ctrl = ttk.Frame(frame)
        ctrl.grid(row=0, column=0, columnspan=2, sticky="ew", pady=4, padx=8)
        ttk.Button(ctrl, text="📊 Гистограмма N_COOH",
                   command=lambda: self._plot_hist("N_COOH")).pack(side="left", padx=4)
        ttk.Button(ctrl, text="📊 Гистограмма N_OH",
                   command=lambda: self._plot_hist("N_OH")).pack(side="left", padx=4)
        ttk.Button(ctrl, text="💾 Экспорт CSV",
                   command=self._export_csv).pack(side="left", padx=4)
        ttk.Button(ctrl, text="📂 Импорт CSV",
                   command=self._import_csv).pack(side="left", padx=4)

        # Левая часть — таблица (без колонок пропусков)
        tbl_frame = ttk.Frame(frame)
        tbl_frame.grid(row=1, column=0, sticky="nsew", padx=(8, 4), pady=4)

        cols = ("mass", "brutto", "N_COOH", "N_OH")
        col_labels = ["m/z", "Формула", "N_COOH", "N_OH"]
        col_widths = [120, 180, 90, 90]

        self.result_tree = ttk.Treeview(
            tbl_frame, columns=cols, show="headings", height=18)
        for c, lbl, w in zip(cols, col_labels, col_widths):
            self.result_tree.heading(c, text=lbl,
                                     command=lambda _c=c: self._sort_tree(_c))
            self.result_tree.column(c, width=w, anchor="center")

        vsb = ttk.Scrollbar(tbl_frame, orient="vertical",
                            command=self.result_tree.yview)
        hsb = ttk.Scrollbar(tbl_frame, orient="horizontal",
                            command=self.result_tree.xview)
        self.result_tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        vsb.pack(side="right", fill="y")
        hsb.pack(side="bottom", fill="x")
        self.result_tree.pack(fill="both", expand=True)
        self.result_tree.bind("<Double-1>", self._on_formula_double_click)
        self.result_tree.bind("<<TreeviewSelect>>", self._on_result_row_select)

        # Правая часть — превью структуры
        preview_frame = ttk.LabelFrame(frame, text="🧪  Структура")
        preview_frame.grid(row=1, column=1, sticky="nsew", padx=(4, 8), pady=4)
        self._structure_preview_label = tk.Label(
            preview_frame, text="Кликните на строку\nтаблицы для просмотра",
            bg=PANEL, fg=FG, font=("Segoe UI", 10), justify="center")
        self._structure_preview_label.pack(expand=True, fill="both", padx=8, pady=8)
        self._structure_preview_img = None

        self.hist_frame = ttk.Frame(frame)
        self.hist_frame.grid(row=2, column=0, columnspan=2, sticky="ew", padx=8, pady=4)

    # ── ВКЛАДКА VAN KREVELEN ─────────────────────────────────────────────────

    def _build_van_krevelen_tab(self):
        frame = self.tab_van_krevelen
        ctrl = ttk.Frame(frame)
        ctrl.pack(fill="x", pady=4, padx=8)
        ttk.Button(
            ctrl,
            text="📈 Построить диаграмму Ван Кревелена",
            command=self._plot_van_krevelen,
        ).pack(side="left", padx=4)
        ttk.Button(
            ctrl,
            text="💾 Скачать PNG",
            command=self._save_van_krevelen_png,
        ).pack(side="left", padx=4)
        ttk.Button(
            ctrl,
            text="🗑 Очистить",
            command=lambda: self._clear_frame(self.vk_canvas_frame),
        ).pack(side="left", padx=4)
        ttk.Label(ctrl, text="  Цвет по:").pack(side="left", padx=(12, 2))
        self._vk_color_cb = ttk.Combobox(
            ctrl, textvariable=self.vk_color_var,
            values=["N_COOH", "N_OH"], width=8, state="readonly")
        self._vk_color_cb.pack(side="left", padx=4)
        self._vk_color_cb.bind("<<ComboboxSelected>>", lambda e: self._plot_van_krevelen())
        self.vk_canvas_frame = ttk.Frame(frame)
        self.vk_canvas_frame.pack(fill="both", expand=True)
        # Храним ссылку на последнюю построенную фигуру для сохранения
        self._vk_figure = None

    # ── ВКЛАДКА ЛОГ ──────────────────────────────────────────────────────────

    def _build_log_tab(self):
        frame = self.tab_log
        ctrl = ttk.Frame(frame)
        ctrl.pack(fill="x", pady=4, padx=8)
        ttk.Button(ctrl, text="🗑 Очистить лог", command=self._clear_log).pack(
            side="left", padx=4
        )
        ttk.Button(ctrl, text="💾 Сохранить лог", command=self._save_log).pack(
            side="left", padx=4
        )

        self.log_text = scrolledtext.ScrolledText(
            frame,
            bg=PANEL,
            fg=FG,
            font=MONO,
            relief="flat",
            insertbackground=FG,
            state="disabled",
        )
        self.log_text.pack(fill="both", expand=True, padx=8, pady=4)
        self.log_text.tag_config("ok", foreground=OK)
        self.log_text.tag_config("warn", foreground=WARN)
        self.log_text.tag_config("info", foreground=ACCENT)

    # ═══════════════════════════════════════════════════════════════════════════
    #  ДЕЙСТВИЯ
    # ═══════════════════════════════════════════════════════════════════════════

    def _browse(self, var: tk.StringVar):
        path = filedialog.askopenfilename(
            filetypes=[
                ("CSV / RAW files", "*.csv;*.raw"),
                ("CSV files", "*.csv"),
                ("RAW files", "*.raw"),
                ("Text files", "*.txt"),
                ("All files", "*.*"),
            ]
        )
        if path:
            var.set(path)

    def _save_browse(self, var: tk.StringVar):
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if path:
            var.set(path)

    # ── Денойз: обновление значения при смене метода ──────────────────────────

    def _on_noise_method_change(self, event=None):
        """Update the parameter field to a suggested default for the selected method."""
        defaults = {"force": "1.5", "intensity": "100", "quantile": "0.01"}
        method = self.noise_method_var.get()
        # Extract the method key from the display string
        for key, name in {
            "force": "Force",
            "intensity": "Абс. интенсивность",
            "quantile": "Квантиль",
        }.items():
            if name in method:
                self.noise_value_var.set(defaults[key])
                return

    # ── Валидация и парсинг параметров ────────────────────────────────────────

    def _parse_params(self) -> Optional[dict]:
        """
        Читает все параметры из виджетов и валидирует типы.
        При ошибке логирует, показывает messagebox и возвращает None.
        """
        errors = []

        def _float(var, name, default=None):
            try:
                return float(var.get())
            except ValueError:
                errors.append(f"  • {name}: ожидается число, получено «{var.get()}»")
                return default

        def _int(var, name, default=None):
            try:
                return int(var.get())
            except ValueError:
                errors.append(f"  • {name}: ожидается целое, получено «{var.get()}»")
                return default

        sep = self.sep_var.get()
        if sep in ("\\t", "tab", "TAB"):
            sep = "\t"

        mass_min = _float(self.mass_min_var, "m/z min", 0.0)
        mass_max = _float(self.mass_max_var, "m/z max", 1000.0)
        # Денойз: взаимоисключающие параметры (intensity > quantile > force)
        noise_method = self.noise_method_var.get()
        noise_value = _float(self.noise_value_var, "Шум значение", 1.5)
        if "intensity" in noise_method or "Абс. интенсивность" in noise_method:
            noise_force, noise_int, noise_quantile = None, noise_value, None
        elif "quantile" in noise_method or "Квантиль" in noise_method:
            noise_force, noise_int, noise_quantile = None, None, noise_value
        else:
            noise_force, noise_int, noise_quantile = noise_value, None, None
        rel_error = _float(self.rel_error_var, "Погрешность ppm", 0.5)
        ppm_tol = _float(self.ppm_tol_var, "Допуск поиска ppm", 0.5)
        max_groups = _int(self.max_groups_var, "Макс. групп", 20)

        try:
            c_min = int(self.c_min.get())
            c_max = int(self.c_max.get())
            h_min = int(self.h_min.get())
            h_max = int(self.h_max.get())
            o_min = int(self.o_min.get())
            o_max = int(self.o_max.get())
            n_min = int(self.n_min.get())
            n_max = int(self.n_max.get())
        except ValueError as e:
            errors.append(f"  • Диапазон элементов: {e}")
            _r = _FORMULA_RANGES
            c_min = _r["C"][0]
            c_max = _r["C"][1]
            h_min = _r["H"][0]
            h_max = _r["H"][1]
            o_min = _r["O"][0]
            o_max = _r["O"][1]
            n_min = _r["N"][0]
            n_max = _r["N"][1]

        if mass_min is not None and mass_max is not None and mass_min >= mass_max:
            errors.append(f"  • m/z min ({mass_min}) ≥ m/z max ({mass_max})")

        if errors:
            msg = "Ошибки в параметрах:\n" + "\n".join(errors)
            self._log("[DEBUG] Ошибки валидации:\n" + msg, color=WARN)
            messagebox.showerror("Ошибки параметров", msg)
            return None

        brutto_dict = {
            "C": (c_min, c_max),
            "H": (h_min, h_max),
            "O": (o_min, o_max),
            "N": (n_min, n_max),
        }

        return dict(
            sep=sep,
            load_mass_min=mass_min,
            load_mass_max=mass_max,
            noise_force=noise_force,
            noise_intensity=noise_int,
            noise_quantile=noise_quantile,
            rel_error=rel_error,
            sign=self.sign_var.get(),
            assign_mass_min=_GUI_DEFAULTS["assign_mass_min"],
            assign_mass_max=_GUI_DEFAULTS["assign_mass_max"],
            ppm_tol=ppm_tol,
            max_groups=max_groups,
            allow_gaps=self.allow_gaps_var.get(),
            isotope_filter=self.isotope_filter_var.get(),
            brutto_dict=brutto_dict,
            output_csv=self.output_csv_var.get() or None,
            visualize=False,  # визуализацию делаем через GUI-вкладку
        )

    # ── Запуск ────────────────────────────────────────────────────────────────

    def _resolve_path(self, spec_var, rt_min_var, rt_max_var, label):
        """Return the actual CSV path, auto-detecting RAW→CSV if needed."""
        path = spec_var.get().strip()
        if not path:
            raise ValueError(f"[{label}] Укажите файл (.csv или .raw)")

        if not os.path.isfile(path):
            raise FileNotFoundError(f"[{label}] Файл не найден: {path}")

        # Автоопределение: если .raw → усреднить
        if path.lower().endswith(".raw"):
            try:
                rt_min = float(rt_min_var.get()) if rt_min_var.get().strip() else 0.0
                rt_max = float(rt_max_var.get()) if rt_max_var.get().strip() else 999.0
            except ValueError:
                raise ValueError(f"[{label}] Некорректный RT-диапазон")

            if not _RAW_LOADED:
                raise RuntimeError(
                    f"[{label}] Обработка RAW недоступна: {_RAW_ERROR}\n"
                    "Установите MSFileReader 3.1 SP4 и comtypes."
                )
            self._log(
                f"[RAW] Усреднение {path} (RT {rt_min:.1f}–{rt_max:.1f} мин)…",
                color=FG,
            )
            self._set_status("Усреднение RAW-спектра…")
            self.progress.start(10)
            self.update_idletasks()
            self.update()  # flush pending GUI events before blocking COM call
            # TODO(#84): вынести RAW-усреднение в фоновый поток (не блокировать GUI)
            path = average_raw_to_csv(path, rt_min, rt_max)
            self.progress.stop()
            self._set_status("Готово")
            self._log(f"[RAW] → {path}", color=OK)

        return path

    def _run(self):
        if not CORE_LOADED:
            messagebox.showerror(
                "Ошибка", f"src.core не загружен:\n{_CORE_ERROR[:800]}"
            )
            return

        spec_paths = []
        for label, spec_var, rt_min, rt_max in [
            ("Исходный", self.src_var, self.src_rt_min, self.src_rt_max),
            ("Дейтерометилирование", self.dmet_var, self.dmet_rt_min, self.dmet_rt_max),
            (
                "Дейтероацилирование",
                self.dacet_var,
                self.dacet_rt_min,
                self.dacet_rt_max,
            ),
        ]:
            try:
                path = self._resolve_path(spec_var, rt_min, rt_max, label)
                spec_var.set(path)  # записываем результат для лога
                spec_paths.append(path)
            except Exception as e:
                messagebox.showerror("Ошибка", str(e))
                return

        params = self._parse_params()
        if params is None:
            return

        self._clear_log()
        self._log("[DEBUG] ═══ Запуск анализа ═══", color="info")
        self._log(f"[DEBUG]   src   = {spec_paths[0]}", color="info")
        self._log(f"[DEBUG]   dmet  = {spec_paths[1]}", color="info")
        self._log(f"[DEBUG]   dacet = {spec_paths[2]}", color="info")
        self._log(f"[DEBUG]   params = {params}", color="info")

        self.progress.start(10)
        self._set_status("Выполняется анализ…")

        t = threading.Thread(
            target=self._run_worker,
            args=(spec_paths[0], spec_paths[1], spec_paths[2], params),
            daemon=True,
        )
        t.start()

    def _run_worker(self, src_path: str, dmet_path: str, dacet_path: str, params: dict):
        """
        Выполняется в фоновом потоке.
        Все print() из pipeline автоматически попадают в GUI-лог через _QueueWriter.
        """
        orig_stdout = sys.stdout
        orig_stderr = sys.stderr
        sys.stdout = _QueueWriter(self._log_queue, orig_stdout)
        sys.stderr = _QueueWriter(self._log_queue, orig_stderr)
        try:
            self._log_queue.put(("log", "[DEBUG] _run_worker: старт пайплайна\n"))
            res = run_pipeline(
                src_path=src_path,
                dmet_path=dmet_path,
                dacet_path=dacet_path,
                **params,
            )

            # res — PipelineRunResult
            table = getattr(res, "table", None)
            n = len(table) if table is not None else "None"
            self._log_queue.put(
                ("log", f"[DEBUG] _run_worker: pipeline завершён, строк={n}\n")
            )
            self._log_queue.put(
                ("success", {"result": table, "stats": getattr(res, "stats", None)})
            )
        except Exception:
            tb = traceback.format_exc()
            self._log_queue.put(("log", f"[DEBUG] _run_worker: EXCEPTION\n{tb}\n"))
            self._log_queue.put(("error", {"traceback": tb}))
        finally:
            sys.stdout = orig_stdout
            sys.stderr = orig_stderr

    # ── Таблица результатов ───────────────────────────────────────────────────

    def _fill_result_table(self, df: pd.DataFrame):
        self._log(
            f"[DEBUG] _fill_result_table: {len(df)} строк, "
            f"колонки={list(df.columns)}",
            color="info",
        )
        for row in self.result_tree.get_children():
            self.result_tree.delete(row)

        for col, fill in [
            ("N_COOH", 0),
            ("N_OH", 0),
            ("missing_dmet", []),
            ("missing_dacet", []),
            ("brutto", ""),
            ("all_candidates", []),
        ]:
            if col not in df.columns:
                df[col] = fill
                self._log(
                    f"[WARN] Колонка '{col}' отсутствует → заполнено {fill!r}",
                    color=WARN,
                )

        warn_count = 0
        for i, (_, r) in enumerate(df.iterrows()):
            try:
                n_cooh = int(r.get("N_COOH", 0))
                n_oh = int(r.get("N_OH", 0))
            except (ValueError, TypeError):
                n_cooh = 0
                n_oh = 0

            missing_d = r.get("missing_dmet", [])
            missing_a = r.get("missing_dacet", [])
            has_missing = (
                (isinstance(missing_d, list) and len(missing_d) > 0) or
                (isinstance(missing_a, list) and len(missing_a) > 0)
            )

            # Визуальный индикатор: если есть альтернативные формулы-кандидаты
            brutto = r.get("brutto", "")
            candidates = r.get("all_candidates", None)
            has_alternatives = isinstance(candidates, list) and len(candidates) > 1
            brutto_display = f"{brutto}  ▾" if has_alternatives else brutto

            vals = (
                f"{r['mass']:.5f}" if pd.notna(r.get("mass")) else "?",
                brutto_display,
                n_cooh,
                n_oh,
            )
            tags = []
            if has_missing:
                tags.append("warn")
            if has_alternatives:
                tags.append("has_alt")
            tag = tuple(tags) if tags else ""
            if has_missing:
                warn_count += 1
            self.result_tree.insert("", "end", iid=str(i), values=vals, tags=tag)

        self.result_tree.tag_configure("warn", foreground=WARN)
        self.result_tree.tag_configure("has_alt", font=("Consolas", 9, "bold"))
        self._log(
            f"[DEBUG] Таблица: {len(df)} строк, {warn_count} с пропусками.",
            color="info",
        )

    # ── Выбор альтернативной формулы (фича #2) ──────────────────────────────

    # ── Превью структуры при клике на строку ─────────────────────────────

    def _on_result_row_select(self, event):
        """Один клик по строке — показать структуру в правой панели."""
        selection = self.result_tree.selection()
        if not selection:
            return
        iid = selection[0]
        try:
            idx = int(iid)
        except ValueError:
            return
        if self.result_df is None or idx >= len(self.result_df):
            return

        row = self.result_df.iloc[idx]
        brutto = row.get("brutto", "")
        n_cooh = int(row.get("N_COOH", 0))
        n_oh = int(row.get("N_OH", 0))

        if not brutto:
            self._structure_preview_label.configure(
                text="Нет формулы\nдля этого пика")
            return

        self._structure_preview_label.configure(
            text=f"Поиск структуры...\n{brutto}")
        self.progress.start(10)

        t = threading.Thread(target=self._load_structure_preview,
                             args=(brutto, n_cooh, n_oh), daemon=True)
        t.start()

    def _load_structure_preview(self, brutto: str, n_cooh: int, n_oh: int):
        """Фоновый поток: поиск структуры (first_only)."""
        try:
            from src.core import find_and_visualize_molecules
            result = find_and_visualize_molecules(
                brutto, num_cooh=n_cooh, num_oh=n_oh,
                max_bases=8, show_images=False, first_only=True,
            )
            molecules = result.get("molecules", [])
            self.after(0, lambda: self._show_structure_preview(molecules, brutto))
        except Exception:
            self.after(0, lambda: (
                self.progress.stop(),
                self._structure_preview_label.configure(
                    text=f"Не удалось найти\nструктуру для {brutto}")
            ))

    def _show_structure_preview(self, molecules: list, brutto: str):
        """Отображение первой найденной структуры в панели превью."""
        self.progress.stop()
        if not molecules:
            self._structure_preview_label.configure(
                text=f"Структуры не найдены\n{brutto}")
            return

        try:
            from src.structures.rdkit_utils import fragment_to_rdkit, RDKIT_OK
            from io import BytesIO
            from PIL import Image, ImageTk

            mol_info = molecules[0]
            frag = mol_info.get("fragment_object")
            rdmol = fragment_to_rdkit(frag) if frag is not None else None

            if rdmol is not None:
                from rdkit import Chem
                from rdkit.Chem import Draw, AllChem, rdDepictor
                # 2D-координаты (CoordGen — зигзаги sp3)
                rdDepictor.SetPreferCoordGen(True)
                AllChem.Compute2DCoords(rdmol)
                rdmol = Chem.AddHs(rdmol, explicitOnly=True)
                final_mol = Chem.RWMol(rdmol)
                atoms_to_remove = []
                for atom in final_mol.GetAtoms():
                    if atom.GetAtomicNum() == 1:
                        for nbr in atom.GetNeighbors():
                            if nbr.GetAtomicNum() == 6:
                                atoms_to_remove.append(atom.GetIdx())
                                break
                for idx in reversed(sorted(atoms_to_remove)):
                    final_mol.RemoveAtom(idx)
                rdmol = final_mol.GetMol()
                try:
                    Chem.SanitizeMol(rdmol)
                except Exception:
                    pass
                img = Draw.MolToImage(rdmol, size=(300, 200))
                buf = BytesIO()
                img.save(buf, format="PNG")
                buf.seek(0)
                pil_img = Image.open(buf)
                self._structure_preview_img = ImageTk.PhotoImage(pil_img)
                self._structure_preview_label.configure(
                    image=self._structure_preview_img, text="")
            else:
                name = mol_info.get("name", brutto)
                self._structure_preview_label.configure(
                    text=f"{name}\n(нет RDKit-изображения)", image="")
        except Exception:
            self._structure_preview_label.configure(
                text=f"Ошибка отрисовки\n{brutto}", image="")

    def _on_formula_double_click(self, event):
        """Двойной клик по строке таблицы — выбор формулы из кандидатов."""
        selection = self.result_tree.selection()
        if not selection:
            return
        iid = selection[0]
        try:
            idx = int(iid)
        except ValueError:
            return

        if self.result_df is None or idx >= len(self.result_df):
            return

        row = self.result_df.iloc[idx]
        candidates = row.get("all_candidates", None)
        if not isinstance(candidates, list) or len(candidates) <= 1:
            return  # только один кандидат — нечего выбирать

        current_brutto = row.get("brutto", "")

        # Диалоговое окно с выпадающим списком
        dialog = tk.Toplevel(self)
        dialog.title("Выбор брутто-формулы")
        dialog.geometry("380x160")
        dialog.configure(bg=BG)
        dialog.transient(self)
        dialog.grab_set()

        # Центрируем относительно родителя
        dialog.update_idletasks()
        x = self.winfo_x() + (self.winfo_width() - 380) // 2
        y = self.winfo_y() + (self.winfo_height() - 160) // 2
        dialog.geometry(f"+{x}+{y}")

        tk.Label(
            dialog,
            text=f"m/z = {row['mass']:.5f}  —  выберите формулу:",
            bg=BG, fg=FG, font=("Segoe UI", 10),
        ).pack(padx=12, pady=(12, 8))

        combo_var = tk.StringVar(value=current_brutto)
        combo = ttk.Combobox(
            dialog, textvariable=combo_var, values=candidates,
            state="readonly", width=30,
        )
        combo.pack(padx=12, pady=4)

        def _on_ok():
            new_formula = combo_var.get()
            if new_formula and new_formula != current_brutto:
                self._apply_formula_change(idx, new_formula)
            dialog.destroy()

        def _on_cancel():
            dialog.destroy()

        btn_frame = ttk.Frame(dialog)
        btn_frame.pack(pady=12)
        ttk.Button(btn_frame, text="OK", command=_on_ok).pack(side="left", padx=4)
        ttk.Button(btn_frame, text="Отмена", command=_on_cancel).pack(side="left", padx=4)

        combo.bind("<Return>", lambda e: _on_ok())
        combo.focus_set()

    def _apply_formula_change(self, idx: int, new_formula: str):
        """Применяет выбор новой формулы: обновляет result_df, таблицу, графики."""
        old_formula = self.result_df.at[idx, "brutto"]
        self.result_df.at[idx, "brutto"] = new_formula
        self._log(
            f"[INFO] Строка {idx}: формула изменена «{old_formula}» → «{new_formula}»",
            color=OK,
        )

        # Обновить отображение в Treeview
        row = self.result_df.iloc[idx]
        candidates = row.get("all_candidates", None)
        has_alternatives = isinstance(candidates, list) and len(candidates) > 1
        brutto_display = f"{new_formula}  ▾" if has_alternatives else new_formula

        vals = list(self.result_tree.item(str(idx), "values"))
        vals[1] = brutto_display  # индекс 1 = колонка «Формула»
        self.result_tree.item(str(idx), values=vals)

        # Перестроить Van Krevelen, если уже был построен
        if self._vk_figure is not None:
            try:
                self._plot_van_krevelen()
            except Exception:
                pass  # intentional: non-critical UI refresh

        self._refresh_structures_tab()

    def _refresh_structures_tab(self):
        """Обновить выпадающий список во вкладке Структуры."""
        if hasattr(self, "tab_struct") and self.tab_struct is not None:
            try:
                self.tab_struct._refresh_peak_list()
            except Exception:
                pass  # intentional: non-critical UI refresh

    def _sort_tree(self, col: str):
        if self.result_df is None:
            return
        if col not in self.result_df.columns:
            self._log(f"[WARN] Сортировка: колонка '{col}' не найдена", color=WARN)
            return
        ascending = getattr(self, f"_sort_{col}_asc", True)
        try:
            self.result_df = self.result_df.sort_values(
                col, ascending=ascending, na_position="last"
            )
        except Exception as e:
            self._log(f"[WARN] Сортировка по '{col}': {e}", color=WARN)
            return
        setattr(self, f"_sort_{col}_asc", not ascending)
        self._fill_result_table(self.result_df)

    # ── Импорт CSV ────────────────────────────────────────────────────

    def _import_csv(self):
        """Загрузить result_table.csv, заполнить таблицу и структуры."""
        path = filedialog.askopenfilename(
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if not path:
            return
        try:
            df = pd.read_csv(path, sep=";", encoding="utf-8-sig")
        except Exception:
            # пробуем другие разделители
            try:
                df = pd.read_csv(path, sep=",", encoding="utf-8")
            except Exception as e:
                messagebox.showerror("Ошибка импорта", str(e))
                return

        self.result_df = df
        self._log(f"[INFO] Импортировано: {path} ({len(df)} строк)", color=OK)
        self._set_status(f"Импортировано {len(df)} соединений.")
        self._fill_result_table(df)
        self._auto_plot_hist()
        self._refresh_structures_tab()

    # ── Импорт папки ──────────────────────────────────────────────────

    _SPECTRUM_PATTERNS = {
        "src": ["original", "src", "source", "исходный", "orig"],
        "dmet": ["deutermethyl", "dmet", "cd3", "дейтерометил"],
        "dacet": ["deuteroacyl", "dacet", "cd3co", "дейтероацил"],
    }

    def _apply_preset(self, presets: list):
        """Применить выбранный пресет параметров."""
        name = self.preset_var.get()
        preset = next((p for p in presets if p["name"] == name), None)
        if not preset:
            return
        params = preset.get("params", {})
        # Масс-фильтр
        for var, key in [(self.mass_min_var, "load_mass_min"), (self.mass_max_var, "load_mass_max")]:
            if key in params:
                var.set(str(params[key]))
        # Шумоподавление
        if "noise_intensity" in params:
            self.noise_int_var.set(str(params["noise_intensity"]))
        if "noise_force" in params:
            self.noise_force_var.set(str(params["noise_force"]))
        # Формулы
        if "rel_error" in params:
            self.rel_error_var.set(str(params["rel_error"]))
        if "ppm_tol" in params:
            self.ppm_tol_var.set(str(params["ppm_tol"]))
        if "max_groups" in params:
            self.max_groups_var.set(str(params["max_groups"]))
        # Диапазоны элементов
        er = params.get("element_ranges", {})
        for el, (var_min, var_max) in [("C", (self.c_min, self.c_max)), ("H", (self.h_min, self.h_max)),
                                         ("O", (self.o_min, self.o_max)), ("N", (self.n_min, self.n_max))]:
            if el in er:
                var_min.set(str(er[el][0]))
                var_max.set(str(er[el][1]))
        self._log(f"[PRESET] Применён: {name}", color="info")

    def _import_folder(self):
        """Автоопределение трёх спектров в папке по шаблонам имён."""
        folder = filedialog.askdirectory(title="Выберите папку со спектрами")
        if not folder:
            return

        import os, glob
        csv_files = glob.glob(os.path.join(folder, "*.csv"))
        raw_files = glob.glob(os.path.join(folder, "*.raw"))
        all_files = csv_files + raw_files
        if not all_files:
            messagebox.showwarning("Нет файлов", f"В папке нет .csv или .raw файлов: {folder}")
            return

        found = {"src": None, "dmet": None, "dacet": None}
        for f in all_files:
            name = os.path.basename(f).lower()
            for key, patterns in self._SPECTRUM_PATTERNS.items():
                if found[key] is None and any(p in name for p in patterns):
                    found[key] = f
                    break

        missing = [k for k, v in found.items() if v is None]
        if missing:
            # пробуем по порядку: первый — src, второй — dmet, третий — dacet
            all_files.sort()
            for key in missing:
                for f in all_files:
                    if f not in found.values():
                        found[key] = f
                        break

        self.src_var.set(found["src"] or "")
        self.dmet_var.set(found["dmet"] or "")
        self.dacet_var.set(found["dacet"] or "")
        if hasattr(self, '_folder_path_var'):
            self._folder_path_var.set(folder)

        found_count = sum(1 for v in found.values() if v)
        self._log(
            f"[INFO] Папка: {folder} → найдено {found_count}/3 спектров", color=OK)
        if found_count < 3:
            messagebox.showwarning(
                "Не все спектры",
                f"Автоматически найдено {found_count} из 3 спектров. Проверьте оставшиеся поля вручную."
            )

    def _export_csv(self):
        if self.result_df is None:
            messagebox.showinfo("Нет данных", "Сначала запустите анализ.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv", filetypes=[("CSV", "*.csv"), ("All", "*.*")]
        )
        if path:
            try:
                self.result_df.to_csv(path, index=False, sep=";", encoding="utf-8-sig")
                self._log(f"Таблица сохранена: {path}", color=OK)
            except Exception as e:
                self._log(f"[ОШИБКА] Сохранение не удалось: {e}", color=WARN)
                messagebox.showerror("Ошибка", str(e))

    # ── Van Krevelen ──────────────────────────────────────────────────────────

    def _plot_van_krevelen(self):
        if self.result_df is None or self.result_df.empty:
            messagebox.showinfo(
                "Нет данных",
                "Сначала запустите анализ, чтобы получить таблицу результатов.",
            )
            return
        if create_van_krevelen_plot is None:
            messagebox.showerror(
                "Ошибка", "Модуль Van Krevelen не загружен (core import failed)."
            )
            return

        self._log("[DEBUG] _plot_van_krevelen: построение диаграммы...", color="info")
        self._clear_frame(self.vk_canvas_frame)
        try:
            # Закрываем предыдущую фигуру, если она есть
            if self._vk_figure is not None:
                plt.close(self._vk_figure)

            fig = create_van_krevelen_plot(self.result_df, color_by=self.vk_color_var.get())
            self._vk_figure = fig
            embed_figure(fig, self.vk_canvas_frame)
            self._log("[DEBUG] _plot_van_krevelen: диаграмма построена", color=OK)
        except Exception:
            self._log(
                f"[ОШИБКА] _plot_van_krevelen:\n{traceback.format_exc()}",
                color=WARN,
            )
            plt.close("all")
            messagebox.showerror(
                "Ошибка", "Не удалось построить диаграмму Ван Кревелена."
            )

    def _save_van_krevelen_png(self):
        if self._vk_figure is None:
            messagebox.showinfo(
                "Нет диаграммы",
                "Сначала постройте диаграмму, нажав «Построить диаграмму Ван Кревелена».",
            )
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[
                ("PNG image", "*.png"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        try:
            self._vk_figure.savefig(path, dpi=300)
            self._log(f"Van Krevelen диаграмма сохранена: {path}", color=OK)
        except Exception as e:
            self._log(f"[ОШИБКА] Сохранение Van Krevelen PNG: {e}", color=WARN)
            messagebox.showerror("Ошибка", str(e))

    # ── Графики спектров ──────────────────────────────────────────────────────

    def _plot_spectra(self):
        paths = [self.src_var.get(), self.dmet_var.get(), self.dacet_var.get()]
        if not all(paths):
            messagebox.showwarning("Нет файлов", "Укажите все три файла.")
            return
        for p in paths:
            if not os.path.exists(p):
                messagebox.showerror("Файл не найден", p)
                return

        sep = self.sep_var.get()
        if sep in ("\\t", "tab", "TAB"):
            sep = "\t"

        self._log("[DEBUG] _plot_spectra: загрузка...", color="info")
        try:
            dfs = {}
            for key, path in zip(
                ["Исходный", "Дейтерометилирование", "Дейтероацилирование"], paths
            ):
                df = pd.read_csv(path, sep=sep)
                df.columns = [c.strip() for c in df.columns]
                # Единый маппинг из spectrum_ops
                from src.core.spectrum_ops import CSV_COLUMN_MAPPER

                df = df.rename(columns=CSV_COLUMN_MAPPER)
                if "mass" not in df.columns or "intensity" not in df.columns:
                    raise ValueError(
                        f"{key}: колонки mass/intensity не найдены. "
                        f"Доступны: {list(df.columns)}"
                    )
                dfs[key] = df
                self._log(f"[DEBUG]   {key}: {len(df)} строк", color="info")
        except Exception as e:
            self._log(f"[ОШИБКА] _plot_spectra: {traceback.format_exc()}", color=WARN)
            messagebox.showerror("Ошибка чтения", str(e))
            return

        self._clear_frame(self.spectra_canvas_frame)
        try:
            fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
            colors = [ACCENT, "#a6e3a1", "#fab387"]
            for ax, (title, df), color in zip(axes, dfs.items(), colors):
                ax.vlines(
                    df["mass"],
                    0,
                    df["intensity"],
                    colors=color,
                    linewidth=0.8,
                    alpha=0.8,
                )
                ax.set_ylabel("Intensity", fontsize=8)
                ax.set_title(title, fontsize=9, loc="left", color=FG)
                ax.grid(True, alpha=0.3)
            axes[-1].set_xlabel("m/z", fontsize=9)
            fig.suptitle("Три масс-спектра", color=ACCENT, fontsize=11)
            fig.tight_layout()
            embed_figure(fig, self.spectra_canvas_frame)
            self._log("[DEBUG] _plot_spectra: успешно", color=OK)
        except Exception:
            self._log(f"[ОШИБКА] _plot_spectра: {traceback.format_exc()}", color=WARN)
            plt.close("all")

    # ── Графики серий ─────────────────────────────────────────────────────────

    def _plot_series(self, which: str):
        if self.result_df is None:
            messagebox.showinfo("Нет данных", "Сначала запустите анализ.")
            return

        # build_result_table возвращает N_OH (не N_OH_total) — исправлено
        col_n = "N_COOH" if which == "dmet" else "N_OH"
        col_m = "missing_dmet" if which == "dmet" else "missing_dacet"
        delta = DELTA_CD3 if which == "dmet" else DELTA_CD3CO
        label = "CD₃ (dmet)" if which == "dmet" else "CD₃CO (dacet)"

        self._log(f"[DEBUG] _plot_series({which}): col_n={col_n}", color="info")

        if col_n not in self.result_df.columns:
            self._log(
                f"[WARN] '{col_n}' нет в result_df. "
                f"Есть: {list(self.result_df.columns)}",
                color=WARN,
            )
            messagebox.showwarning("Нет данных", f"Колонка '{col_n}' отсутствует.")
            return

        df = _safe_df(self.result_df)[self.result_df[col_n] > 0].copy()
        self._log(f"[DEBUG] Соединений с {col_n}>0: {len(df)}", color="info")
        if df.empty:
            self._log(f"Серии {label}: нет соединений с n>0.", color=WARN)
            return

        n_plots = min(len(df), 9)
        ncols = 3
        nrows = (n_plots + ncols - 1) // ncols

        self._clear_frame(self.series_canvas_frame)
        try:
            fig, axes = plt.subplots(nrows, ncols, figsize=(9, nrows * 2.8))
            if nrows * ncols == 1:
                axes_flat = [axes]
            elif nrows == 1:
                axes_flat = list(axes)
            else:
                axes_flat = list(axes.flatten())

            last_i = -1
            for last_i, (_, row) in enumerate(df.head(n_plots).iterrows()):
                ax = axes_flat[last_i]
                m0 = row["mass"]
                n = int(row[col_n])
                steps = list(range(1, n + 1))

                missing = row.get(col_m, [])
                if isinstance(missing, str):
                    try:
                        missing = ast.literal_eval(missing)
                    except Exception:
                        missing = []
                if not isinstance(missing, list):
                    missing = []

                colors_bars = [WARN if s in missing else OK for s in steps]
                ax.bar(steps, [1] * len(steps), color=colors_bars, alpha=0.8, width=0.6)
                ax.set_xticks(steps)
                ax.set_xticklabels([str(s) for s in steps], fontsize=7)
                ax.set_yticks([])
                ax.set_title(
                    f"m/z={m0:.3f}\n{row.get('brutto','')}, n={n}", fontsize=7, color=FG
                )
                if missing:
                    ax.set_xlabel(f"⚠ пропуски: {missing}", fontsize=6, color=WARN)

            for j in range(last_i + 1, len(axes_flat)):
                axes_flat[j].set_visible(False)

            fig.suptitle(
                f"Серии {label}  (зелёный=найден, красный=пропущен)",
                color=ACCENT,
                fontsize=10,
            )
            fig.tight_layout()
            embed_figure(fig, self.series_canvas_frame)
            self._log(f"[DEBUG] _plot_series: {n_plots} графиков построено", color=OK)
        except Exception:
            self._log(f"[ОШИБКА] _plot_series: {traceback.format_exc()}", color=WARN)
            plt.close("all")

    # ── Гистограммы ───────────────────────────────────────────────────────────

    def _auto_plot_hist(self):
        if self.result_df is None:
            return
        self._log("[DEBUG] _auto_plot_hist", color="info")
        self._clear_frame(self.hist_frame)
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 2.5))
            for ax, col, color in [
                (ax1, "N_COOH", "#f38ba8"),
                (ax2, "N_OH", "#a6e3a1"),
            ]:
                if col not in self.result_df.columns:
                    self._log(f"[WARN] _auto_plot_hist: нет '{col}'", color=WARN)
                    continue
                vals = _safe_df(self.result_df)[col].dropna().astype(int)
                if not vals.empty:
                    ax.hist(
                        vals,
                        bins=range(vals.max() + 2),
                        color=color,
                        alpha=0.85,
                        edgecolor=BG,
                        rwidth=0.7,
                    )
                ax.set_xlabel(col, fontsize=8)
                ax.set_ylabel("Кол-во", fontsize=8)
                ax.grid(True, alpha=0.3)
            fig.tight_layout()
            embed_figure(fig, self.hist_frame, toolbar=False)
        except Exception:
            messagebox.showwarning(
                "Ошибка построения гистограмм",
                "Не удалось построить гистограммы функциональных групп.\n"
                "Проверьте данные в таблице результатов."
            )
            self._log(f"[ОШИБКА] _auto_plot_hist: {traceback.format_exc()}", color=WARN)
            plt.close("all")

    def _plot_hist(self, col: str):
        if self.result_df is None:
            messagebox.showinfo("Нет данных", "Сначала запустите анализ.")
            return
        if col not in self.result_df.columns:
            self._log(
                f"[WARN] _plot_hist: нет '{col}'. "
                f"Есть: {list(self.result_df.columns)}",
                color=WARN,
            )
            messagebox.showwarning("Нет данных", f"Колонка '{col}' отсутствует.")
            return
        self._clear_frame(self.series_canvas_frame)
        try:
            fig, ax = plt.subplots(figsize=(7, 4))
            vals = _safe_df(self.result_df)[col].dropna().astype(int)
            if vals.empty:
                ax.text(
                    0.5,
                    0.5,
                    "Нет данных",
                    transform=ax.transAxes,
                    ha="center",
                    color=FG,
                )
            else:
                ax.hist(
                    vals,
                    bins=range(vals.max() + 2),
                    color=ACCENT,
                    alpha=0.85,
                    edgecolor=BG,
                    rwidth=0.7,
                )
            ax.set_xlabel(col)
            ax.set_ylabel("Количество соединений")
            ax.set_title(f"Распределение {col}")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            embed_figure(fig, self.series_canvas_frame)
        except Exception:
            self._log(
                f"[ОШИБКА] _plot_hist({col}): {traceback.format_exc()}", color=WARN
            )
            plt.close("all")


# ═══════════════════════════════════════════════════════════════════════════════
#  ТОЧКА ВХОДА
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    """Entry point for ``nom-hrms-fga`` CLI / ``python -m src``."""
    warnings.filterwarnings("always")
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
