"""
app.py  —  GUI-интерфейс для пайплайна определения -COOH и -OH групп
           Запускать: python app.py
           Требует: tkinter (стандартная библиотека Python), matplotlib, pandas
"""
from __future__ import annotations

import ast
import io
import os
import queue
import threading
import traceback
import warnings
from pathlib import Path
from typing import Optional
from rdkit import *

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

# ── Импорт UI-утилит ─────────────────────────────────────────────────────────
try:
    from src.ui.plots import embed_figure
    from src.ui.theme import (
        ACCENT, BG, FG, IMG_H, IMG_W, MONO, OK, PANEL, WARN,
        _mpl_style, _style
    )
    from src.structures.tab import StructureViewerTab
    _UI_LOADED = True
    _UI_ERROR = ""
except Exception as _ui_err:
    _UI_LOADED = False
    _UI_ERROR = traceback.format_exc()
    # Fallback-константы, чтобы GUI хотя бы запустился без src.ui
    BG = "#1e1e2e"; ACCENT = "#cba6f7"; PANEL = "#313244"
    WARN = "#f38ba8"; FG = "#cdd6f4"; OK = "#a6e3a1"
    MONO = ("Consolas", 9)
    IMG_H = 400; IMG_W = 800
    def _mpl_style(): pass
    def _style(root): pass
    StructureViewerTab = None

    def embed_figure(app, fig, frame, toolbar=True):
        """Минимальный fallback через FigureCanvasTkAgg."""
        try:
            from matplotlib.backends.backend_tkagg import (
                FigureCanvasTkAgg, NavigationToolbar2Tk)
            canvas = FigureCanvasTkAgg(fig, master=frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)
            if toolbar:
                NavigationToolbar2Tk(canvas, frame)
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
    # Заглушки, чтобы имена не были undefined при старте
    DELTA_CD3 = 17.034
    DELTA_CD3CO = 44.028
    run_pipeline = load_spectrum = find_series = visualize_series = None


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
        if self._orig and hasattr(self._orig, 'fileno'):
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
        self.title("MS Functional Groups Analyzer")
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
        self.src_spec   = None
        self.dmet_spec  = None
        self.dacet_spec = None
        self.df_dmet_series  = None
        self.df_dacet_series = None

        # ── очередь для потоко-безопасного логирования ──
        self._log_queue: queue.Queue = queue.Queue()

        # ── файловые переменные ──
        self.src_var   = tk.StringVar()
        self.dmet_var  = tk.StringVar()
        self.dacet_var = tk.StringVar()

        # ── параметры (значения совпадают с тестовыми дефолтами pipeline) ──
        self.sep_var         = tk.StringVar(value=",")
        self.mass_min_var    = tk.StringVar(value="0.0")
        self.mass_max_var    = tk.StringVar(value="1000.0")
        self.noise_force_var = tk.StringVar(value="10")
        self.noise_int_var   = tk.StringVar(value="100")
        self.rel_error_var   = tk.StringVar(value="0.5")
        self.sign_var        = tk.StringVar(value="-")
        self.ppm_tol_var     = tk.StringVar(value="0.5")
        self.max_groups_var  = tk.StringVar(value="20")
        self.allow_gaps_var  = tk.BooleanVar(value=True)
        self.output_csv_var  = tk.StringVar(value="result_table.csv")
        self.c_min = tk.StringVar(value="4");  self.c_max = tk.StringVar(value="50")
        self.h_min = tk.StringVar(value="4");  self.h_max = tk.StringVar(value="100")
        self.o_min = tk.StringVar(value="0");  self.o_max = tk.StringVar(value="25")
        self.n_min = tk.StringVar(value="0");  self.n_max = tk.StringVar(value="2")

        self._build_ui()

        # Опрос очереди стартует после построения UI
        self._poll_log_queue()

        # Отложенные предупреждения об ошибках импорта
        if not CORE_LOADED:
            self._log(f"[ОШИБКА] src.core не загружен:\n{_CORE_ERROR}", color=WARN)
        if not _UI_LOADED:
            self._log(f"[WARN] src.ui / src.structures не загружены:\n{_UI_ERROR}", color=WARN)

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
            self.after(50, self._poll_log_queue)

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
            defaultextension=".txt",
            filetypes=[("Text", "*.txt"), ("All", "*.*")]
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
        hdr = tk.Label(self, text="⚗  MS Functional Groups Analyzer",
                       bg=BG, fg=ACCENT, font=("Segoe UI", 16, "bold"))
        hdr.pack(fill="x", padx=16, pady=(12, 4))
        ttk.Separator(self, orient="horizontal").pack(fill="x", padx=16, pady=2)

        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=16, pady=8)

        self.tab_params  = ttk.Frame(nb)
        self.tab_spectra = ttk.Frame(nb)
        self.tab_series  = ttk.Frame(nb)
        self.tab_result  = ttk.Frame(nb)
        self.tab_log     = ttk.Frame(nb)

        nb.add(self.tab_params,  text="⚙  Параметры")
        nb.add(self.tab_spectra, text="📈  Спектры")
        nb.add(self.tab_series,  text="🔗  Серии")
        nb.add(self.tab_result,  text="📊  Результаты")
        nb.add(self.tab_log,     text="📋  Лог")

        if StructureViewerTab is not None:
            try:
                self.tab_struct = StructureViewerTab(nb, app=self)
                nb.add(self.tab_struct, text="🧪  Структуры")
            except Exception as e:
                self._log_queue.put(("log", f"[WARN] StructureViewerTab init failed: {e}\n"))

        self._build_params_tab()
        self._build_spectra_tab()
        self._build_series_tab()
        self._build_result_tab()
        self._build_log_tab()

        self.status_var = tk.StringVar(value="Готов к работе")
        tk.Label(self, textvariable=self.status_var,
                 bg=PANEL, fg=FG, font=("Segoe UI", 9),
                 anchor="w", padx=8).pack(fill="x", side="bottom")
        self.progress = ttk.Progressbar(self, mode="indeterminate", length=200)
        self.progress.pack(fill="x", side="bottom")

    # ── ВКЛАДКА ПАРАМЕТРОВ ────────────────────────────────────────────────────

    def _build_params_tab(self):
        p = self.tab_params
        p.columnconfigure(0, weight=1)
        p.columnconfigure(1, weight=1)

        files_lf = ttk.LabelFrame(p, text="📂  Входные файлы")
        files_lf.grid(row=0, column=0, columnspan=2, sticky="ew", padx=8, pady=6)
        files_lf.columnconfigure(1, weight=1)
        for i, (label, var) in enumerate([
            ("Исходный спектр:",     self.src_var),
            ("Дейтерометилирование:", self.dmet_var),
            ("Дейтероацилирование:",  self.dacet_var),
        ]):
            ttk.Label(files_lf, text=label).grid(row=i, column=0, sticky="w", padx=6, pady=3)
            ttk.Entry(files_lf, textvariable=var, width=55).grid(
                row=i, column=1, sticky="ew", padx=4, pady=3)
            ttk.Button(files_lf, text="…",
                       command=lambda v=var: self._browse(v)).grid(
                row=i, column=2, padx=4, pady=3)

        load_lf = ttk.LabelFrame(p, text="📥  Загрузка и диапазон масс")
        load_lf.grid(row=1, column=0, sticky="ew", padx=8, pady=6)
        for row, (lbl, var) in enumerate([
            ("Разделитель CSV:",  self.sep_var),
            ("m/z min:",         self.mass_min_var),
            ("m/z max:",         self.mass_max_var),
            ("Шум (force):",     self.noise_force_var),
            ("Шум (intensity):", self.noise_int_var),
        ]):
            ttk.Label(load_lf, text=lbl).grid(row=row, column=0, sticky="w", padx=6, pady=3)
            ttk.Entry(load_lf, textvariable=var, width=12).grid(
                row=row, column=1, sticky="w", padx=4, pady=3)

        form_lf = ttk.LabelFrame(p, text="🔬  Назначение брутто-формул")
        form_lf.grid(row=1, column=1, sticky="ew", padx=8, pady=6)
        ttk.Label(form_lf, text="Знак иона:").grid(row=0, column=0, sticky="w", padx=6, pady=3)
        ttk.Combobox(form_lf, textvariable=self.sign_var,
                     values=["-", "+", "0"], width=5,
                     state="readonly").grid(row=0, column=1, sticky="w", padx=4, pady=3)
        ttk.Label(form_lf, text="Погрешность (ppm):").grid(row=1, column=0, sticky="w", padx=6, pady=3)
        ttk.Entry(form_lf, textvariable=self.rel_error_var, width=8).grid(
            row=1, column=1, sticky="w", padx=4, pady=3)
        ttk.Label(form_lf, text="Диапазоны элементов:").grid(
            row=2, column=0, columnspan=2, sticky="w", padx=6, pady=(8, 2))
        for i, (sym, mn, mx) in enumerate([
            ("C", self.c_min, self.c_max), ("H", self.h_min, self.h_max),
            ("O", self.o_min, self.o_max), ("N", self.n_min, self.n_max),
        ]):
            r = 3 + i
            ttk.Label(form_lf, text=f"{sym}:").grid(row=r, column=0, sticky="w", padx=20, pady=2)
            ef = ttk.Frame(form_lf)
            ef.grid(row=r, column=1, sticky="w", padx=4, pady=2)
            ttk.Entry(ef, textvariable=mn, width=5).pack(side="left")
            ttk.Label(ef, text="–").pack(side="left", padx=2)
            ttk.Entry(ef, textvariable=mx, width=5).pack(side="left")

        ser_lf = ttk.LabelFrame(p, text="🔍  Поиск серий")
        ser_lf.grid(row=2, column=0, sticky="ew", padx=8, pady=6)
        ttk.Label(ser_lf, text="Допуск поиска (ppm):").grid(row=0, column=0, sticky="w", padx=6, pady=3)
        ttk.Entry(ser_lf, textvariable=self.ppm_tol_var, width=8).grid(
            row=0, column=1, sticky="w", padx=4, pady=3)
        ttk.Label(ser_lf, text="Макс. групп:").grid(row=1, column=0, sticky="w", padx=6, pady=3)
        ttk.Entry(ser_lf, textvariable=self.max_groups_var, width=8).grid(
            row=1, column=1, sticky="w", padx=4, pady=3)
        ttk.Checkbutton(ser_lf, text="Разрешить пропуски в сериях",
                        variable=self.allow_gaps_var).grid(
            row=2, column=0, columnspan=2, sticky="w", padx=6, pady=4)

        out_lf = ttk.LabelFrame(p, text="💾  Выходной файл")
        out_lf.grid(row=2, column=1, sticky="ew", padx=8, pady=6)
        out_lf.columnconfigure(0, weight=1)
        ttk.Entry(out_lf, textvariable=self.output_csv_var, width=35).grid(
            row=0, column=0, sticky="ew", padx=6, pady=4)
        ttk.Button(out_lf, text="…",
                   command=lambda: self._save_browse(self.output_csv_var)).grid(
            row=0, column=1, padx=4, pady=4)

        try:
            run_btn = ttk.Button(p, text="▶  Запустить анализ",
                                 style="Accent.TButton", command=self._run)
        except Exception:
            run_btn = ttk.Button(p, text="▶  Запустить анализ", command=self._run)
        run_btn.grid(row=3, column=0, columnspan=2, pady=12, ipadx=20, ipady=4)

    # ── ВКЛАДКА СПЕКТРЫ ───────────────────────────────────────────────────────

    def _build_spectra_tab(self):
        frame = self.tab_spectra
        ctrl = ttk.Frame(frame)
        ctrl.pack(fill="x", pady=4, padx=8)
        ttk.Button(ctrl, text="📈 Построить спектры",
                   command=self._plot_spectra).pack(side="left", padx=4)
        ttk.Button(ctrl, text="🗑 Очистить",
                   command=lambda: self._clear_frame(self.spectra_canvas_frame)).pack(
            side="left", padx=4)
        self.spectra_canvas_frame = ttk.Frame(frame)
        self.spectra_canvas_frame.pack(fill="both", expand=True)

    # ── ВКЛАДКА СЕРИИ ─────────────────────────────────────────────────────────

    def _build_series_tab(self):
        frame = self.tab_series
        ctrl = ttk.Frame(frame)
        ctrl.pack(fill="x", pady=4, padx=8)
        ttk.Button(ctrl, text="🔗 Показать серии CD₃",
                   command=lambda: self._plot_series("dmet")).pack(side="left", padx=4)
        ttk.Button(ctrl, text="🔗 Показать серии CD₃CO",
                   command=lambda: self._plot_series("dacet")).pack(side="left", padx=4)
        ttk.Button(ctrl, text="🗑 Очистить",
                   command=lambda: self._clear_frame(self.series_canvas_frame)).pack(
            side="left", padx=4)
        self.series_canvas_frame = ttk.Frame(frame)
        self.series_canvas_frame.pack(fill="both", expand=True)

    # ── ВКЛАДКА РЕЗУЛЬТАТОВ ───────────────────────────────────────────────────

    def _build_result_tab(self):
        frame = self.tab_result
        ctrl = ttk.Frame(frame)
        ctrl.pack(fill="x", pady=4, padx=8)
        ttk.Button(ctrl, text="📊 Гистограмма N_COOH",
                   command=lambda: self._plot_hist("N_COOH")).pack(side="left", padx=4)
        ttk.Button(ctrl, text="📊 Гистограмма N_OH",
                   command=lambda: self._plot_hist("N_OH")).pack(side="left", padx=4)
        ttk.Button(ctrl, text="💾 Экспорт CSV",
                   command=self._export_csv).pack(side="left", padx=4)

        tbl_frame = ttk.Frame(frame)
        tbl_frame.pack(fill="both", expand=True, padx=8, pady=4)

        cols       = ("mass", "brutto", "N_COOH", "N_OH", "missing_dmet", "missing_dacet")
        col_labels = ["m/z", "Формула", "N_COOH", "N_OH", "Пропуски dmet", "Пропуски dacet"]
        col_widths = [110, 140, 80, 80, 150, 150]

        self.result_tree = ttk.Treeview(tbl_frame, columns=cols, show="headings", height=18)
        for c, lbl, w in zip(cols, col_labels, col_widths):
            self.result_tree.heading(c, text=lbl,
                                     command=lambda _c=c: self._sort_tree(_c))
            self.result_tree.column(c, width=w, anchor="center")

        vsb = ttk.Scrollbar(tbl_frame, orient="vertical",   command=self.result_tree.yview)
        hsb = ttk.Scrollbar(tbl_frame, orient="horizontal", command=self.result_tree.xview)
        self.result_tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        vsb.pack(side="right",  fill="y")
        hsb.pack(side="bottom", fill="x")
        self.result_tree.pack(fill="both", expand=True)

        self.hist_frame = ttk.Frame(frame)
        self.hist_frame.pack(fill="x", padx=8, pady=4)

    # ── ВКЛАДКА ЛОГ ──────────────────────────────────────────────────────────

    def _build_log_tab(self):
        frame = self.tab_log
        ctrl = ttk.Frame(frame)
        ctrl.pack(fill="x", pady=4, padx=8)
        ttk.Button(ctrl, text="🗑 Очистить лог",  command=self._clear_log).pack(side="left", padx=4)
        ttk.Button(ctrl, text="💾 Сохранить лог", command=self._save_log).pack(side="left", padx=4)

        self.log_text = scrolledtext.ScrolledText(
            frame, bg=PANEL, fg=FG, font=MONO,
            relief="flat", insertbackground=FG, state="disabled",
        )
        self.log_text.pack(fill="both", expand=True, padx=8, pady=4)
        self.log_text.tag_config("ok",   foreground=OK)
        self.log_text.tag_config("warn", foreground=WARN)
        self.log_text.tag_config("info", foreground=ACCENT)

    # ═══════════════════════════════════════════════════════════════════════════
    #  ДЕЙСТВИЯ
    # ═══════════════════════════════════════════════════════════════════════════

    def _browse(self, var: tk.StringVar):
        path = filedialog.askopenfilename(
            filetypes=[("CSV files", "*.csv"), ("Text files", "*.txt"), ("All files", "*.*")]
        )
        if path:
            var.set(path)

    def _save_browse(self, var: tk.StringVar):
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if path:
            var.set(path)

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

        mass_min    = _float(self.mass_min_var,    "m/z min",            0.0)
        mass_max    = _float(self.mass_max_var,    "m/z max",            1000.0)
        noise_force = _float(self.noise_force_var, "Шум force",          10.0)
        noise_int   = _float(self.noise_int_var,   "Шум intensity",      100.0)
        rel_error   = _float(self.rel_error_var,   "Погрешность ppm",    0.5)
        ppm_tol     = _float(self.ppm_tol_var,     "Допуск поиска ppm",  0.5)
        max_groups  = _int(self.max_groups_var,    "Макс. групп",        20)

        try:
            c_min = int(self.c_min.get()); c_max = int(self.c_max.get())
            h_min = int(self.h_min.get()); h_max = int(self.h_max.get())
            o_min = int(self.o_min.get()); o_max = int(self.o_max.get())
            n_min = int(self.n_min.get()); n_max = int(self.n_max.get())
        except ValueError as e:
            errors.append(f"  • Диапазон элементов: {e}")
            c_min=4; c_max=50; h_min=4; h_max=100; o_min=0; o_max=25; n_min=0; n_max=2

        if mass_min is not None and mass_max is not None and mass_min >= mass_max:
            errors.append(f"  • m/z min ({mass_min}) ≥ m/z max ({mass_max})")

        if errors:
            msg = "Ошибки в параметрах:\n" + "\n".join(errors)
            self._log("[DEBUG] Ошибки валидации:\n" + msg, color=WARN)
            messagebox.showerror("Ошибки параметров", msg)
            return None

        brutto_dict = {
            "C": (c_min, c_max), "H": (h_min, h_max),
            "O": (o_min, o_max), "N": (n_min, n_max),
        }

        return dict(
            sep=sep,
            load_mass_min=mass_min,
            load_mass_max=mass_max,
            noise_force=noise_force,
            noise_intensity=noise_int,
            rel_error=rel_error,
            sign=self.sign_var.get(),
            assign_mass_min=0,
            assign_mass_max=1000,
            ppm_tol=ppm_tol,
            max_groups=max_groups,
            allow_gaps=self.allow_gaps_var.get(),
            brutto_dict=brutto_dict,
            output_csv=self.output_csv_var.get() or None,
            visualize=False,   # визуализацию делаем через GUI-вкладку
        )

    # ── Запуск ────────────────────────────────────────────────────────────────

    def _run(self):
        if not CORE_LOADED:
            messagebox.showerror("Ошибка", f"src.core не загружен:\n{_CORE_ERROR[:800]}")
            return

        for label, var in [("Исходный", self.src_var),
                           ("Дейтерометилирование", self.dmet_var),
                           ("Дейтероацилирование",  self.dacet_var)]:
            path = var.get()
            if not path:
                messagebox.showwarning("Файл не выбран", f"Укажите файл: «{label}»")
                return
            if not os.path.exists(path):
                messagebox.showwarning("Файл не найден",
                                       f"Файл не существует: «{label}»\n{path}")
                return

        params = self._parse_params()
        if params is None:
            return

        self._clear_log()
        self._log("[DEBUG] ═══ Запуск анализа ═══", color="info")
        self._log(f"[DEBUG]   src   = {self.src_var.get()}", color="info")
        self._log(f"[DEBUG]   dmet  = {self.dmet_var.get()}", color="info")
        self._log(f"[DEBUG]   dacet = {self.dacet_var.get()}", color="info")
        self._log(f"[DEBUG]   params = {params}", color="info")

        self.progress.start(10)
        self._set_status("Выполняется анализ…")

        t = threading.Thread(
            target=self._run_worker,
            args=(self.src_var.get(), self.dmet_var.get(),
                  self.dacet_var.get(), params),
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
            self._log_queue.put(("log", f"[DEBUG] _run_worker: pipeline завершён, строк={n}\n"))
            self._log_queue.put(("success", {"result": table, "stats": getattr(res, "stats", None)}))
        except Exception:
            tb = traceback.format_exc()
            self._log_queue.put(("log", f"[DEBUG] _run_worker: EXCEPTION\n{tb}\n"))
            self._log_queue.put(("error", {"traceback": tb}))
        finally:
            sys.stdout = orig_stdout
            sys.stderr = orig_stderr

    # ── Таблица результатов ───────────────────────────────────────────────────

    def _fill_result_table(self, df: pd.DataFrame):
        self._log(f"[DEBUG] _fill_result_table: {len(df)} строк, "
                  f"колонки={list(df.columns)}", color="info")
        for row in self.result_tree.get_children():
            self.result_tree.delete(row)

        for col, fill in [("N_COOH", 0), ("N_OH", 0),
                          ("missing_dmet", []), ("missing_dacet", []),
                          ("brutto", "")]:
            if col not in df.columns:
                df[col] = fill
                self._log(f"[WARN] Колонка '{col}' отсутствует → заполнено {fill!r}", color=WARN)

        warn_count = 0
        for _, r in df.iterrows():
            try:
                n_cooh = int(r.get("N_COOH", 0))
                n_oh   = int(r.get("N_OH",   0))
            except (ValueError, TypeError):
                n_cooh = 0; n_oh = 0

            missing_d = r.get("missing_dmet",  [])
            missing_a = r.get("missing_dacet", [])
            has_missing = bool(missing_d) or bool(missing_a)

            vals = (
                f"{r['mass']:.5f}" if pd.notna(r.get("mass")) else "?",
                r.get("brutto", ""),
                n_cooh, n_oh,
                str(missing_d), str(missing_a),
            )
            tag = "warn" if has_missing else ""
            if has_missing:
                warn_count += 1
            self.result_tree.insert("", "end", values=vals, tags=(tag,))

        self.result_tree.tag_configure("warn", foreground=WARN)
        self._log(f"[DEBUG] Таблица: {len(df)} строк, {warn_count} с пропусками.", color="info")

    def _sort_tree(self, col: str):
        if self.result_df is None:
            return
        if col not in self.result_df.columns:
            self._log(f"[WARN] Сортировка: колонка '{col}' не найдена", color=WARN)
            return
        ascending = getattr(self, f"_sort_{col}_asc", True)
        try:
            self.result_df = self.result_df.sort_values(
                col, ascending=ascending, na_position="last")
        except Exception as e:
            self._log(f"[WARN] Сортировка по '{col}': {e}", color=WARN)
            return
        setattr(self, f"_sort_{col}_asc", not ascending)
        self._fill_result_table(self.result_df)

    def _export_csv(self):
        if self.result_df is None:
            messagebox.showinfo("Нет данных", "Сначала запустите анализ.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("All", "*.*")]
        )
        if path:
            try:
                self.result_df.to_csv(path, index=False, sep=";", encoding="utf-8-sig")
                self._log(f"Таблица сохранена: {path}", color=OK)
            except Exception as e:
                self._log(f"[ОШИБКА] Сохранение не удалось: {e}", color=WARN)
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
                    ["Исходный", "Дейтерометилирование", "Дейтероацилирование"], paths):
                df = pd.read_csv(path, sep=sep)
                df.columns = [c.strip() for c in df.columns]
                col_map = {}
                for c in df.columns:
                    lc = c.lower()
                    if lc in ("m/z", "mz", "mass"):
                        col_map[c] = "mass"
                    elif lc in ("i", "intensity", "int"):
                        col_map[c] = "intensity"
                if col_map:
                    df = df.rename(columns=col_map)
                if "mass" not in df.columns or "intensity" not in df.columns:
                    raise ValueError(
                        f"{key}: колонки mass/intensity не найдены. "
                        f"Доступны: {list(df.columns)}")
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
                ax.vlines(df["mass"], 0, df["intensity"],
                          colors=color, linewidth=0.8, alpha=0.8)
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
            self._log(f"[WARN] '{col_n}' нет в result_df. "
                      f"Есть: {list(self.result_df.columns)}", color=WARN)
            messagebox.showwarning("Нет данных",
                                   f"Колонка '{col_n}' отсутствует.")
            return

        df = self.result_df[self.result_df[col_n] > 0].copy()
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
                n  = int(row[col_n])
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
                ax.bar(steps, [1] * len(steps),
                       color=colors_bars, alpha=0.8, width=0.6)
                ax.set_xticks(steps)
                ax.set_xticklabels([str(s) for s in steps], fontsize=7)
                ax.set_yticks([])
                ax.set_title(f"m/z={m0:.3f}\n{row.get('brutto','')}, n={n}",
                             fontsize=7, color=FG)
                if missing:
                    ax.set_xlabel(f"⚠ пропуски: {missing}", fontsize=6, color=WARN)

            for j in range(last_i + 1, len(axes_flat)):
                axes_flat[j].set_visible(False)

            fig.suptitle(f"Серии {label}  (зелёный=найден, красный=пропущен)",
                         color=ACCENT, fontsize=10)
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
            for ax, col, color in [(ax1, "N_COOH", "#f38ba8"),
                                   (ax2, "N_OH", "#a6e3a1")]:
                if col not in self.result_df.columns:
                    self._log(f"[WARN] _auto_plot_hist: нет '{col}'", color=WARN)
                    continue
                vals = self.result_df[col].dropna().astype(int)
                if not vals.empty:
                    ax.hist(vals, bins=range(vals.max() + 2),
                            color=color, alpha=0.85, edgecolor=BG, rwidth=0.7)
                ax.set_xlabel(col, fontsize=8)
                ax.set_ylabel("Кол-во", fontsize=8)
                ax.grid(True, alpha=0.3)
            fig.tight_layout()
            embed_figure( fig, self.hist_frame, toolbar=False)
        except Exception:
            self._log(f"[ОШИБКА] _auto_plot_hist: {traceback.format_exc()}", color=WARN)
            plt.close("all")

    def _plot_hist(self, col: str):
        if self.result_df is None:
            messagebox.showinfo("Нет данных", "Сначала запустите анализ.")
            return
        if col not in self.result_df.columns:
            self._log(f"[WARN] _plot_hist: нет '{col}'. "
                      f"Есть: {list(self.result_df.columns)}", color=WARN)
            messagebox.showwarning("Нет данных", f"Колонка '{col}' отсутствует.")
            return
        self._clear_frame(self.series_canvas_frame)
        try:
            fig, ax = plt.subplots(figsize=(7, 4))
            vals = self.result_df[col].dropna().astype(int)
            if vals.empty:
                ax.text(0.5, 0.5, "Нет данных",
                        transform=ax.transAxes, ha="center", color=FG)
            else:
                ax.hist(vals, bins=range(vals.max() + 2),
                        color=ACCENT, alpha=0.85, edgecolor=BG, rwidth=0.7)
            ax.set_xlabel(col)
            ax.set_ylabel("Количество соединений")
            ax.set_title(f"Распределение {col}")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            embed_figure( fig, self.series_canvas_frame)
        except Exception:
            self._log(f"[ОШИБКА] _plot_hist({col}): {traceback.format_exc()}", color=WARN)
            plt.close("all")


# ═══════════════════════════════════════════════════════════════════════════════
#  ТОЧКА ВХОДА
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    warnings.filterwarnings("always")
    app = App()
    app.mainloop()