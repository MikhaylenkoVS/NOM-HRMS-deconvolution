"""
app.py  —  GUI-интерфейс для пайплайна определения -COOH и -OH групп
           Запускать: python app.py
           Требует: tkinter (стандартная библиотека Python), matplotlib, pandas
"""

import os
import sys
import threading
import traceback
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import structures
import core
import ui
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from .ui.plots import embed_figure, clear_canvas
from .structures.tab import StructureViewerTab
# ── Импорт пайплайна из core.py ──────────────────────────────────────────────
try:
    from core import (
        run_pipeline,
        load_spectrum,
        find_series,
        visualize_series,
        DELTA_CD3,
        DELTA_CD3CO,
    )
    CORE_LOADED = True
except Exception as _core_err:
    CORE_LOADED = False
    _CORE_ERROR = str(_core_err)

# ═══════════════════════════════════════════════════════════════════════════════
#  ГЛАВНОЕ ОКНО
# ═══════════════════════════════════════════════════════════════════════════════

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("MS Functional Groups Analyzer")
        self.geometry("1200x760")
        self.configure(bg=BG)
        self.resizable(True, True)
        _style(self)
        _mpl_style()

        # ── данные ──
        self.result_df: pd.DataFrame | None = None
        self.src_spec   = None
        self.dmet_spec  = None
        self.dacet_spec = None
        self.df_dmet_series  = None
        self.df_dacet_series = None

        # ── файловые переменные ──
        self.src_var   = tk.StringVar()
        self.dmet_var  = tk.StringVar()
        self.dacet_var = tk.StringVar()

        # ── параметры ──
        self.sep_var         = tk.StringVar(value=",")
        self.mass_min_var    = tk.StringVar(value="200.0")
        self.mass_max_var    = tk.StringVar(value="700.0")
        self.noise_var       = tk.StringVar(value="1.5")
        self.rel_error_var   = tk.StringVar(value="0.5")
        self.sign_var        = tk.StringVar(value="-")
        self.ppm_tol_var     = tk.StringVar(value="5.0")
        self.max_groups_var  = tk.StringVar(value="20")
        self.allow_gaps_var  = tk.BooleanVar(value=True)
        self.output_csv_var  = tk.StringVar(value="result_table.csv")
        self.c_min = tk.StringVar(value="4");  self.c_max = tk.StringVar(value="50")
        self.h_min = tk.StringVar(value="4");  self.h_max = tk.StringVar(value="100")
        self.o_min = tk.StringVar(value="0");  self.o_max = tk.StringVar(value="25")
        self.n_min = tk.StringVar(value="0");  self.n_max = tk.StringVar(value="2")

        self._build_ui()

        if not CORE_LOADED:
            self._log(f"[ОШИБКА] Не удалось импортировать core.py:\n{_CORE_ERROR}",
                      color=WARN)

    # ── ПОСТРОЕНИЕ ИНТЕРФЕЙСА ─────────────────────────────────────────────────

    def _build_ui(self):
        # Заголовок
        hdr = tk.Label(self, text="⚗  MS Functional Groups Analyzer",
                       bg=BG, fg=ACCENT,
                       font=("Segoe UI", 16, "bold"))
        hdr.pack(fill="x", padx=16, pady=(12, 4))

        ttk.Separator(self, orient="horizontal").pack(fill="x", padx=16, pady=2)

        # Вкладки
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
        self.tab_struct = StructureViewerTab(nb, app=self)
        nb.add(self.tab_struct, text="🧪  Структуры")
        self._build_params_tab()
        self._build_spectra_tab()
        self._build_series_tab()
        self._build_result_tab()
        self._build_log_tab()

        # Статусная строка
        self.status_var = tk.StringVar(value="Готов к работе")
        status_bar = tk.Label(self, textvariable=self.status_var,
                              bg=PANEL, fg=FG,
                              font=("Segoe UI", 9), anchor="w", padx=8)
        status_bar.pack(fill="x", side="bottom")
        self.progress = ttk.Progressbar(self, mode="indeterminate", length=200)
        self.progress.pack(fill="x", side="bottom")

    # ── ВКЛАДКА ПАРАМЕТРОВ ────────────────────────────────────────────────────

    def _build_params_tab(self):
        p = self.tab_params
        p.columnconfigure(0, weight=1)
        p.columnconfigure(1, weight=1)

        # --- Файлы ---
        files_lf = ttk.LabelFrame(p, text="📂  Входные файлы")
        files_lf.grid(row=0, column=0, columnspan=2, sticky="ew", padx=8, pady=6)
        files_lf.columnconfigure(1, weight=1)

        for i, (label, var) in enumerate([
            ("Исходный спектр:",         self.src_var),
            ("Дейтерометилирование:",     self.dmet_var),
            ("Дейтероацилирование:",      self.dacet_var),
        ]):
            ttk.Label(files_lf, text=label).grid(row=i, column=0, sticky="w",
                                                  padx=6, pady=3)
            ent = ttk.Entry(files_lf, textvariable=var, width=55)
            ent.grid(row=i, column=1, sticky="ew", padx=4, pady=3)
            btn = ttk.Button(files_lf, text="…",
                             command=lambda v=var: self._browse(v))
            btn.grid(row=i, column=2, padx=4, pady=3)

        # --- Загрузка ---
        load_lf = ttk.LabelFrame(p, text="📥  Загрузка и диапазон масс")
        load_lf.grid(row=1, column=0, sticky="ew", padx=8, pady=6)

        for row, (lbl, var) in enumerate([
            ("Разделитель CSV:",  self.sep_var),
            ("m/z min:",         self.mass_min_var),
            ("m/z max:",         self.mass_max_var),
            ("Шум (force):",     self.noise_var),
        ]):
            ttk.Label(load_lf, text=lbl).grid(row=row, column=0, sticky="w",
                                               padx=6, pady=3)
            ttk.Entry(load_lf, textvariable=var, width=12).grid(
                row=row, column=1, sticky="w", padx=4, pady=3)

        # --- Назначение формул ---
        form_lf = ttk.LabelFrame(p, text="🔬  Назначение брутто-формул")
        form_lf.grid(row=1, column=1, sticky="ew", padx=8, pady=6)

        ttk.Label(form_lf, text="Знак иона:").grid(row=0, column=0, sticky="w",
                                                     padx=6, pady=3)
        sign_cb = ttk.Combobox(form_lf, textvariable=self.sign_var,
                               values=["-", "+", "0"], width=5, state="readonly")
        sign_cb.grid(row=0, column=1, sticky="w", padx=4, pady=3)

        ttk.Label(form_lf, text="Погрешность (ppm):").grid(row=1, column=0,
                                                             sticky="w", padx=6, pady=3)
        ttk.Entry(form_lf, textvariable=self.rel_error_var, width=8).grid(
            row=1, column=1, sticky="w", padx=4, pady=3)

        ttk.Label(form_lf, text="Диапазоны элементов:").grid(row=2, column=0,
                                                               columnspan=2, sticky="w",
                                                               padx=6, pady=(8, 2))
        for i, (sym, mn, mx) in enumerate([
            ("C", self.c_min, self.c_max),
            ("H", self.h_min, self.h_max),
            ("O", self.o_min, self.o_max),
            ("N", self.n_min, self.n_max),
        ]):
            r = 3 + i
            ttk.Label(form_lf, text=f"{sym}:").grid(row=r, column=0, sticky="w",
                                                      padx=20, pady=2)
            ef = ttk.Frame(form_lf)
            ef.grid(row=r, column=1, sticky="w", padx=4, pady=2)
            ttk.Entry(ef, textvariable=mn, width=5).pack(side="left")
            ttk.Label(ef, text="–").pack(side="left", padx=2)
            ttk.Entry(ef, textvariable=mx, width=5).pack(side="left")

        # --- Поиск серий ---
        ser_lf = ttk.LabelFrame(p, text="🔍  Поиск серий")
        ser_lf.grid(row=2, column=0, sticky="ew", padx=8, pady=6)

        ttk.Label(ser_lf, text="Допуск поиска (ppm):").grid(row=0, column=0,
                                                              sticky="w", padx=6, pady=3)
        ttk.Entry(ser_lf, textvariable=self.ppm_tol_var, width=8).grid(
            row=0, column=1, sticky="w", padx=4, pady=3)

        ttk.Label(ser_lf, text="Макс. групп:").grid(row=1, column=0, sticky="w",
                                                      padx=6, pady=3)
        ttk.Entry(ser_lf, textvariable=self.max_groups_var, width=8).grid(
            row=1, column=1, sticky="w", padx=4, pady=3)

        ttk.Checkbutton(ser_lf, text="Разрешить пропуски в сериях",
                        variable=self.allow_gaps_var).grid(row=2, column=0,
                                                            columnspan=2, sticky="w",
                                                            padx=6, pady=4)

        # --- Вывод ---
        out_lf = ttk.LabelFrame(p, text="💾  Выходной файл")
        out_lf.grid(row=2, column=1, sticky="ew", padx=8, pady=6)
        out_lf.columnconfigure(0, weight=1)

        ttk.Entry(out_lf, textvariable=self.output_csv_var, width=35).grid(
            row=0, column=0, sticky="ew", padx=6, pady=4)
        ttk.Button(out_lf, text="…",
                   command=lambda: self._save_browse(self.output_csv_var)).grid(
            row=0, column=1, padx=4, pady=4)

        # --- Кнопка запуска ---
        run_btn = ttk.Button(p, text="▶  Запустить анализ",
                             style="Accent.TButton",
                             command=self._run)
        run_btn.grid(row=3, column=0, columnspan=2, pady=12, ipadx=20, ipady=4)

    # ── ВКЛАДКА СПЕКТРЫ ───────────────────────────────────────────────────────

    def _build_spectra_tab(self):
        frame = self.tab_spectra
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)

        ctrl = ttk.Frame(frame)
        ctrl.pack(fill="x", pady=4, padx=8)

        ttk.Button(ctrl, text="📈 Построить спектры",
                   command=self._plot_spectra).pack(side="left", padx=4)
        ttk.Button(ctrl, text="🗑 Очистить",
                   command=lambda: self._clear_canvas(self.spectra_canvas_frame)).pack(
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
                   command=lambda: self._clear_canvas(self.series_canvas_frame)).pack(
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

        # Таблица
        tbl_frame = ttk.Frame(frame)
        tbl_frame.pack(fill="both", expand=True, padx=8, pady=4)

        cols = ("mass", "brutto", "N_COOH", "N_OH_total", "N_OH",
                "missing_dmet", "missing_dacet")
        self.result_tree = ttk.Treeview(tbl_frame, columns=cols,
                                        show="headings", height=18)
        col_widths = [100, 130, 80, 100, 80, 140, 140]
        col_labels = ["m/z", "Формула", "N_COOH", "N_OH_total", "N_OH",
                      "Пропуски dmet", "Пропуски dacet"]
        for c, lbl, w in zip(cols, col_labels, col_widths):
            self.result_tree.heading(c, text=lbl,
                                     command=lambda _c=c: self._sort_tree(_c))
            self.result_tree.column(c, width=w, anchor="center")

        vsb = ttk.Scrollbar(tbl_frame, orient="vertical",
                            command=self.result_tree.yview)
        hsb = ttk.Scrollbar(tbl_frame, orient="horizontal",
                            command=self.result_tree.xview)
        self.result_tree.configure(yscrollcommand=vsb.set,
                                   xscrollcommand=hsb.set)
        vsb.pack(side="right", fill="y")
        hsb.pack(side="bottom", fill="x")
        self.result_tree.pack(fill="both", expand=True)

        # Мини-гистограмма под таблицей
        self.hist_frame = ttk.Frame(frame)
        self.hist_frame.pack(fill="x", padx=8, pady=4)

    # ── ВКЛАДКА ЛОГ ──────────────────────────────────────────────────────────

    def _build_log_tab(self):
        frame = self.tab_log
        ctrl = ttk.Frame(frame)
        ctrl.pack(fill="x", pady=4, padx=8)
        ttk.Button(ctrl, text="🗑 Очистить лог",
                   command=self._clear_log).pack(side="left", padx=4)

        self.log_text = scrolledtext.ScrolledText(
            frame, bg=PANEL, fg=FG,
            font=MONO, relief="flat",
            insertbackground=FG
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
            filetypes=[("CSV files", "*.csv"), ("Text files", "*.txt"),
                       ("All files", "*.*")]
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

    def _log(self, msg: str, color: str = FG):
        tag = "ok" if color == OK else ("warn" if color == WARN else "info")
        self.log_text.insert("end", msg + "\n", tag)
        self.log_text.see("end")

    def _clear_log(self):
        self.log_text.delete("1.0", "end")

    def _set_status(self, msg: str):
        self.status_var.set(msg)
        self.update_idletasks()

    # ── Запуск пайплайна в отдельном потоке ──────────────────────────────────

    def _run(self):
        if not CORE_LOADED:
            messagebox.showerror("Ошибка", f"core.py не загружен:\n{_CORE_ERROR}")
            return
        for label, var in [("Исходный", self.src_var),
                            ("Дейтерометилирование", self.dmet_var),
                            ("Дейтероацилирование", self.dacet_var)]:
            if not var.get() or not os.path.exists(var.get()):
                messagebox.showwarning("Файл не найден",
                                       f"Укажите корректный путь: «{label}»")
                return

        self.progress.start(10)
        self._set_status("Выполняется анализ…")
        t = threading.Thread(target=self._run_worker, daemon=True)
        t.start()

    def _run_worker(self):
        try:
            # Сепаратор
            sep = self.sep_var.get()
            if sep in ("\\t", "tab", "TAB"):
                sep = "\t"

            brutto_dict = {
                'C': (int(self.c_min.get()), int(self.c_max.get())),
                'H': (int(self.h_min.get()), int(self.h_max.get())),
                'O': (int(self.o_min.get()), int(self.o_max.get())),
                'N': (int(self.n_min.get()), int(self.n_max.get())),
            }

            # Перехват вывода в лог
            import io
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()

            result = run_pipeline(
                src_path   = self.src_var.get(),
                dmet_path  = self.dmet_var.get(),
                dacet_path = self.dacet_var.get(),
                sep        = sep,
                mass_min   = float(self.mass_min_var.get()),
                mass_max   = float(self.mass_max_var.get()),
                noise_force = float(self.noise_var.get()),
                brutto_dict = brutto_dict,
                rel_error   = float(self.rel_error_var.get()),
                sign        = self.sign_var.get(),
                ppm_tol     = float(self.ppm_tol_var.get()),
                max_groups  = int(self.max_groups_var.get()),
                allow_gaps  = self.allow_gaps_var.get(),
                visualize   = False,
                save_dmet=None,
                save_dacet=None,
                output_csv  = self.output_csv_var.get() or None,
            )

            captured = sys.stdout.getvalue()
            sys.stdout = old_stdout

            self.result_df = result
            self.after(0, lambda: self._on_run_success(captured))

        except Exception as e:
            sys.stdout = old_stdout if 'old_stdout' in dir() else sys.stdout
            tb = traceback.format_exc()
            self.after(0, lambda: self._on_run_error(tb))

    def _on_run_success(self, log_text: str):
        self.progress.stop()
        self._set_status(f"Готово. Найдено {len(self.result_df)} соединений.")
        self._log(log_text, color=FG)
        self._log("✅ Анализ завершён успешно.", color=OK)
        self._fill_result_table(self.result_df)
        self._auto_plot_hist()

    def _on_run_error(self, tb: str):
        self.progress.stop()
        self._set_status("Ошибка! Смотри лог.")
        self._log(tb, color=WARN)
        messagebox.showerror("Ошибка выполнения", tb[:400])

    # ── Таблица результатов ───────────────────────────────────────────────────

    def _fill_result_table(self, df: pd.DataFrame):
        for row in self.result_tree.get_children():
            self.result_tree.delete(row)

        for _, r in df.iterrows():
            vals = (
                f"{r['mass']:.5f}",
                r.get("brutto", ""),
                int(r.get("N_COOH", 0)),
                int(r.get("N_OH_total", 0)),
                int(r.get("N_OH", 0)),
                str(r.get("missing_dmet", [])),
                str(r.get("missing_dacet", [])),
            )
            tag = ""
            if r.get("missing_dmet") or r.get("missing_dacet"):
                tag = "warn"
            self.result_tree.insert("", "end", values=vals, tags=(tag,))

        self.result_tree.tag_configure("warn", foreground=WARN)

    def _sort_tree(self, col: str):
        if self.result_df is None:
            return
        ascending = getattr(self, f"_sort_{col}_asc", True)
        self.result_df = self.result_df.sort_values(
            col, ascending=ascending, na_position="last"
        )
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
            self.result_df.to_csv(path, index=False, sep=";", encoding="utf-8-sig")
            self._log(f"Таблица сохранена: {path}", color=OK)

    # ── Графики спектров ──────────────────────────────────────────────────────

    def _plot_spectra(self):
        if not all([self.src_var.get(), self.dmet_var.get(), self.dacet_var.get()]):
            messagebox.showwarning("Нет файлов", "Укажите все три файла.")
            return

        sep = self.sep_var.get()
        if sep in ("\\t", "tab", "TAB"):
            sep = "\t"

        try:
            dfs = {}
            for key, var in [("Исходный", self.src_var),
                              ("Дейтерометилирование", self.dmet_var),
                              ("Дейтероацилирование", self.dacet_var)]:
                df = pd.read_csv(var.get(), sep=sep)
                df.columns = [c.strip() for c in df.columns]
                if "m/z" in df.columns:
                    df = df.rename(columns={"m/z": "mass"})
                dfs[key] = df
        except Exception as e:
            messagebox.showerror("Ошибка чтения", str(e))
            return

        self._clear_canvas(self.spectra_canvas_frame)

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

        self._embed_figure(fig, self.spectra_canvas_frame)

    # ── Графики серий ─────────────────────────────────────────────────────────

    def _plot_series(self, which: str):
        if self.result_df is None:
            messagebox.showinfo("Нет данных", "Сначала запустите анализ.")
            return

        col_n = "N_COOH" if which == "dmet" else "N_OH_total"
        col_m = "missing_dmet" if which == "dmet" else "missing_dacet"
        delta = DELTA_CD3 if which == "dmet" else DELTA_CD3CO
        label = "CD₃ (dmet)" if which == "dmet" else "CD₃CO (dacet)"

        df = self.result_df[self.result_df[col_n] > 0].copy()
        if df.empty:
            self._log(f"Серии {label}: не найдено.", color=WARN)
            return

        n_plots = min(len(df), 9)
        ncols = 3
        nrows = (n_plots + ncols - 1) // ncols

        self._clear_canvas(self.series_canvas_frame)
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(9, nrows * 2.8))
        axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

        for i, (_, row) in enumerate(df.head(n_plots).iterrows()):
            ax = axes[i]
            m0 = row["mass"]
            n = int(row[col_n])
            steps = list(range(1, n + 1))
            mz_vals = [m0 + s * delta for s in steps]

            missing = row.get(col_m, [])
            if isinstance(missing, str):
                import ast
                try:
                    missing = ast.literal_eval(missing)
                except Exception:
                    missing = []

            colors_bars = [WARN if s in missing else OK for s in steps]
            ax.bar(steps, [1] * len(steps), color=colors_bars, alpha=0.8, width=0.6)
            ax.set_xticks(steps)
            ax.set_xticklabels([str(s) for s in steps], fontsize=7)
            ax.set_yticks([])
            ax.set_title(f"m/z={m0:.3f}\n{row.get('brutto','')}, n={n}",
                         fontsize=7, color=FG)
            if missing:
                ax.set_xlabel(f"⚠ пропуски: {missing}", fontsize=6,
                              color=WARN)

        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        fig.suptitle(f"Серии {label}  (зелёный=найден, красный=пропущен)",
                     color=ACCENT, fontsize=10)
        fig.tight_layout()
        self._embed_figure(fig, self.series_canvas_frame)

    # ── Гистограммы ───────────────────────────────────────────────────────────

    def _auto_plot_hist(self):
        if self.result_df is None:
            return
        self._clear_canvas(self.hist_frame)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 2.5))
        for ax, col, color in [(ax1, "N_COOH", "#f38ba8"),
                               (ax2, "N_OH", "#a6e3a1")]:
            vals = self.result_df[col].dropna().astype(int)
            if not vals.empty:
                ax.hist(vals, bins=range(vals.max() + 2),
                        color=color, alpha=0.85, edgecolor=BG, rwidth=0.7)
            ax.set_xlabel(col, fontsize=8)
            ax.set_ylabel("Кол-во", fontsize=8)
            ax.grid(True, alpha=0.3)
        fig.tight_layout()
        self._embed_figure(fig, self.hist_frame, toolbar=False)

    def _plot_hist(self, col: str):
        if self.result_df is None:
            messagebox.showinfo("Нет данных", "Сначала запустите анализ.")
            return
        self._clear_canvas(self.series_canvas_frame)
        fig, ax = plt.subplots(figsize=(7, 4))
        vals = self.result_df[col].dropna().astype(int)
        if vals.empty:
            ax.text(0.5, 0.5, "Нет данных", transform=ax.transAxes,
                    ha="center", color=FG)
        else:
            ax.hist(vals, bins=range(vals.max() + 2),
                    color=ACCENT, alpha=0.85,
                    edgecolor=BG, rwidth=0.7)
        ax.set_xlabel(col)
        ax.set_ylabel("Количество соединений")
        ax.set_title(f"Распределение {col}")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        self._embed_figure(fig, self.series_canvas_frame)



# ═══════════════════════════════════════════════════════════════════════════════
#  ТОЧКА ВХОДА
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    app = App()
    app.mainloop()