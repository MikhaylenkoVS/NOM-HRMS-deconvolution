# structures/tab.py
import os
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import Optional
import pandas as pd
from src.ui import BG, ACCENT, PANEL
from .rdkit_utils import fragment_to_rdkit, RDKIT_OK, save_mol, save_png
from .widgets import StructureCard


class StructureViewerTab(ttk.Frame):
    """Notebook tab that generates and displays candidate structures.

    Lets the user pick an assigned peak (by ``m/z``) from the pipeline
    results, enumerate plausible molecular structures for its brutto
    formula (respecting the assigned -COOH/-OH counts), and browse the
    results as RDKit-rendered cards with per-structure and batch export
    to ``.mol``/``.png``.

    Parameters
    ----------
    parent : tkinter.ttk.Notebook
        Parent notebook widget hosting the tab.
    app : App
        Main application instance from :mod:`src.app`, used to access the
        results table ``result_df`` and the shared log widget.
    **kw
        Additional keyword arguments forwarded to ``ttk.Frame``.
    """

    def __init__(self, parent, app, **kw):
        super().__init__(parent, **kw)
        self.app = app  # ссылка на основное приложение
        self._cards: list = []
        self._rdmols: list = []
        self._cur_row_mass: Optional[float] = None

        # Регистрация стиля карточки
        s = ttk.Style()
        s.configure("Card.TFrame", background=PANEL, relief="solid", borderwidth=1)

        self._build_ui()

    # ── UI ───────────────────────────────────────────────────────────────────

    def _build_ui(self):
        # ── Панель управления ──
        ctrl = ttk.Frame(self)
        ctrl.pack(fill="x", padx=8, pady=6)

        # Выбор строки из результатов
        ttk.Label(ctrl, text="Соединение (m/z):").pack(side="left", padx=(0, 4))
        self.peak_var = tk.StringVar()
        self.peak_cb = ttk.Combobox(
            ctrl, textvariable=self.peak_var, width=22, state="readonly"
        )
        self.peak_cb.pack(side="left", padx=4)
        self.peak_cb.bind("<<ComboboxSelected>>", lambda e: self._on_peak_select())

        ttk.Button(
            ctrl, text="🔄 Обновить список", command=self._refresh_peak_list
        ).pack(side="left", padx=4)

        ttk.Button(ctrl, text="▶ Найти структуры", command=self._run_search).pack(
            side="left", padx=8
        )

        # Экспорт всех
        ttk.Separator(ctrl, orient="vertical").pack(side="left", fill="y", padx=8)
        ttk.Button(
            ctrl, text="📦 Экспорт всех .mol", command=lambda: self._export_all("mol")
        ).pack(side="left", padx=4)
        ttk.Button(
            ctrl, text="📦 Экспорт всех .png", command=lambda: self._export_all("png")
        ).pack(side="left", padx=4)

        # ── Информационная строка ──
        self.info_var = tk.StringVar(
            value="Выберите соединение и нажмите «Найти структуры»"
        )
        tk.Label(
            self,
            textvariable=self.info_var,
            bg=BG,
            fg=ACCENT,
            font=("Segoe UI", 9),
            anchor="w",
        ).pack(fill="x", padx=12, pady=(0, 4))

        # ── Прогресс-бар ──
        self.progress = ttk.Progressbar(self, mode="indeterminate")
        self.progress.pack(fill="x", padx=8, pady=2)

        # ── Область карточек со скроллом ──
        container = ttk.Frame(self)
        container.pack(fill="both", expand=True, padx=8, pady=4)
        container.rowconfigure(0, weight=1)
        container.columnconfigure(0, weight=1)

        canvas = tk.Canvas(container, bg=BG, highlightthickness=0)
        vsb = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        hsb = ttk.Scrollbar(container, orient="horizontal", command=canvas.xview)
        canvas.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        canvas.grid(row=0, column=0, sticky="nsew")

        self.cards_frame = tk.Frame(canvas, bg=BG)
        self._canvas_window = canvas.create_window(
            (0, 0), window=self.cards_frame, anchor="nw"
        )

        self.cards_frame.bind(
            "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.bind(
            "<Configure>",
            lambda e: canvas.itemconfig(self._canvas_window, width=e.width),
        )
        # Прокрутка колёсиком
        canvas.bind_all(
            "<MouseWheel>", lambda e: canvas.yview_scroll(-int(e.delta / 120), "units")
        )

        self._canvas = canvas

    # ── Логика ───────────────────────────────────────────────────────────────

    def _refresh_peak_list(self):
        """Заполняет выпадающий список из result_df."""
        df = getattr(self.app, "result_df", None)
        if df is None or df.empty:
            messagebox.showinfo(
                "Нет данных", "Сначала запустите анализ на вкладке «Параметры»."
            )
            return
        values = []
        for _, row in df.iterrows():
            mass = row.get("mass", "?")
            brutto = row.get("brutto", "")
            ncooh = int(row.get("N_COOH", 0))
            noh = int(row.get("N_OH", 0))
            values.append(f"{float(mass):.4f}  {brutto}  COOH={ncooh} OH={noh}")
        self.peak_cb["values"] = values
        self.info_var.set(f"Загружено {len(values)} пиков из результатов.")

    def _on_peak_select(self):
        val = self.peak_var.get()
        if val:
            mass_str = val.split()[0]
            try:
                self._cur_row_mass = float(mass_str)
            except ValueError:
                self._cur_row_mass = None

    def _get_selected_row(self) -> Optional[pd.Series]:
        df = getattr(self.app, "result_df", None)
        if df is None or self._cur_row_mass is None:
            return None
        matches = df[abs(df["mass"] - self._cur_row_mass) < 0.001]
        if matches.empty:
            return None
        return matches.iloc[0]

    def _run_search(self):
        row = self._get_selected_row()
        if row is None:
            messagebox.showwarning("Не выбрано", "Выберите соединение из списка.")
            return
        brutto = row.get("brutto", "")
        n_cooh = int(row.get("N_COOH", 0))
        n_oh = int(row.get("N_OH", 0))
        max_bases = 12  # разумный максимум для поиска фрагментов

        if not brutto:
            messagebox.showwarning(
                "Нет формулы", "Для выбранного пика не назначена брутто-формула."
            )
            return

        self.info_var.set(f"Ищу структуры для {brutto}  COOH={n_cooh}  OH={n_oh}…")
        self.progress.start(10)
        self._clear_cards()

        t = threading.Thread(
            target=self._search_worker, args=(brutto, n_cooh, n_oh, max_bases), daemon=True
        )
        t.start()

    def _search_worker(self, brutto: str, n_cooh: int, n_oh: int, max_bases: int):
        try:
            from ..core import find_and_visualize_molecules

            result = find_and_visualize_molecules(
                brutto,
                num_cooh=n_cooh,
                num_oh=n_oh,
                max_bases=max_bases,
                show_images=False,
            )
            molecules = result.get("molecules", [])
            self.after(0, lambda: self._on_search_done(molecules, brutto, n_cooh, n_oh))
        except Exception as e:
            import traceback

            tb = traceback.format_exc()
            self.after(0, lambda: self._on_search_error(tb))

    def _on_search_done(self, molecules: list, brutto: str, n_cooh: int, n_oh: int):
        self.progress.stop()
        self._clear_cards()
        self._rdmols = []

        if not molecules:
            self.info_var.set(
                f"Структуры для {brutto} COOH={n_cooh} OH={n_oh} не найдены."
            )
            return

        self.info_var.set(
            f"Найдено {len(molecules)} вариантов для {brutto}  "
            f"COOH={n_cooh}  OH={n_oh}"
        )

        cols = max(1, min(3, len(molecules)))  # до 3 колонок
        for i, mol_info in enumerate(molecules):
            frag = mol_info.get("fragment_object")
            rdmol = fragment_to_rdkit(frag) if frag is not None else None
            self._rdmols.append(rdmol)

            card = StructureCard(self.cards_frame, mol_info, rdmol, index=i + 1)
            row_idx = i // cols
            col_idx = i % cols
            card.grid(row=row_idx, column=col_idx, padx=6, pady=6, sticky="nw")
            self._cards.append(card)

        # Обновить размер canvas
        self.cards_frame.update_idletasks()
        self._canvas.configure(scrollregion=self._canvas.bbox("all"))

    def _on_search_error(self, tb: str):
        self.progress.stop()
        self.info_var.set("Ошибка при поиске структур — см. лог.")
        # Попытка вывести в лог app.py
        log = getattr(self.app, "log_text", None)
        if log:
            log.insert("end", tb + "\n", "warn")
            log.see("end")
        else:
            messagebox.showerror("Ошибка", tb[:500])

    def _clear_cards(self):
        for card in self._cards:
            card.cleanup()  # ← сначала обнуляем PhotoImage
        for widget in self.cards_frame.winfo_children():
            widget.destroy()  # ← потом уничтожаем виджеты
        self._cards = []
        self._rdmols = []

    # ── Экспорт всех ─────────────────────────────────────────────────────────

    def _export_all(self, fmt: str):
        if not self._rdmols:
            messagebox.showinfo(
                "Нет структур", "Сначала найдите структуры для какого-либо пика."
            )
            return
        folder = filedialog.askdirectory(title="Выберите папку для сохранения")
        if not folder:
            return

        saved = 0
        for i, (rdmol, card) in enumerate(zip(self._rdmols, self._cards)):
            if rdmol is None:
                continue
            name_safe = card.mol_info.get("name", f"mol_{i}").replace("+", "_")
            path = os.path.join(folder, f"{name_safe}.{fmt}")
            try:
                if fmt == "mol":
                    save_mol(rdmol, path)
                else:
                    save_png(rdmol, path)
                saved += 1
            except Exception as e:
                pass

        messagebox.showinfo(
            "Экспорт завершён",
            f"Сохранено файлов: {saved} из {len(self._rdmols)}\n" f"Папка: {folder}",
        )
