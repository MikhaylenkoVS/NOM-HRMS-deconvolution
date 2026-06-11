from __future__ import annotations
import os
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import Optional

import pandas as pd

# PIL нужен для отображения PIL.Image в Tk
try:
    from PIL import Image, ImageTk
    PIL_OK = True
except ImportError:
    PIL_OK = False

# RDKit — для .mol и .png
try:
    from rdkit import Chem
    from rdkit.Chem import Draw, AllChem, rdMolDescriptors
    RDKIT_OK = True
except ImportError:
    RDKIT_OK = False

# ── Цвета (совпадают с app.py) ────────────────────────────────────────────────
BG      = "#1e1e2e"
FG      = "#cdd6f4"
ACCENT  = "#89b4fa"
PANEL   = "#313244"
BTN     = "#45475a"
OK      = "#a6e3a1"
WARN    = "#f38ba8"
FONT    = ("Segoe UI", 10)
MONO    = ("Consolas", 9)

IMG_W, IMG_H = 340, 260   # размер одного превью


# ═══════════════════════════════════════════════════════════════════════════════
#  ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ═══════════════════════════════════════════════════════════════════════════════

def _fragment_to_rdkit(mol_fragment):
    """Конвертирует MoleculeFragment → rdkit.Mol (с 2D-координатами)."""
    if not RDKIT_OK:
        return None
    try:
        from core import to_rdkit_mol
        return to_rdkit_mol(mol_fragment)
    except Exception:
        pass
    # Запасной путь — прямая конвертация
    try:
        rw = Chem.RWMol()
        atom_map = {}
        for i, sym in enumerate(mol_fragment.atoms):
            idx = rw.AddAtom(Chem.Atom(sym))
            atom_map[i] = idx
        bond_types = {
            1: Chem.BondType.SINGLE,
            2: Chem.BondType.DOUBLE,
            3: Chem.BondType.TRIPLE,
        }
        for a, b, order in mol_fragment.bonds:
            rw.AddBond(atom_map[a], atom_map[b],
                       bond_types.get(order, Chem.BondType.SINGLE))
        mol = rw.GetMol()
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            pass
        mol = Chem.AddHs(mol)
        AllChem.Compute2DCoords(mol)
        return mol
    except Exception:
        return None


def _mol_to_pil(rdmol, size=(IMG_W, IMG_H)) -> Optional["Image.Image"]:
    if not RDKIT_OK or not PIL_OK or rdmol is None:
        return None
    try:
        img = Draw.MolToImage(rdmol, size=size)
        return img
    except Exception:
        return None


def _pil_to_tk(pil_img) -> Optional["ImageTk.PhotoImage"]:
    if not PIL_OK or pil_img is None:
        return None
    return ImageTk.PhotoImage(pil_img)


def _save_mol(rdmol, path: str):
    block = Chem.MolToMolBlock(rdmol)
    with open(path, "w", encoding="utf-8") as f:
        f.write(block)


def _save_png(rdmol, path: str, size=(800, 600)):
    img = Draw.MolToImage(rdmol, size=size)
    img.save(path)


# ═══════════════════════════════════════════════════════════════════════════════
#  ДИАЛОГ ОДНОЙ СТРУКТУРЫ
# ═══════════════════════════════════════════════════════════════════════════════

class StructureDialog(tk.Toplevel):
    """Всплывающее окно с увеличенным изображением и кнопками экспорта."""

    def __init__(self, parent, mol_info: dict, rdmol):
        super().__init__(parent)
        self.configure(bg=BG)
        self.title(f"Структура: {mol_info.get('name', '?')}")
        self.resizable(True, True)
        self.rdmol   = rdmol
        self.mol_info = mol_info

        self._build(mol_info, rdmol)

    def _build(self, info: dict, rdmol):
        # Название
        tk.Label(self, text=info.get("name", ""), bg=BG, fg=ACCENT,
                 font=("Segoe UI", 12, "bold")).pack(pady=(10, 2))
        formula_str = "  ".join(f"{k}{v}" for k, v in
                                sorted(info.get("formula", {}).items()))
        tk.Label(self, text=formula_str, bg=BG, fg=FG,
                 font=FONT).pack()
        tk.Label(self,
                 text=f"IHD = {info.get('ihd', '?')}  |  "
                      f"Атомов (тяж.) = {info.get('num_atoms', '?')}  |  "
                      f"Связей = {info.get('num_bonds', '?')}",
                 bg=BG, fg=FG, font=MONO).pack(pady=2)

        # Изображение
        if rdmol is not None and PIL_OK:
            pil = _mol_to_pil(rdmol, size=(560, 420))
            if pil:
                self._tk_img = _pil_to_tk(pil)
                tk.Label(self, image=self._tk_img, bg=BG).pack(
                    padx=16, pady=8)

        # Фрагменты
        combo_info = info.get("combination", {})
        bases = combo_info.get("bases", [])
        tk.Label(self,
                 text=f"Фрагменты: {', '.join(bases) if bases else '—'}",
                 bg=BG, fg=FG, font=MONO).pack()

        # Кнопки экспорта
        btn_frame = ttk.Frame(self)
        btn_frame.pack(pady=10)
        ttk.Button(btn_frame, text="💾 Сохранить .mol",
                   command=self._save_mol).pack(side="left", padx=6)
        ttk.Button(btn_frame, text="🖼 Сохранить .png",
                   command=self._save_png).pack(side="left", padx=6)
        ttk.Button(btn_frame, text="✕ Закрыть",
                   command=self.destroy).pack(side="left", padx=6)

    def _save_mol(self):
        if not RDKIT_OK or self.rdmol is None:
            messagebox.showerror("Ошибка", "RDKit не установлен или молекула недоступна.")
            return
        name_safe = self.mol_info.get("name", "molecule").replace("+", "_")
        path = filedialog.asksaveasfilename(
            defaultextension=".mol",
            initialfile=f"{name_safe}.mol",
            filetypes=[("MDL Molfile", "*.mol"), ("All", "*.*")]
        )
        if path:
            try:
                _save_mol(self.rdmol, path)
                messagebox.showinfo("Сохранено", path)
            except Exception as e:
                messagebox.showerror("Ошибка", str(e))

    def _save_png(self):
        if not RDKIT_OK or self.rdmol is None:
            messagebox.showerror("Ошибка", "RDKit не установлен или молекула недоступна.")
            return
        name_safe = self.mol_info.get("name", "molecule").replace("+", "_")
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            initialfile=f"{name_safe}.png",
            filetypes=[("PNG", "*.png"), ("All", "*.*")]
        )
        if path:
            try:
                _save_png(self.rdmol, path)
                messagebox.showinfo("Сохранено", path)
            except Exception as e:
                messagebox.showerror("Ошибка", str(e))


# ═══════════════════════════════════════════════════════════════════════════════
#  ВИДЖЕТ — ОДНА КАРТОЧКА СТРУКТУРЫ
# ═══════════════════════════════════════════════════════════════════════════════

class StructureCard(ttk.Frame):
    """Миниатюрная карточка с изображением и кнопками."""

    def __init__(self, parent, mol_info: dict, rdmol, index: int, **kw):
        super().__init__(parent, **kw)
        self.configure(style="Card.TFrame")
        self.mol_info = mol_info
        self.rdmol    = rdmol
        self._build(mol_info, rdmol, index)

    def cleanup(self):
        """Явно обнуляем PhotoImage до уничтожения виджета."""
        self._tk_img = None

    def _build(self, info: dict, rdmol, idx: int):
        # Заголовок карточки
        name = info.get("name", f"вариант {idx}")
        short = name if len(name) <= 30 else name[:27] + "…"
        tk.Label(self, text=f"#{idx}  {short}", bg=PANEL, fg=ACCENT,
                 font=("Segoe UI", 9, "bold"), anchor="w").pack(
            fill="x", padx=4, pady=(4, 0))

        formula_str = " ".join(f"{k}{v}" for k, v in
                               sorted(info.get("formula", {}).items()))
        tk.Label(self, text=f"{formula_str}  IHD={info.get('ihd', '?')}",
                 bg=PANEL, fg=FG, font=MONO, anchor="w").pack(
            fill="x", padx=4)

        # Изображение
        self._tk_img = None
        if rdmol is not None and PIL_OK:
            pil = _mol_to_pil(rdmol, size=(IMG_W, IMG_H))
            if pil:
                self._tk_img = _pil_to_tk(pil)
                lbl = tk.Label(self, image=self._tk_img, bg=PANEL,
                               cursor="hand2")
                lbl.pack(padx=4, pady=4)
                lbl.bind("<Button-1>", lambda e: self._open_detail())
        else:
            tk.Label(self, text="(нет изображения)", bg=PANEL, fg=WARN,
                     font=MONO).pack(padx=4, pady=20)

        # Кнопки
        bf = tk.Frame(self, bg=PANEL)
        bf.pack(fill="x", padx=4, pady=(0, 6))
        tk.Button(bf, text="🔍 Подробнее", bg=BTN, fg=FG, font=MONO,
                  relief="flat", cursor="hand2",
                  command=self._open_detail).pack(side="left", padx=2)
        tk.Button(bf, text=".mol", bg=BTN, fg=FG, font=MONO,
                  relief="flat", cursor="hand2",
                  command=self._export_mol).pack(side="left", padx=2)
        tk.Button(bf, text=".png", bg=BTN, fg=FG, font=MONO,
                  relief="flat", cursor="hand2",
                  command=self._export_png).pack(side="left", padx=2)

    def _open_detail(self):
        StructureDialog(self, self.mol_info, self.rdmol)

    def _export_mol(self):
        if not RDKIT_OK or self.rdmol is None:
            messagebox.showerror("Нет RDKit", "Установите: pip install rdkit")
            return
        name_safe = self.mol_info.get("name", "mol").replace("+", "_")
        path = filedialog.asksaveasfilename(
            defaultextension=".mol", initialfile=f"{name_safe}.mol",
            filetypes=[("MDL Molfile", "*.mol"), ("All", "*.*")])
        if path:
            _save_mol(self.rdmol, path)

    def _export_png(self):
        if not RDKIT_OK or self.rdmol is None:
            messagebox.showerror("Нет RDKit", "Установите: pip install rdkit")
            return
        name_safe = self.mol_info.get("name", "mol").replace("+", "_")
        path = filedialog.asksaveasfilename(
            defaultextension=".png", initialfile=f"{name_safe}.png",
            filetypes=[("PNG", "*.png"), ("All", "*.*")])
        if path:
            _save_png(self.rdmol, path)


# ═══════════════════════════════════════════════════════════════════════════════
#  ГЛАВНАЯ ВКЛАДКА
# ═══════════════════════════════════════════════════════════════════════════════

class StructureViewerTab(ttk.Frame):
    """
    Вкладка «🧪 Структуры» для app.py.

    Параметры:
        parent  — ttk.Notebook (родитель)
        app     — экземпляр App из app.py (для доступа к result_df)
    """

    def __init__(self, parent, app, **kw):
        super().__init__(parent, **kw)
        self.app = app         # ссылка на основное приложение
        self._cards: list     = []
        self._rdmols: list    = []
        self._cur_row_mass: Optional[float] = None

        # Регистрация стиля карточки
        s = ttk.Style()
        s.configure("Card.TFrame", background=PANEL,
                    relief="solid", borderwidth=1)

        self._build_ui()

    # ── UI ───────────────────────────────────────────────────────────────────

    def _build_ui(self):
        # ── Панель управления ──
        ctrl = ttk.Frame(self)
        ctrl.pack(fill="x", padx=8, pady=6)

        # Выбор строки из результатов
        ttk.Label(ctrl, text="Соединение (m/z):").pack(side="left", padx=(0, 4))
        self.peak_var = tk.StringVar()
        self.peak_cb  = ttk.Combobox(ctrl, textvariable=self.peak_var,
                                      width=22, state="readonly")
        self.peak_cb.pack(side="left", padx=4)
        self.peak_cb.bind("<<ComboboxSelected>>", lambda e: self._on_peak_select())

        ttk.Button(ctrl, text="🔄 Обновить список",
                   command=self._refresh_peak_list).pack(side="left", padx=4)

        ttk.Separator(ctrl, orient="vertical").pack(
            side="left", fill="y", padx=8)

        ttk.Label(ctrl, text="Макс. вариантов:").pack(side="left", padx=(0, 4))
        self.max_var = tk.StringVar(value="12")
        ttk.Entry(ctrl, textvariable=self.max_var, width=5).pack(
            side="left", padx=4)

        ttk.Button(ctrl, text="▶ Найти структуры",
                   command=self._run_search).pack(side="left", padx=8)

        # Экспорт всех
        ttk.Separator(ctrl, orient="vertical").pack(
            side="left", fill="y", padx=8)
        ttk.Button(ctrl, text="📦 Экспорт всех .mol",
                   command=lambda: self._export_all("mol")).pack(
            side="left", padx=4)
        ttk.Button(ctrl, text="📦 Экспорт всех .png",
                   command=lambda: self._export_all("png")).pack(
            side="left", padx=4)

        # ── Информационная строка ──
        self.info_var = tk.StringVar(value="Выберите соединение и нажмите «Найти структуры»")
        tk.Label(self, textvariable=self.info_var, bg=BG, fg=ACCENT,
                 font=("Segoe UI", 9), anchor="w").pack(
            fill="x", padx=12, pady=(0, 4))

        # ── Прогресс-бар ──
        self.progress = ttk.Progressbar(self, mode="indeterminate")
        self.progress.pack(fill="x", padx=8, pady=2)

        # ── Область карточек со скроллом ──
        container = ttk.Frame(self)
        container.pack(fill="both", expand=True, padx=8, pady=4)
        container.rowconfigure(0, weight=1)
        container.columnconfigure(0, weight=1)

        canvas = tk.Canvas(container, bg=BG, highlightthickness=0)
        vsb = ttk.Scrollbar(container, orient="vertical",
                            command=canvas.yview)
        hsb = ttk.Scrollbar(container, orient="horizontal",
                            command=canvas.xview)
        canvas.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        canvas.grid(row=0, column=0, sticky="nsew")

        self.cards_frame = tk.Frame(canvas, bg=BG)
        self._canvas_window = canvas.create_window(
            (0, 0), window=self.cards_frame, anchor="nw")

        self.cards_frame.bind("<Configure>",
                              lambda e: canvas.configure(
                                  scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>",
                    lambda e: canvas.itemconfig(
                        self._canvas_window, width=e.width))
        # Прокрутка колёсиком
        canvas.bind_all("<MouseWheel>",
                        lambda e: canvas.yview_scroll(-int(e.delta / 120), "units"))

        self._canvas = canvas

    # ── Логика ───────────────────────────────────────────────────────────────

    def _refresh_peak_list(self):
        """Заполняет выпадающий список из result_df."""
        df = getattr(self.app, "result_df", None)
        if df is None or df.empty:
            messagebox.showinfo("Нет данных",
                                "Сначала запустите анализ на вкладке «Параметры».")
            return
        values = []
        for _, row in df.iterrows():
            mass  = row.get("mass", "?")
            brutto = row.get("brutto", "")
            ncooh = int(row.get("N_COOH", 0))
            noh   = int(row.get("N_OH", 0))
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
            messagebox.showwarning("Не выбрано",
                                   "Выберите соединение из списка.")
            return
        brutto  = row.get("brutto", "")
        n_cooh  = int(row.get("N_COOH", 0))
        n_oh    = int(row.get("N_OH", 0))
        max_b   = int(self.max_var.get() or 12)

        if not brutto:
            messagebox.showwarning("Нет формулы",
                                   "Для выбранного пика не назначена брутто-формула.")
            return

        self.info_var.set(f"Ищу структуры для {brutto}  COOH={n_cooh}  OH={n_oh}…")
        self.progress.start(10)
        self._clear_cards()

        t = threading.Thread(target=self._search_worker,
                             args=(brutto, n_cooh, n_oh, max_b),
                             daemon=True)
        t.start()

    def _search_worker(self, brutto: str, n_cooh: int, n_oh: int, max_bases: int):
        try:
            from core import find_and_visualize_molecules
            result = find_and_visualize_molecules(
                brutto, num_cooh=n_cooh, num_oh=n_oh,
                max_bases=max_bases, show_images=False
            )
            molecules = result.get("molecules", [])
            self.after(0, lambda: self._on_search_done(molecules, brutto,
                                                        n_cooh, n_oh))
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            self.after(0, lambda: self._on_search_error(tb))

    def _on_search_done(self, molecules: list, brutto: str,
                         n_cooh: int, n_oh: int):
        self.progress.stop()
        self._clear_cards()
        self._rdmols = []

        if not molecules:
            self.info_var.set(
                f"Структуры для {brutto} COOH={n_cooh} OH={n_oh} не найдены.")
            return

        self.info_var.set(
            f"Найдено {len(molecules)} вариантов для {brutto}  "
            f"COOH={n_cooh}  OH={n_oh}")

        cols = max(1, min(3, len(molecules)))   # до 3 колонок
        for i, mol_info in enumerate(molecules):
            frag = mol_info.get("fragment_object")
            rdmol = _fragment_to_rdkit(frag) if frag is not None else None
            self._rdmols.append(rdmol)

            card = StructureCard(
                self.cards_frame, mol_info, rdmol, index=i + 1
            )
            row_idx = i // cols
            col_idx = i  % cols
            card.grid(row=row_idx, column=col_idx,
                      padx=6, pady=6, sticky="nw")
            self._cards.append(card)

        # Обновить размер canvas
        self.cards_frame.update_idletasks()
        self._canvas.configure(
            scrollregion=self._canvas.bbox("all"))

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
            messagebox.showinfo("Нет структур",
                                "Сначала найдите структуры для какого-либо пика.")
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
                    _save_mol(rdmol, path)
                else:
                    _save_png(rdmol, path)
                saved += 1
            except Exception as e:
                pass

        messagebox.showinfo("Экспорт завершён",
                            f"Сохранено файлов: {saved} из {len(self._rdmols)}\n"
                            f"Папка: {folder}")