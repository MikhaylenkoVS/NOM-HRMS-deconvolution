# structures/widgets.py
import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from ..ui.theme import BG, FG, ACCENT, PANEL, BTN, WARN, FONT, MONO, IMG_W, IMG_H
from .rdkit_utils import (
    RDKIT_OK,
    PIL_OK,
    mol_to_pil,
    pil_to_tk,
    save_mol,
    save_png,
)

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
            pil = mol_to_pil(rdmol, size=(560, 420))
            if pil:
                self._tk_img = pil_to_tk(pil)
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
                save_mol(self.rdmol, path)
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
                save_png(self.rdmol, path)
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
            pil = mol_to_pil(rdmol, size=(IMG_W, IMG_H))
            if pil:
                self._tk_img = pil_to_tk(pil)
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
            save_mol(self.rdmol, path)

    def _export_png(self):
        if not RDKIT_OK or self.rdmol is None:
            messagebox.showerror("Нет RDKit", "Установите: pip install rdkit")
            return
        name_safe = self.mol_info.get("name", "mol").replace("+", "_")
        path = filedialog.asksaveasfilename(
            defaultextension=".png", initialfile=f"{name_safe}.png",
            filetypes=[("PNG", "*.png"), ("All", "*.*")])
        if path:
            save_png(self.rdmol, path)
