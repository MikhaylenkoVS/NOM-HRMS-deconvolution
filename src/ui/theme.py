# ═══════════════════════════════════════════════════════════════════════════════
#  ВСПОМОГАТЕЛЬНЫЕ УТИЛИТЫ
# ═══════════════════════════════════════════════════════════════════════════════

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import matplotlib.pyplot as plt

# ui/theme.py
BG      = "#1e1e2e"
FG      = "#cdd6f4"
ACCENT  = "#89b4fa"
PANEL   = "#313244"
BTN     = "#45475a"
OK      = "#a6e3a1"
WARN    = "#f38ba8"
FONT    = ("Segoe UI", 10)
MONO    = ("Consolas", 9)

IMG_W, IMG_H = 340, 260

def _style(root: tk.Tk) -> ttk.Style:
    s = ttk.Style(root)
    s.theme_use("clam")
    s.configure(".",           background=BG, foreground=FG,
                               font=FONT, fieldbackground=PANEL)
    s.configure("TFrame",      background=BG)
    s.configure("TLabelframe", background=BG, foreground=ACCENT)
    s.configure("TLabelframe.Label", background=BG, foreground=ACCENT,
                font=("Segoe UI", 10, "bold"))
    s.configure("TLabel",      background=BG, foreground=FG)
    s.configure("TButton",     background=BTN, foreground=FG,
                               relief="flat", padding=4)
    s.map("TButton",           background=[("active", ACCENT)],
                               foreground=[("active", BG)])
    s.configure("Accent.TButton", background=ACCENT, foreground=BG,
                                  font=("Segoe UI", 10, "bold"))
    s.map("Accent.TButton",    background=[("active", "#74c7ec")])
    s.configure("TEntry",      fieldbackground=PANEL, foreground=FG,
                               insertcolor=FG)
    s.configure("TCombobox",   fieldbackground=PANEL, foreground=FG,
                               selectbackground=ACCENT)
    s.configure("TNotebook",   background=BG, tabmargins=0)
    s.configure("TNotebook.Tab", background=BTN, foreground=FG,
                                 padding=[12, 5])
    s.map("TNotebook.Tab",     background=[("selected", ACCENT)],
                               foreground=[("selected", BG)])
    s.configure("TCheckbutton", background=BG, foreground=FG)
    s.configure("Treeview",    background=PANEL, foreground=FG,
                               fieldbackground=PANEL, rowheight=22)
    s.configure("Treeview.Heading", background=BTN, foreground=ACCENT,
                                    font=("Segoe UI", 10, "bold"))
    s.map("Treeview",          background=[("selected", ACCENT)],
                               foreground=[("selected", BG)])
    s.configure("TScrollbar",  background=BTN, troughcolor=PANEL,
                               arrowcolor=FG)
    s.configure("TProgressbar", troughcolor=PANEL, background=ACCENT)
    return s


def _mpl_style():
    plt.rcParams.update({
        "figure.facecolor":  BG,
        "axes.facecolor":    PANEL,
        "axes.edgecolor":    FG,
        "axes.labelcolor":   FG,
        "text.color":        FG,
        "xtick.color":       FG,
        "ytick.color":       FG,
        "grid.color":        "#45475a",
        "grid.alpha":        0.4,
        "lines.linewidth":   1.2,
    })

