# ui/plots.py
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.pyplot as plt
from tkinter import ttk

# ── Утилиты встраивания Matplotlib ────────────────────────────────────────

def embed_figure(self, fig, parent: ttk.Frame, toolbar: bool = True):
    canvas = FigureCanvasTkAgg(fig, master=parent)
    canvas.draw()
    if toolbar:
        tb = NavigationToolbar2Tk(canvas, parent, pack_toolbar=False)
        tb.update()
        tb.pack(side="bottom", fill="x")
    canvas.get_tk_widget().pack(fill="both", expand=True)

def clear_canvas(self, parent: ttk.Frame):
    for widget in parent.winfo_children():
        widget.destroy()
    plt.close("all")