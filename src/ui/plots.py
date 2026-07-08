# ui/plots.py
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.pyplot as plt
from tkinter import ttk

# ── Утилиты встраивания Matplotlib ────────────────────────────────────────


def embed_figure(fig, parent, toolbar=True):
    """Embed a Matplotlib figure into a Tkinter container widget.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to render.
    parent : tkinter.Widget
        Container that will host the figure canvas.
    toolbar : bool, optional
        If ``True`` (default) add an interactive navigation toolbar
        below the canvas.

    Returns
    -------
    None
        The canvas (and optional toolbar) are packed into ``parent``.
    """
    canvas = FigureCanvasTkAgg(fig, master=parent)
    canvas.draw()
    if toolbar:
        tb = NavigationToolbar2Tk(canvas, parent, pack_toolbar=False)
        tb.update()
        tb.pack(side="bottom", fill="x")
    canvas.get_tk_widget().pack(fill="both", expand=True)


def clear_canvas(parent: ttk.Frame):
    """Remove all child widgets from a frame and close open figures.

    Parameters
    ----------
    parent : tkinter.ttk.Frame
        Frame whose children will be destroyed.

    Returns
    -------
    None
        Destroys every child widget of ``parent`` and closes all
        Matplotlib figures to free memory.
    """
    for widget in parent.winfo_children():
        widget.destroy()
    plt.close("all")
