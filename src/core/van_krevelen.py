"""
src/core/van_krevelen.py
========================
Van Krevelen diagram builder for NOM (Natural Organic Matter) compound
classification, used both as a pipeline step and in the GUI.

Typical usage::

    from src.core.van_krevelen import create_van_krevelen_plot

    fig = create_van_krevelen_plot(result_df)
    fig.savefig("output.png", dpi=300)
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Polygon
from matplotlib import patheffects

from src.core.molecule import parse_formula

# ======================================================================
# ПАРАМЕТРЫ ПОСТРОЕНИЯ ГРАФИКА
# Все константы ниже можно безопасно редактировать для настройки
# внешнего вида диаграммы Ван Кревелена.
# ======================================================================

# --- Размер фигуры и DPI по умолчанию ---
FIGURE_FIGSIZE: tuple[float, float] = (10, 8)
FIGURE_DEFAULT_DPI: float = 300

# --- Пределы осей ---
X_LIM: tuple[float, float] = (0.0, 1.2)
Y_LIM: tuple[float, float] = (0.0, 2.5)

# --- Подписи осей и заголовок ---
X_LABEL: str = "O/C atomic ratio"
Y_LABEL: str = "H/C atomic ratio"
TITLE: str = "Van Krevelen Diagram"

# --- Параметры сетки ---
GRID_ENABLED: bool = True
GRID_STYLE: str = "--"
GRID_ALPHA: float = 0.25
GRID_COLOR: str = "#cccccc"

# --- Параметры точек (scatter plot) ---
SCATTER_CMAP: str = "YlOrRd"
# Альтернативные цветовые гаммы: для COOH (тёплая) и OH (холодная)
SCATTER_CMAP_COOH: str = "YlOrRd"   # жёлто-оранжево-красная
SCATTER_CMAP_OH: str = "YlGnBu"     # жёлто-зелёно-синяя
SCATTER_EDGECOLOR: str = "#444444"
SCATTER_LINEWIDTH: float = 0.5
SCATTER_ALPHA: float = 0.85
# Размер точки для минимальной и максимальной интенсивности
SCATTER_SIZE_MIN: float = 50.0
SCATTER_SIZE_MAX: float = 300.0
# Размер по умолчанию, когда все интенсивности равны
SCATTER_SIZE_FALLBACK: float = 120.0

# --- Параметры colorbar ---
COLORBAR_LABEL: str = "Number of –COOH groups"

# --- Параметры областей NOM ---
NOM_REGION_ALPHA: float = 0.22
NOM_LABEL_FONTSIZE: float = 10
NOM_LABEL_ALPHA: float = 0.85
NOM_LABEL_WEIGHT: str = "bold"
NOM_LABEL_COLOR: str = "#1a1a1a"    # почти чёрный на белом фоне

# ======================================================================
# ОБЛАСТИ NOM НА ДИАГРАММЕ
# Каждая область задаётся словарём с ключами:
#   name     — подпись (будет отображена в центре области)
#   color    — цвет заливки (hex или named)
#   vertices — список (x, y) вершин многоугольника
# ======================================================================
NOM_REGIONS: list[dict] = [
    {
        "name": "Lipids",
        "color": "#E8A87C",  # насыщенный персиковый
        "vertices": [(0.0, 1.5), (0.3, 1.5), (0.3, 2.2), (0.0, 2.2)],
    },
    {
        "name": "Proteins",
        "color": "#76B5D4",  # насыщенный голубой
        "vertices": [(0.3, 1.5), (0.55, 1.5), (0.55, 2.2), (0.3, 2.2)],
    },
    {
        "name": "Carbohydrates",
        "color": "#9B8EC4",  # насыщенный фиолетовый
        "vertices": [(0.6, 1.5), (1.2, 1.5), (1.2, 2.2), (0.6, 2.2)],
    },
    {
        "name": "Lignin",
        "color": "#7BC86C",  # насыщенный зелёный
        "vertices": [(0.1, 0.7), (0.45, 0.7), (0.45, 1.5), (0.1, 1.5)],
    },
    {
        "name": "Tannins",
        "color": "#F0A45A",  # насыщенный оранжевый
        "vertices": [(0.5, 0.5), (0.9, 0.5), (0.9, 1.5), (0.5, 1.5)],
    },
    {
        "name": "Condensed aromatics\n(black carbon)",
        "color": "#A0A0A0",  # насыщенный серый
        "vertices": [(0.0, 0.2), (0.2, 0.2), (0.2, 0.7), (0.0, 0.7)],
    },
]


# ======================================================================
# Вычисление данных для диаграммы
# ======================================================================


def compute_van_krevelen_data(df: pd.DataFrame) -> pd.DataFrame:
    """Compute H/C and O/C ratios from a result table.

    Parameters
    ----------
    df : pandas.DataFrame
        Result table. Must contain columns ``mass``, ``intensity``, ``brutto``
        and ``N_COOH``.

    Returns
    -------
    pandas.DataFrame
        Table with columns ``h_c``, ``o_c``, ``n_cooh``, ``intensity``,
        ``mass``, ``brutto``. Rows with unparsable or zero-carbon formulas
        are silently dropped (a warning is issued for each).

    Raises
    ------
    ValueError
        If no valid data remains after filtering.
    """
    required = ["mass", "intensity", "brutto", "N_COOH"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            f"Result table is missing required columns: {missing}. "
            f"Available: {list(df.columns)}"
        )

    h_c_list: list[float] = []
    o_c_list: list[float] = []
    n_cooh_list: list[int] = []
    intensities_list: list[float] = []
    masses_list: list[float] = []
    brutto_list: list[str] = []
    skipped = 0

    for _, row in df.iterrows():
        brutto = str(row["brutto"])
        if not brutto or brutto.lower() in ("nan", "none", ""):
            continue

        counts = parse_formula(brutto)
        c = counts.get("C", 0)
        h = counts.get("H", 0)
        o = counts.get("O", 0)

        if c == 0:
            warnings.warn(f"Skipping row: zero carbon atoms in formula '{brutto}'")
            skipped += 1
            continue

        h_c_list.append(h / c)
        o_c_list.append(o / c)
        n_cooh_list.append(int(row["N_COOH"]))
        intensities_list.append(float(row["intensity"]))
        masses_list.append(float(row["mass"]))
        brutto_list.append(brutto)

    if skipped:
        warnings.warn(f"{skipped} row(s) skipped due to zero carbon.", stacklevel=2)

    if not h_c_list:
        raise ValueError(
            "No valid data after filtering — all formulas have zero carbon."
        )

    return pd.DataFrame(
        {
            "h_c": h_c_list,
            "o_c": o_c_list,
            "n_cooh": n_cooh_list,
            "intensity": intensities_list,
            "mass": masses_list,
            "brutto": brutto_list,
        }
    )


# ======================================================================
# Построение диаграммы
# ======================================================================


def create_van_krevelen_plot(
    df: pd.DataFrame,
    output_path: Optional[str | Path] = None,
    *,
    color_by: str = "N_COOH",
    figsize: tuple[float, float] = FIGURE_FIGSIZE,
    dpi: float = FIGURE_DEFAULT_DPI,
) -> plt.Figure:
    """Build a Van Krevelen diagram from a result table.

    Parameters
    ----------
    df : pandas.DataFrame
        Result table. Must contain columns ``mass``, ``intensity``, ``brutto``
        and either ``N_COOH`` or ``N_OH`` (depending on *color_by*).
    output_path : str or Path or None, optional
        If given, the figure is saved to this path (300 DPI).
    color_by : {'N_COOH', 'N_OH'}, optional
        Column to colour points by. Default ``'N_COOH'``.
    figsize : tuple of float, optional
        Figure size in inches. Default ``(10, 8)``.
    dpi : float, optional
        Save DPI when ``output_path`` is used. Default ``300``.

    Returns
    -------
    matplotlib.figure.Figure
        The figure object. The caller is responsible for closing it.
    """
    color_by = color_by.strip()
    if color_by not in ("N_COOH", "N_OH"):
        color_by = "N_COOH"

    cmap = SCATTER_CMAP_COOH if color_by == "N_COOH" else SCATTER_CMAP_OH
    clabel = "Number of –COOH groups" if color_by == "N_COOH" else "Number of –OH groups"

    # Если df уже содержит колонки h_c, o_c — используем как есть,
    # иначе вычисляем из сырой result-таблицы.
    if "h_c" in df.columns and "o_c" in df.columns:
        data = df
    else:
        data = compute_van_krevelen_data(df)

    h_c_arr = np.asarray(data["h_c"], dtype=float)
    o_c_arr = np.asarray(data["o_c"], dtype=float)
    # Разрешаем обе колонки: n_cooh (историческая) и прямые N_COOH / N_OH
    color_col = color_by.lower()
    if color_col in data.columns:
        color_arr = np.asarray(data[color_col], dtype=float)
    elif color_col == "n_cooh" and "N_COOH" not in data.columns:
        color_arr = np.asarray(data["n_cooh"], dtype=float)
    else:
        # fallback: возьмём N_COOH
        color_arr = np.asarray(data["n_cooh"], dtype=float)
    intensities_arr = np.asarray(data["intensity"], dtype=float)

    # ── Нормализация размера точек (sqrt-шкала) ──────────────────────
    intensity_min = intensities_arr.min()
    intensity_max = intensities_arr.max()
    if intensity_max == intensity_min:
        sizes = np.full_like(intensities_arr, SCATTER_SIZE_FALLBACK, dtype=float)
    else:
        # sqrt для сглаживания: мелкие пики не исчезают на фоне одного гиганта
        norm = np.sqrt((intensities_arr - intensity_min) / (intensity_max - intensity_min))
        sizes = SCATTER_SIZE_MIN + norm * (SCATTER_SIZE_MAX - SCATTER_SIZE_MIN)

    # ── Создание фигуры ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=figsize)
    # Белый фон диаграммы
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # ── Области NOM ───────────────────────────────────────────────────
    for region in NOM_REGIONS:
        poly = Polygon(
            region["vertices"],
            closed=True,
            facecolor=region["color"],
            edgecolor="none",
            alpha=NOM_REGION_ALPHA,
        )
        ax.add_patch(poly)

        verts = np.array(region["vertices"])
        cx, cy = verts.mean(axis=0)
        ax.text(
            cx,
            cy,
            region["name"],
            ha="center",
            va="center",
            fontsize=NOM_LABEL_FONTSIZE,
            color=NOM_LABEL_COLOR,
            alpha=NOM_LABEL_ALPHA,
            weight=NOM_LABEL_WEIGHT,
            path_effects=[
                patheffects.withStroke(linewidth=2.5, foreground="white")
            ],
        )

    # ── Точки ─────────────────────────────────────────────────────────
    sc = ax.scatter(
        o_c_arr,
        h_c_arr,
        c=color_arr,
        cmap=cmap,
        s=sizes,
        edgecolor=SCATTER_EDGECOLOR,
        linewidth=SCATTER_LINEWIDTH,
        alpha=SCATTER_ALPHA,
    )

    # ── Colorbar ──────────────────────────────────────────────────────
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label(clabel, color="#333333")
    cbar.ax.yaxis.set_tick_params(color="#333333")
    plt.setp(plt.getp(cbar.ax, "yticklabels"), color="#333333")

    # ── Оси ───────────────────────────────────────────────────────────
    ax.set_xlim(*X_LIM)
    ax.set_ylim(*Y_LIM)
    ax.set_xlabel(X_LABEL, color="#222222", fontsize=12)
    ax.set_ylabel(Y_LABEL, color="#222222", fontsize=12)
    ax.set_title(TITLE, color="#222222", fontsize=14, weight="bold")
    ax.tick_params(colors="#222222", labelsize=10)
    for spine in ax.spines.values():
        spine.set_color("#444444")

    # ── Сетка ─────────────────────────────────────────────────────────
    if GRID_ENABLED:
        ax.grid(True, linestyle=GRID_STYLE, alpha=GRID_ALPHA, color=GRID_COLOR)

    fig.tight_layout()

    # ── Сохранение ────────────────────────────────────────────────────
    if output_path:
        fig.savefig(str(output_path), dpi=dpi)
        print(f"Van Krevelen plot saved to '{output_path}'")

    return fig
