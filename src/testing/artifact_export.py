# ============================================================
# src/testing/artifact_export.py
# ============================================================
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple
import numpy as np

# ----------------------------------------------------------------------
# Вспомогательные функции загрузки спектров для графиков
# ----------------------------------------------------------------------
def load_spectrum_csv(path: str) -> pd.DataFrame:
    """Load a mass spectrum from CSV into a ``mass``/``intensity`` frame.

    Column names are lower-cased and common aliases (``m/z``, ``mz``,
    ``i``, ``int``) are renamed to the canonical ``mass``/``intensity``.

    Parameters
    ----------
    path : str
        Path to the CSV file.

    Returns
    -------
    pandas.DataFrame
        Frame with exactly the ``mass`` (m/z, Da) and ``intensity``
        columns.

    Raises
    ------
    ValueError
        If the file does not contain recognizable mass and intensity
        columns.
    """
    df = pd.read_csv(path)
    # Нормализуем колонки
    df.columns = [c.strip().lower() for c in df.columns]
    rename = {'m/z': 'mass', 'mz': 'mass', 'i': 'intensity', 'int': 'intensity'}
    df.rename(columns={k: v for k,v in rename.items() if k in df.columns}, inplace=True)
    # Убедимся, что есть mass и intensity
    if 'mass' not in df.columns or 'intensity' not in df.columns:
        raise ValueError(f"CSV {path} не содержит mass/intensity. Колонки: {df.columns.tolist()}")
    return df[['mass', 'intensity']]

def plot_three_spectra(orig_path: str, dmet_path: str, dacet_path: str,
                       save_path: Path, title: str = "Mass Spectra Overlay"):
    """Plot original, deuteromethylated and deuteroacylated spectra.

    Draws the three spectra as stacked stem plots sharing an ``m/z`` axis
    and saves the figure to ``save_path``.

    Parameters
    ----------
    orig_path : str
        Path to the underivatized spectrum CSV.
    dmet_path : str
        Path to the deuteromethylated (-COOH → COOCD3) spectrum CSV.
    dacet_path : str
        Path to the deuteroacylated (-OH → OCOCD3) spectrum CSV.
    save_path : pathlib.Path
        Output PNG path.
    title : str, optional
        Figure super-title. Default ``"Mass Spectra Overlay"``.

    Returns
    -------
    None
        The figure is written to ``save_path`` at 150 dpi.
    """
    orig = load_spectrum_csv(orig_path)
    dmet = load_spectrum_csv(dmet_path)
    dacet = load_spectrum_csv(dacet_path)

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    labels = ['Original', 'Deuteromethylated', 'Deuteroacylated']
    for ax, df, color, label in zip(axes, [orig, dmet, dacet], colors, labels):
        ax.vlines(df['mass'], 0, df['intensity'], colors=color, linewidth=0.8, alpha=0.8)
        ax.set_ylabel('Intensity')
        ax.set_title(label, loc='left')
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel('m/z')
    fig.suptitle(title, fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

def plot_series_grid(df_series: pd.DataFrame, deriv_mz_array: np.ndarray,
                     delta: float, ppm_tol: float, label: str,
                     save_path: Path):
    """Save a grid plot of homologous series to a file.

    Wraps :func:`src.core.spectrum_ops.visualize_series`, rebuilding the
    minimal ``Spectrum`` objects it needs from the series table so the
    figure can be written headlessly.

    Parameters
    ----------
    df_series : pandas.DataFrame
        Series table with at least a ``mass_src`` column of source m/z
        values. If empty the function returns immediately.
    deriv_mz_array : numpy.ndarray
        Array of derivatized-spectrum m/z values.
    delta : float
        Mass shift per derivatized group, in Da (e.g. ``DELTA_CD3``).
    ppm_tol : float
        Matching tolerance in ppm.
    label : str
        Label describing the derivatization series.
    save_path : pathlib.Path
        Output image path.

    Returns
    -------
    None
        The figure is written to ``save_path``; nothing is saved when
        ``df_series`` is empty.
    """
    if df_series.empty:
        return
    # Используем существующую функцию из spectrum_ops с параметром save_path
    from src.core.spectrum_ops import visualize_series as _vis_series
    # Создаем временный Spectrum-объект? Но visualize_series требует Spectrum.
    # Для упрощения вызываем с пустыми Spectrum, если надо.
    # Однако функция ожидает src и deriv - Spectrum объекты. Мы можем передать None?
    # Лучше воссоздать минимальный Spectrum с нужным table.
    from nomspectra.spectrum import Spectrum
    src = Spectrum(table=pd.DataFrame({'mass': df_series['mass_src'].values,
                                       'intensity': np.ones(len(df_series))}))
    deriv = Spectrum(table=pd.DataFrame({'mass': deriv_mz_array,
                                         'intensity': np.ones(len(deriv_mz_array))}))
    _vis_series(src, deriv, df_series, delta=delta, label=label,
                ppm_tol=ppm_tol, save_path=str(save_path))

def plot_histogram(df: pd.DataFrame, column: str, save_path: Path, title: str = ""):
    """Save a histogram of the integer values in one column.

    Parameters
    ----------
    df : pandas.DataFrame
        Source table.
    column : str
        Name of the column to histogram (e.g. ``"N_COOH"`` or
        ``"N_OH"``); values are dropped of NaNs and cast to int.
    save_path : pathlib.Path
        Output PNG path.
    title : str, optional
        Plot title. Defaults to ``"Distribution of {column}"``.

    Returns
    -------
    None
        The figure is written to ``save_path``; nothing is saved when the
        column has no values.
    """
    vals = df[column].dropna().astype(int)
    if vals.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(vals, bins=range(vals.max() + 2), color='#a6e3a1', edgecolor='black', alpha=0.85)
    ax.set_xlabel(column)
    ax.set_ylabel('Count')
    ax.set_title(title or f'Distribution of {column}')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)