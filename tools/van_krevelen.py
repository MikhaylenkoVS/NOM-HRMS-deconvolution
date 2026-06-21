#!/usr/bin/env python3
"""
van_krevelen_plot.py

Create a Van Krevelen diagram from a result table (result_table.csv).
Expected input columns (semicolon‑delimited):
    mass, intensity, brutto, N_COOH, N_OH_total, N_OH, missing_dmet, missing_dacet

Output: van_krevelen.png (300 dpi) with compound class regions and point size
        proportional to intensity, coloured by number of carboxyl groups.
"""

import re
import argparse
import sys
import warnings
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
from matplotlib.collections import PatchCollection


# ----------------------------------------------------------------------
# Formula parsing
# ----------------------------------------------------------------------
def parse_formula(formula: str) -> Dict[str, int]:
    """
    Parse a molecular formula string (e.g. 'C12H18O5') into a dictionary
    {element: count}. Only elements C, H, O, N, S, P are returned.
    """
    pattern = re.compile(r'([A-Z][a-z]?)(\d*)')
    counts: Dict[str, int] = {}
    for match in pattern.finditer(formula):
        elem, num_str = match.groups()
        num = int(num_str) if num_str else 1
        counts[elem] = counts.get(elem, 0) + num
    # Keep only expected elements
    expected = {'C', 'H', 'O', 'N', 'S', 'P'}
    return {el: counts[el] for el in expected if el in counts}


# ----------------------------------------------------------------------
# NOM class regions
# ----------------------------------------------------------------------
NOM_REGIONS = [
    {
        'name': 'Lipids',
        'color': '#F4A582',   # light red/orange
        'vertices': [(0.0, 1.5), (0.3, 1.5), (0.3, 2.2), (0.0, 2.2)]
    },
    {
        'name': 'Proteins',
        'color': '#92C5DE',   # light blue
        'vertices': [(0.3, 1.5), (0.55, 1.5), (0.55, 2.2), (0.3, 2.2)]
    },
    {
        'name': 'Carbohydrates',
        'color': '#B2ABD2',   # light purple
        'vertices': [(0.6, 1.5), (1.2, 1.5), (1.2, 2.2), (0.6, 2.2)]
    },
    {
        'name': 'Lignin',
        'color': '#A6D96A',   # light green
        'vertices': [(0.1, 0.7), (0.45, 0.7), (0.45, 1.5), (0.1, 1.5)]
    },
    {
        'name': 'Tannins',
        'color': '#FDAE61',   # light orange
        'vertices': [(0.5, 0.5), (0.9, 0.5), (0.9, 1.5), (0.5, 1.5)]
    },
    {
        'name': 'Condensed aromatics\n(black carbon)',
        'color': '#B3B3B3',   # light grey
        'vertices': [(0.0, 0.2), (0.2, 0.2), (0.2, 0.7), (0.0, 0.7)]
    },
]


# ----------------------------------------------------------------------
# Main plot creation
# ----------------------------------------------------------------------
def create_van_krevelen_plot(
    input_path: str = r'C:\Users\mikha\PycharmProjects\AnaliticsSpectra\src\result_table.csv',
    output_path: str = r'C:\Users\mikha\PycharmProjects\AnaliticsSpectra\src\van_krevelen.png'
) -> None:
    """
    Read the result table, compute elemental ratios, and produce a
    Van Krevelen diagram.

    Parameters
    ----------
    input_path : str
        Path to the semicolon‑separated CSV file.
    output_path : str
        Path where the PNG image will be saved (300 dpi).
    """
    # --- 1. Load data ---
    try:
        df = pd.read_csv(input_path, sep=';', encoding='utf-8-sig')
    except FileNotFoundError:
        print(f"Error: file '{input_path}' not found.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error reading CSV: {e}", file=sys.stderr)
        sys.exit(1)

    required = ['mass', 'intensity', 'brutto', 'N_COOH', 'N_OH_total', 'N_OH',
                'missing_dmet', 'missing_dacet']
    missing = [col for col in required if col not in df.columns]
    if missing:
        print(f"Error: missing required columns: {missing}", file=sys.stderr)
        print(f"Available columns: {list(df.columns)}", file=sys.stderr)
        sys.exit(1)

    # --- 2. Parse formulas and compute ratios ---
    h_c = []
    o_c = []
    n_cooh = []
    intensities = []
    skipped = 0

    for _, row in df.iterrows():
        brutto = str(row['brutto'])
        if not brutto or brutto.lower() == 'nan':
            continue

        counts = parse_formula(brutto)
        c = counts.get('C', 0)
        h = counts.get('H', 0)
        o = counts.get('O', 0)

        if c == 0:
            warnings.warn(f"Skipping row: zero carbon atoms in formula '{brutto}'")
            skipped += 1
            continue

        h_c.append(h / c)
        o_c.append(o / c)
        n_cooh.append(int(row['N_COOH']))
        intensities.append(float(row['intensity']))

    if skipped:
        print(f"Warning: {skipped} rows skipped due to C=0.", file=sys.stderr)

    if not h_c:
        print("Error: no valid data after filtering.", file=sys.stderr)
        sys.exit(1)

    h_c_arr = np.array(h_c)
    o_c_arr = np.array(o_c)
    n_cooh_arr = np.array(n_cooh)
    intensities_arr = np.array(intensities)

    # --- 3. Normalize point sizes ---
    intensity_min = intensities_arr.min()
    intensity_max = intensities_arr.max()
    if intensity_max == intensity_min:
        sizes = np.full_like(intensities_arr, 100, dtype=float)
    else:
        sizes = 20 + (intensities_arr - intensity_min) / (intensity_max - intensity_min) * (200 - 20)

    # --- 4. Create plot ---
    fig, ax = plt.subplots(figsize=(10, 8))

    # Draw NOM class regions
    for region in NOM_REGIONS:
        poly = Polygon(region['vertices'], closed=True,
                       facecolor=region['color'], edgecolor='none',
                       alpha=0.12)
        ax.add_patch(poly)

        # Compute centroid for label placement
        verts = np.array(region['vertices'])
        cx, cy = verts.mean(axis=0)
        ax.text(cx, cy, region['name'],
                ha='center', va='center', fontsize=9,
                color='black', alpha=0.7, weight='bold')

    # Scatter points
    sc = ax.scatter(o_c_arr, h_c_arr, c=n_cooh_arr, cmap='YlOrRd',
                    s=sizes, edgecolor='k', linewidth=0.3, alpha=0.85)

    # Colorbar
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('Number of –COOH groups')

    # Axis limits and labels
    ax.set_xlim(0.0, 1.2)
    ax.set_ylim(0.0, 2.5)
    ax.set_xlabel('O/C atomic ratio')
    ax.set_ylabel('H/C atomic ratio')
    ax.set_title('Van Krevelen Diagram')

    # Grid and layout
    ax.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()

    # --- 5. Save ---
    fig.savefig(output_path, dpi=300)
    print(f"Van Krevelen plot saved to '{output_path}'")
    plt.close(fig)


# ----------------------------------------------------------------------
# Command‑line interface
# ----------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a Van Krevelen diagram from a result table."
    )
    parser.add_argument(
        '--input', '-i', default=r'C:\Users\mikha\PycharmProjects\AnaliticsSpectra\src\result_table.csv',
        help='Path to input CSV file (semicolon delimited). Default: result_table.csv'
    )
    parser.add_argument(
        '--output', '-o', default=r'C:\Users\mikha\PycharmProjects\AnaliticsSpectra\src\van_krevelen.png',
        help='Path to output PNG file. Default: van_krevelen.png'
    )
    args = parser.parse_args()

    create_van_krevelen_plot(input_path=args.input, output_path=args.output)


if __name__ == '__main__':
    main()