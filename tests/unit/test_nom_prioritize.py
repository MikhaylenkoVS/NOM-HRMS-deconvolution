"""Test NOM prioritization with real test-set data."""

import pandas as pd
from tests.conftest import TEST_SETS_ROOT
from nomspectra.spectrum import Spectrum
from src.core.spectrum_ops import (
    assign_formulas_simple,
    FormulaSearchConfig,
    _nom_distance,
    _generate_candidate_formulas,
    _neutral_to_ion_mass,
)
from src.configs import CHEM
from pathlib import Path


def test_nom_distance_known_values():
    """Lignin-like (H/C~1.1, O/C~0.3) should be closer than condensed aromatics."""
    dist_lignin = _nom_distance(1.1, 0.3)
    dist_aromatics = _nom_distance(0.3, 0.1)
    assert dist_lignin < dist_aromatics


def test_nom_prioritize_on_set_01():
    """NOM-приоритизация (теперь всегда включена) корректно назначает формулы."""
    df = pd.read_csv(TEST_SETS_ROOT / "set_01" / "original.csv")
    cfg = FormulaSearchConfig(
        elements=("C", "H", "O", "N"),
        ranges={"C": (1, 50), "H": (4, 100), "O": (0, 20), "N": (0, 6)},
    )

    # NOM-приоритизация включена по умолчанию
    src = Spectrum(table=df)
    result = assign_formulas_simple(
        src,
        rel_error_ppm=0.5,
        mass_min=0,
        mass_max=1000,
        search_config=cfg,
        ion_mode="[M-H]-",
        brutto_generation_mode="nom_like",
        nom_weight=1.0,
    )
    n_assigned = result.table["assign"].sum()

    print(f"NOM-prioritized: {n_assigned} assigned peaks")
    assert n_assigned > 0
