# ============================================================
# src/testing/report_models.py
# ============================================================
"""Dataclasses describing the results of the smoke-test suite.

These containers hold, at increasing levels of aggregation, the outcome
of exporting structures for a single compound
(:class:`CompoundExportResult`), running one synthetic test set
(:class:`SetSmokeResult`), and the whole suite
(:class:`SmokeSuiteResult`).
"""
from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path

@dataclass
class CompoundExportResult:
    """Outcome of structure generation/export for one assigned compound.

    Attributes
    ----------
    compound_index : int
        Row index of the compound within the result table.
    mass : float
        Neutral/observed mass of the compound, in Da.
    brutto : str
        Assigned brutto (molecular) formula.
    n_cooh : int
        Number of carboxyl (-COOH) groups inferred from the D-shift.
    n_oh : int
        Number of hydroxyl (-OH) groups inferred from the D-shift.
    structures_found : int
        Number of candidate structures generated.
    structure_paths : list of pathlib.Path
        Paths to exported structure files (images/mol blocks).
    error : str or None
        Error message if export failed, else ``None``.
    """
    compound_index: int
    mass: float
    brutto: str
    n_cooh: int
    n_oh: int
    structures_found: int
    structure_paths: List[Path] = field(default_factory=list)
    error: Optional[str] = None

@dataclass
class SetSmokeResult:
    """Outcome of running the pipeline on one synthetic test set.

    Attributes
    ----------
    set_name : str
        Name of the test set (e.g. ``"set_01"``).
    success : bool
        ``True`` if the whole set (pipeline + exports) completed.
    error : str or None
        Error message if the set failed, else ``None``.
    pipeline_success : bool
        ``True`` if the core pipeline stage completed without error.
    result_table_path : pathlib.Path or None
        Path to the exported result table (CSV).
    spectra_plot_path : pathlib.Path or None
        Path to the combined three-spectra plot.
    series_dmet_path : pathlib.Path or None
        Path to the deuteromethylation (-COOH) series grid image.
    series_dacet_path : pathlib.Path or None
        Path to the deuteroacylation (-OH) series grid image.
    hist_cooh_path : pathlib.Path or None
        Path to the -COOH count histogram.
    hist_oh_path : pathlib.Path or None
        Path to the -OH count histogram.
    compound_results : list of CompoundExportResult
        Per-compound export results.
    artifacts_dir : pathlib.Path or None
        Directory holding all artifacts for this set.
    """
    set_name: str
    success: bool = False          # <-- default
    error: Optional[str] = None
    pipeline_success: bool = False  # <-- default
    result_table_path: Optional[Path] = None
    spectra_plot_path: Optional[Path] = None
    series_dmet_path: Optional[Path] = None
    series_dacet_path: Optional[Path] = None
    hist_cooh_path: Optional[Path] = None
    hist_oh_path: Optional[Path] = None
    compound_results: List[CompoundExportResult] = field(default_factory=list)
    artifacts_dir: Optional[Path] = None

@dataclass
class SmokeSuiteResult:
    """Aggregated result of running the full smoke-test suite.

    Attributes
    ----------
    sets : list of SetSmokeResult
        Per-set results.
    overall_success : bool
        ``True`` only if every set succeeded.
    started_at : str
        ISO timestamp when the suite started.
    finished_at : str
        ISO timestamp when the suite finished.
    """
    sets: List[SetSmokeResult] = field(default_factory=list)
    overall_success: bool = False
    started_at: str = ""
    finished_at: str = ""