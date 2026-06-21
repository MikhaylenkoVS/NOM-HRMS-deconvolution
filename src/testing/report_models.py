# ============================================================
# src/testing/report_models.py
# ============================================================
from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path

@dataclass
class CompoundExportResult:
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
    sets: List[SetSmokeResult] = field(default_factory=list)
    overall_success: bool = False
    started_at: str = ""
    finished_at: str = ""