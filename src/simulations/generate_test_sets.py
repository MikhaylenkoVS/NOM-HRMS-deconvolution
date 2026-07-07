"""Generate synthetic MS test sets for NOM-like mixtures.

Builds the ``original``/``deutermethylated``/``deuteroacylated`` spectra
for each ``set_01``..``set_05`` from a per-set ``molecules.csv``, applies a
configurable ppm mass error and additive noise, and writes the spectra and
annotation tables under ``data/test_sets``.

Notes
-----
The per-group mass shifts are sourced from
``CHEM.derivatization_shifts``, which is the single source of truth also
used by the analysis pipeline (``DELTA_CD3`` = 17.03448 Da,
``DELTA_CD3CO`` = 45.02939 Da).
"""

from pathlib import Path
from typing import Dict, Any, List
import json
import csv
from typing import Dict  # уже есть через Any, List — но пусть будет явно
import pandas as pd
import math
import random

from src.configs import CHEM, PATHS


# Корень подпроекта генератора: AnalyticsSpectra/Генерация тестовых спектров
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Базовые пути данных внутри подпроекта (относительные пути из paths.json).
DATA_ROOT = PROJECT_ROOT / PATHS.data_dir
TEST_SETS_ROOT = PROJECT_ROOT / PATHS.test_sets_dir

# Monoisotopic element masses (single source of truth: chemistry.json).
element_masses: Dict[str, float] = dict(CHEM.monoisotopic_masses)


def parse_formula(formula: str) -> Dict[str, int]:
    """Parse a simple brutto formula such as ``C7H6O5``.

    Parameters
    ----------
    formula : str
        Molecular formula without brackets or charges.

    Returns
    -------
    dict of {str: int}
        Mapping of element symbol to atom count.

    Raises
    ------
    ValueError
        If the formula is empty or cannot be fully parsed.
    """

    import re

    if not formula:
        raise ValueError("Пустая формула")

    pattern = re.compile(r"([A-Z][a-z]?)(\d*)")
    pos = 0
    composition: Dict[str, int] = {}

    for match in pattern.finditer(formula):
        element, count_str = match.groups()
        count = int(count_str) if count_str else 1
        composition[element] = composition.get(element, 0) + count
        pos = match.end()

    # Если мы не дошли до конца строки формулы — остались непарсенные символы
    if pos != len(formula):
        raise ValueError(f"Не удалось полностью распарсить формулу: {formula}")

    return composition


def exact_mass_from_formula(formula: str) -> float:
    """Compute the monoisotopic mass of a brutto formula.

    Parameters
    ----------
    formula : str
        Molecular formula without brackets or charges.

    Returns
    -------
    float
        Monoisotopic mass in Da, using :data:`element_masses`.

    Raises
    ------
    ValueError
        If the formula contains an element absent from
        :data:`element_masses`.
    """

    composition = parse_formula(formula)
    mass = 0.0
    for element, count in composition.items():
        if element not in element_masses:
            raise ValueError(f"Неизвестный элемент в формуле {formula}: {element}")
        mass += element_masses[element] * count
    return mass


def load_molecules_for_set(set_path: Path) -> pd.DataFrame:
    """Load the ``molecules.csv`` table for one test set.

    Parameters
    ----------
    set_path : pathlib.Path
        Directory of the test set.

    Returns
    -------
    pandas.DataFrame
        Contents of ``molecules.csv``.
    """
    df = pd.read_csv(set_path / PATHS.spectrum_files["molecules"])
    return df


def init_test_sets_structure() -> None:
    """Create the base directory layout for the test sets.

    Returns
    -------
    None
        Creates ``set_01``..``set_05`` folders under
        :data:`TEST_SETS_ROOT`.
    """

    for i in range(1, PATHS.num_test_sets + 1):
        set_dir = TEST_SETS_ROOT / f"set_{i:02d}"
        set_dir.mkdir(parents=True, exist_ok=True)


def load_or_create_config(set_dir: Path) -> Dict[str, Any]:
    """Load a test set's ``config.json``, creating a default if absent.

    Parameters
    ----------
    set_dir : pathlib.Path
        Directory of the test set.

    Returns
    -------
    dict
        Parsed configuration (mass range, ppm-error model, intensity,
        noise, derivatization shifts and adducts).
    """

    config_path = set_dir / "config.json"

    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    # Дефолтная конфигурация (значения из chemistry.json для точных сдвигов)
    _dm_shift = CHEM.derivatization_shifts["delta_cd3"]
    _da_shift = CHEM.derivatization_shifts["delta_cd3co"]
    default_config: Dict[str, Any] = {
        "set_id": set_dir.name,
        "mass_range": {
            "min": 100.0,
            "max": 1000.0,
        },
        "ppm_error": {
            "type": "normal",
            "mean": 0.0,
            "std": 2.0,
            "max_abs": 5.0,
        },
        "intensity": {
            "scale": "linear",
            "max_intensity": 100000.0,
        },
        "noise": {
            "peak_count": 500,
            "intensity_fraction_max": 0.1,
        },
        "derivatization": {
            "deutermethyl": {
                "target_groups": ["COOH"],
                "mass_shift_per_group": _dm_shift,
                "label": "CD3",
                "conversion_yield": 1.0,
            },
            "deuteroacyl": {
                "target_groups": ["OH"],
                "mass_shift_per_group": _da_shift,
                "label": "CD3CO",
                "conversion_yield": 1.0,
            },
        },
        "adducts": {
            "pos": ["[M+H]+"],
            "neg": [CHEM.default_ion_mode],
        },
        "rounding": {
            "mass_decimals": 5,
            "intensity_decimals": 1,
        },
    }

    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(default_config, f, ensure_ascii=False, indent=2)

    return default_config


def generate_all_test_sets(overwrite: bool = False) -> None:
    """Generate every test set (``set_01``..``set_05``).

    Parameters
    ----------
    overwrite : bool, optional
        If ``True`` existing files in the sets may be overwritten.
        Default ``False``.

    Returns
    -------
    None
    """

    for i in range(1, PATHS.num_test_sets + 1):
        set_id = f"set_{i:02d}"
        generate_single_test_set(set_id=set_id, overwrite=overwrite)


def generate_spectra_for_set(
    set_id: str,
    molecules: List[Dict[str, Any]],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Build original/deutermethylated/deuteroacylated spectra for a set.

    Parameters
    ----------
    set_id : str
        Identifier of the test set (e.g. ``"set_01"``).
    molecules : list of dict
        Molecule records (from ``molecules.csv``).
    config : dict
        Set configuration; per-group mass shifts come from
        ``config["derivatization"]``.

    Returns
    -------
    dict
        Mapping ``spectrum_type -> list of peak records``. The original
        spectrum has one peak per molecule; the derivatized spectra add
        one peak per derivatization degree (0..n) for -COOH and -OH
        respectively.

    Notes
    -----
    This is the initial implementation; it is superseded later in the
    module by a version that also injects noise peaks.
    """

    spectra: Dict[str, list[Dict[str, Any]]] = {
        "original": [],
        "deutermethylated": [],
        "deuteroacylated": [],
    }

    base_intensity = 1000

    der_cfg = config.get("derivatization", {})
    dm_cfg = der_cfg.get("deutermethyl", {})
    da_cfg = der_cfg.get("deuteroacyl", {})

    dm_shift_per_group = dm_cfg.get(
        "mass_shift_per_group", CHEM.derivatization_shifts["delta_cd3"]
    )
    da_shift_per_group = da_cfg.get(
        "mass_shift_per_group", CHEM.derivatization_shifts["delta_cd3co"]
    )

    def _to_int_or_zero(value):
        if value in (None, ""):
            return 0
        if isinstance(value, int):
            return value
        try:
            return int(value)
        except ValueError:
            return 0

    for mol in molecules:
        formula = mol.get("formula")
        if not formula:
            continue

        # Берём mass из molecules.csv, а если её нет/NaN – считаем из формулы
        mass_from_file = mol.get("mass")
        exact_mass = None
        if mass_from_file not in (None, ""):
            try:
                exact_mass = float(mass_from_file)
            except ValueError:
                exact_mass = None

        if exact_mass is None:
            try:
                exact_mass = exact_mass_from_formula(formula)
            except ValueError:
                continue

        compound_id = mol.get("compound_id")
        compound_number = mol.get("compound_number")

        carboxyl_count = _to_int_or_zero(mol.get("carboxyl_count"))
        hydroxyl_count = _to_int_or_zero(mol.get("hydroxyl_count"))

        # ORIGINAL: один пик
        spectra["original"].append(
            {
                "set_id": set_id,
                "spectrum_type": "original",
                "compound_id": compound_id,
                "compound_number": compound_number,
                "formula": formula,
                "mass": exact_mass,
                "intensity": base_intensity,
                "derivatization_state": "none",
                "deriv_degree": 0,
            }
        )

        # DEUTEROMETHYLATED: исходный + все степени (0..nCOOH)
        spectra["deutermethylated"].append(
            {
                "set_id": set_id,
                "spectrum_type": "deutermethylated",
                "compound_id": compound_id,
                "compound_number": compound_number,
                "formula": formula,
                "mass": exact_mass,
                "intensity": base_intensity,
                "derivatization_state": "deutermethyl",
                "deriv_degree": 0,
            }
        )

        for k in range(1, carboxyl_count + 1):
            dm_mass = exact_mass + dm_shift_per_group * k
            spectra["deutermethylated"].append(
                {
                    "set_id": set_id,
                    "spectrum_type": "deutermethylated",
                    "compound_id": compound_id,
                    "compound_number": compound_number,
                    "formula": formula,
                    "mass": dm_mass,
                    "intensity": base_intensity,
                    "derivatization_state": "deutermethyl",
                    "deriv_degree": k,
                }
            )

        # DEUTEROACYLATED: исходный + все степени (0..nOH)
        spectra["deuteroacylated"].append(
            {
                "set_id": set_id,
                "spectrum_type": "deuteroacylated",
                "compound_id": compound_id,
                "compound_number": compound_number,
                "formula": formula,
                "mass": exact_mass,
                "intensity": base_intensity,
                "derivatization_state": "deuteroacyl",
                "deriv_degree": 0,
            }
        )

        for k in range(1, hydroxyl_count + 1):
            da_mass = exact_mass + da_shift_per_group * k
            spectra["deuteroacylated"].append(
                {
                    "set_id": set_id,
                    "spectrum_type": "deuteroacylated",
                    "compound_id": compound_id,
                    "compound_number": compound_number,
                    "formula": formula,
                    "mass": da_mass,
                    "intensity": base_intensity,
                    "derivatization_state": "deuteroacyl",
                    "deriv_degree": k,
                }
            )

    return spectra


def generate_single_test_set(set_id: str, overwrite: bool = False) -> None:
    """Generate one test set from its ``molecules.csv``.

    Loads (or creates) the config, computes masses if missing, generates
    theoretical spectra, applies the ppm mass error and writes the spectra
    and annotation CSVs.

    Parameters
    ----------
    set_id : str
        Identifier of the test set (e.g. ``"set_01"``).
    overwrite : bool, optional
        If ``True`` existing output files may be overwritten. Default
        ``False``.

    Returns
    -------
    None

    Raises
    ------
    FileNotFoundError
        If the set has no ``molecules.csv``.
    """

    set_dir = TEST_SETS_ROOT / set_id
    set_dir.mkdir(parents=True, exist_ok=True)

    # 1. конфиг
    config = load_or_create_config(set_dir)

    # 2. загрузка molecules.csv (единственный источник истины)
    mol_path = set_dir / PATHS.spectrum_files["molecules"]
    if not mol_path.exists():
        raise FileNotFoundError(f"Для {set_id} нет файла {mol_path}")

    df_mol = pd.read_csv(mol_path)

    # 3. при желании можно здесь не хранить mass в файле, а только использовать её в расчётах
    #    но для генерации спектров mass нам всё равно нужна -> считаем на лету, если нет
    if "mass" not in df_mol.columns:

        def safe_exact_mass(formula: str) -> float | None:
            if not isinstance(formula, str) or not formula:
                return None
            try:
                return exact_mass_from_formula(formula)
            except ValueError as e:
                print(f"[WARN] {mol_path.name}: {e}")
                return None

        df_mol["mass"] = df_mol["formula"].apply(safe_exact_mass)

    # 4. превращаем в список dict'ов для generate_spectra_for_set
    molecules = df_mol.to_dict(orient="records")

    # 5. генерация теоретических спектров (без ошибки)
    spectra_raw = generate_spectra_for_set(
        set_id=set_id, molecules=molecules, config=config
    )

    # 6. применяем mass error и добавляем его в записи (mass -> mass_obs)
    spectra_raw = generate_spectra_for_set(
        set_id=set_id, molecules=molecules, config=config
    )

    spectra_with_obs = apply_observed_mass_to_spectra(
        spectra=spectra_raw,
        config=config,
    )

    write_spectra_csv(set_dir=set_dir, spectra=spectra_with_obs, overwrite=overwrite)
    write_annotations_csv(
        set_dir=set_dir,
        spectra=spectra_with_obs,
        molecules=molecules,
        overwrite=overwrite,
    )


def generate_spectra_for_set(
    set_id: str,
    molecules: List[Dict[str, Any]],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Build original/derivatized spectra for a set, including noise.

    Parameters
    ----------
    set_id : str
        Identifier of the test set (e.g. ``"set_01"``).
    molecules : list of dict
        Molecule records (from ``molecules.csv``).
    config : dict
        Set configuration; per-group mass shifts and noise settings come
        from ``config["derivatization"]`` and ``config["noise"]``.

    Returns
    -------
    dict
        Mapping ``spectrum_type -> list of peak records``. Signal peaks
        carry ``is_signal=True``; additive noise peaks (``is_signal=False``)
        are appended to each spectrum.
    """

    spectra: Dict[str, list[Dict[str, Any]]] = {
        "original": [],
        "deutermethylated": [],
        "deuteroacylated": [],
    }

    base_intensity = 1000

    der_cfg = config.get("derivatization", {})
    dm_cfg = der_cfg.get("deutermethyl", {})
    da_cfg = der_cfg.get("deuteroacyl", {})

    dm_shift_per_group = dm_cfg.get(
        "mass_shift_per_group", CHEM.derivatization_shifts["delta_cd3"]
    )
    da_shift_per_group = da_cfg.get(
        "mass_shift_per_group", CHEM.derivatization_shifts["delta_cd3co"]
    )

    # в config target_groups есть, но на этом уровне удобнее
    # явно использовать carboxyl_count/hydroxyl_count

    def _to_int_or_zero(value):
        if value in (None, ""):
            return 0
        if isinstance(value, int):
            return value
        try:
            return int(value)
        except ValueError:
            return 0

    for mol in molecules:
        formula = mol.get("formula")
        if not formula:
            continue

        # если в mol уже есть mass, можно использовать её
        # но для надёжности посчитаем из формулы (или проверим совпадения позднее)
        try:
            exact_mass = exact_mass_from_formula(formula)
        except ValueError:
            continue

        compound_id = mol.get("compound_id")
        compound_number = mol.get("compound_number")

        carboxyl_count = _to_int_or_zero(mol.get("carboxyl_count"))
        hydroxyl_count = _to_int_or_zero(mol.get("hydroxyl_count"))

        # --- ORIGINAL ---
        spectra["original"].append(
            {
                "set_id": set_id,
                "spectrum_type": "original",
                "compound_id": compound_id,
                "compound_number": compound_number,
                "formula": formula,
                "mass": exact_mass,
                "intensity": base_intensity,
                "derivatization_state": "none",
                "deriv_degree": 0,
                "is_signal": True,
            }
        )

        # --- DEUTEROMETHYLATION (по -COOH) ---
        # исходный пик (degree 0) ОБЯЗАТЕЛЬНО остаётся и в этом спектре
        spectra["deutermethylated"].append(
            {
                "set_id": set_id,
                "spectrum_type": "deutermethylated",
                "compound_id": compound_id,
                "compound_number": compound_number,
                "formula": formula,
                "mass": exact_mass,
                "intensity": base_intensity,
                "derivatization_state": "deutermethyl",
                "deriv_degree": 0,
                "is_signal": True,
            }
        )

        for k in range(1, carboxyl_count + 1):
            dm_mass = exact_mass + dm_shift_per_group * k
            spectra["deutermethylated"].append(
                {
                    "set_id": set_id,
                    "spectrum_type": "deutermethylated",
                    "compound_id": compound_id,
                    "compound_number": compound_number,
                    "formula": formula,
                    "mass": dm_mass,
                    "intensity": base_intensity,
                    "derivatization_state": "deutermethyl",
                    "deriv_degree": k,
                    "is_signal": True,
                }
            )

        # --- DEUTEROACYLATION (по -OH) ---
        spectra["deuteroacylated"].append(
            {
                "set_id": set_id,
                "spectrum_type": "deuteroacylated",
                "compound_id": compound_id,
                "compound_number": compound_number,
                "formula": formula,
                "mass": exact_mass,
                "intensity": base_intensity,
                "derivatization_state": "deuteroacyl",
                "deriv_degree": 0,
                "is_signal": True,
            }
        )

        for k in range(1, hydroxyl_count + 1):
            da_mass = exact_mass + da_shift_per_group * k
            spectra["deuteroacylated"].append(
                {
                    "set_id": set_id,
                    "spectrum_type": "deuteroacylated",
                    "compound_id": compound_id,
                    "compound_number": compound_number,
                    "formula": formula,
                    "mass": da_mass,
                    "intensity": base_intensity,
                    "derivatization_state": "deuteroacyl",
                    "deriv_degree": k,
                    "is_signal": True,
                }
            )

    max_signal_intensity = base_intensity  # у нас она фиксирована

    for spectrum_type in ["original", "deutermethylated", "deuteroacylated"]:
        noise_peaks = generate_noise_peaks(
            config=config,
            spectrum_type=spectrum_type,
            set_id=set_id,
            max_signal_intensity=max_signal_intensity,
        )
        spectra[spectrum_type].extend(noise_peaks)

    return spectra


def write_molecules_csv(
    set_dir: Path, molecules: List[Dict[str, Any]], overwrite: bool = False
) -> None:
    """Write the ``molecules.csv`` table for a test set.

    Parameters
    ----------
    set_dir : pathlib.Path
        Directory of the test set.
    molecules : list of dict
        Molecule records; only the canonical fields are written, extra
        keys (``source``, ``pubchem_cid``, ...) are ignored.
    overwrite : bool, optional
        If ``False`` (default) an existing file is left untouched.

    Returns
    -------
    None
    """
    file_path = set_dir / PATHS.spectrum_files["molecules"]
    if file_path.exists() and not overwrite:
        return

    fieldnames = [
        "set_id",
        "compound_id",
        "compound_number",
        "name",
        "smiles",
        "inchi",
        "formula",
        "charge",
        "mode",
        "carboxyl_count",
        "hydroxyl_count",
        "other_fg",
        "nom_like_flag",
        "comment",
    ]

    file_path.parent.mkdir(parents=True, exist_ok=True)

    with file_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for mol in molecules:
            # Оставляем только те поля, которые есть в fieldnames,
            # чтобы игнорировать source, pubchem_cid и прочие служебные.
            row = {key: mol.get(key) for key in fieldnames}
            writer.writerow(row)


def write_spectra_csv(
    set_dir: Path, spectra: Dict[str, Any], overwrite: bool = False
) -> None:
    """Write the three spectrum CSVs (``mass``/``intensity`` columns).

    Parameters
    ----------
    set_dir : pathlib.Path
        Directory of the test set.
    spectra : dict
        Mapping ``spectrum_type -> list of peak records``.
    overwrite : bool, optional
        If ``False`` (default) existing files are left untouched.

    Returns
    -------
    None
        Writes ``original.csv``, ``deutermethylated.csv`` and
        ``deuteroacylated.csv``.
    """

    def _write_single_spectrum(
        filename: str, records: List[Dict[str, Any]] | None
    ) -> None:
        file_path = set_dir / filename
        if file_path.exists() and not overwrite:
            return

        file_path.parent.mkdir(parents=True, exist_ok=True)

        with file_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["mass", "intensity"])
            writer.writeheader()

            if records:
                for row in records:
                    writer.writerow(
                        {
                            "mass": row.get("mass"),
                            "intensity": row.get("intensity"),
                        }
                    )

    _write_single_spectrum(PATHS.spectrum_files["original"], spectra.get("original"))
    _write_single_spectrum(
        PATHS.spectrum_files["deutermethylated"], spectra.get("deutermethylated")
    )
    _write_single_spectrum(
        PATHS.spectrum_files["deuteroacylated"], spectra.get("deuteroacylated")
    )


def apply_observed_mass_to_spectra(
    spectra: dict[str, list[dict]],
    config: dict,
) -> dict[str, list[dict]]:
    """Apply the ppm mass error to every peak in a spectra dict.

    Parameters
    ----------
    spectra : dict of {str: list of dict}
        Mapping ``spectrum_type -> list of peak records``.
    config : dict
        Configuration providing the ``ppm_error`` model.

    Returns
    -------
    dict of {str: list of dict}
        New spectra where each record has ``mass`` set to the observed
        mass, ``mass_theor`` to the original theoretical mass, and
        ``mass_error_ppm`` to the applied error in ppm.
    """

    new_spectra: dict[str, list[dict]] = {
        "original": [],
        "deutermethylated": [],
        "deuteroacylated": [],
    }

    for spectrum_type in ["original", "deutermethylated", "deuteroacylated"]:
        records = spectra.get(spectrum_type) or []
        updated: list[dict] = []

        for rec in records:
            mass_theor = rec.get("mass")
            if mass_theor is None:
                updated.append(rec)
                continue

            mass_obs, err_ppm = apply_mass_error(mass_theor, config)

            rec_new = dict(rec)
            rec_new["mass"] = mass_obs
            rec_new["mass_theor"] = mass_theor
            rec_new["mass_error_ppm"] = err_ppm
            updated.append(rec_new)

        new_spectra[spectrum_type] = updated

    return new_spectra


def apply_mass_error(mass_theor: float, config: Dict[str, Any]) -> tuple[float, float]:
    """Add a small ppm-scale mass error to a theoretical mass.

    Parameters
    ----------
    mass_theor : float
        Theoretical (exact) mass in Da.
    config : dict
        Configuration providing the ``ppm_error`` model (``type``,
        ``mean``, ``std``, ``max_abs``).

    Returns
    -------
    mass_obs : float
        Observed mass in Da after applying the error.
    error_ppm : float
        Applied error in ppm, clamped to ``±max_abs``.
    """

    ppm_cfg = config.get("ppm_error", {})
    dist_type = ppm_cfg.get("type", "normal")
    mean_ppm = float(ppm_cfg.get("mean", 0.0))
    std_ppm = float(ppm_cfg.get("std", 2.0))
    max_abs_ppm = float(ppm_cfg.get("max_abs", 5.0))

    if dist_type == "normal":
        err = random.gauss(mean_ppm, std_ppm)
    else:
        # fallback: равномерное распределение
        err = random.uniform(-max_abs_ppm, max_abs_ppm)

    # обрежем по max_abs_ppm
    err = max(-max_abs_ppm, min(max_abs_ppm, err))

    mass_obs = mass_theor * (1.0 + err * 1e-6)
    return mass_obs, err


def generate_noise_peaks(
    config: Dict[str, Any],
    spectrum_type: str,
    set_id: str,
    max_signal_intensity: float,
) -> List[Dict[str, Any]]:
    """Generate additive noise peaks for a spectrum.

    Parameters
    ----------
    config : dict
        Configuration providing ``noise`` and ``mass_range`` settings.
    spectrum_type : str
        Spectrum the peaks belong to (``"original"`` etc.).
    set_id : str
        Identifier of the test set.
    max_signal_intensity : float
        Reference intensity; noise intensities are drawn up to
        ``intensity_fraction_max`` of this value.

    Returns
    -------
    list of dict
        Noise peak records (``is_signal=False``); empty if the configured
        peak count is non-positive.
    """

    noise_cfg = config.get("noise", {})
    peak_count = int(noise_cfg.get("peak_count", 0))
    if peak_count <= 0:
        return []

    mass_range = config.get("mass_range", {})
    mass_min = float(mass_range.get("min", 100.0))
    mass_max = float(mass_range.get("max", 1000.0))

    frac_max = float(noise_cfg.get("intensity_fraction_max", 0.1))
    noise_max_int = max_signal_intensity * frac_max

    peaks: List[Dict[str, Any]] = []

    for _ in range(peak_count):
        m = random.uniform(mass_min, mass_max)
        inten = random.uniform(0.0, noise_max_int)

        peaks.append(
            {
                "set_id": set_id,
                "spectrum_type": spectrum_type,
                "compound_id": None,
                "compound_number": None,
                "formula": None,
                "mass": m,
                "intensity": inten,
                "derivatization_state": "noise",
                "deriv_degree": None,
                "is_signal": False,
            }
        )

    return peaks


def write_annotations_csv(
    set_dir: Path,
    spectra: Dict[str, Any],
    molecules: List[Dict[str, Any]],
    overwrite: bool = False,
) -> None:
    """Write the ``annotations.csv`` ground-truth table for a set.

    Parameters
    ----------
    set_dir : pathlib.Path
        Directory of the test set.
    spectra : dict
        Mapping ``spectrum_type -> list of peak records``, where each
        record already carries ``mass`` (observed), ``mass_theor`` and
        ``mass_error_ppm``.
    molecules : list of dict
        Molecule records (currently unused but kept for symmetry).
    overwrite : bool, optional
        If ``False`` (default) an existing file is left untouched.

    Returns
    -------
    None
        Peaks are annotated as ``[M-H]-`` (charge ``-1``) and written with
        a generated ``peak_id``.
    """

    file_path = set_dir / PATHS.spectrum_files["annotations"]
    if file_path.exists() and not overwrite:
        return

    fieldnames = [
        "set_id",
        "spectrum_type",
        "peak_id",
        "mass_obs",
        "intensity",
        "mass_theor",
        "mass_error_ppm",
        "compound_id",
        "compound_number",
        "formula",
        "derivatization_state",
        "adduct_type",
        "charge",
        "assignment_confidence",
        "is_signal",
    ]

    file_path.parent.mkdir(parents=True, exist_ok=True)

    with file_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        peak_counter = 0

        for spectrum_type in ["original", "deutermethylated", "deuteroacylated"]:
            records = spectra.get(spectrum_type) or []
            for rec in records:
                peak_counter += 1

                mass_obs = rec.get("mass")
                mass_theor = rec.get("mass_theor", mass_obs)
                err_ppm = rec.get("mass_error_ppm", 0.0)
                intensity = rec.get("intensity")
                is_signal = rec.get("is_signal", True)

                writer.writerow(
                    {
                        "set_id": rec.get("set_id"),
                        "spectrum_type": rec.get("spectrum_type"),
                        "peak_id": f"P{peak_counter:05d}",
                        "mass_obs": mass_obs,
                        "intensity": intensity,
                        "mass_theor": mass_theor,
                        "mass_error_ppm": err_ppm,
                        "compound_id": rec.get("compound_id"),
                        "compound_number": rec.get("compound_number"),
                        "formula": rec.get("formula"),
                        "derivatization_state": rec.get("derivatization_state"),
                        "adduct_type": CHEM.default_ion_mode,
                        "charge": -1,
                        "assignment_confidence": "high",
                        "is_signal": bool(is_signal),
                    }
                )


def normalize_molecules_header_for_all_sets() -> None:
    """Rewrite every set's ``molecules.csv`` with a canonical header.

    Missing columns are filled with ``None`` so all sets share the same
    schema; sets without a ``molecules.csv`` are skipped with a message.

    Returns
    -------
    None
    """
    expected_header = [
        "set_id",
        "compound_id",
        "compound_number",
        "name",
        "smiles",
        "inchi",
        "formula",
        "charge",
        "mode",
        "carboxyl_count",
        "hydroxyl_count",
        "other_fg",
        "nom_like_flag",
        "comment",
    ]

    for i in range(1, PATHS.num_test_sets + 1):
        set_dir = TEST_SETS_ROOT / f"set_{i:02d}"
        molecules_path = set_dir / PATHS.spectrum_files["molecules"]
        if not molecules_path.exists():
            print(
                f"[normalize_molecules_header] Файл не найден: {molecules_path}, пропускаю"
            )
            continue

        df = pd.read_csv(molecules_path)

        data = {}
        for col in expected_header:
            data[col] = df[col] if col in df.columns else None

        df_norm = pd.DataFrame(data, columns=expected_header)
        df_norm.to_csv(molecules_path, index=False)
        print(f"[normalize_molecules_header] Обновлён header: {molecules_path}")


if __name__ == "__main__":
    generate_all_test_sets(overwrite=True)
    normalize_molecules_header_for_all_sets()
