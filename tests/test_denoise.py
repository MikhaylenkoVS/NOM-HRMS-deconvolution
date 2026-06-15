from pathlib import Path
import pandas as pd
import pytest
from src.core.spectrum_ops import load_spectrum, denoise


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent

TEST_SETS = [
    PROJECT_ROOT / "data" / "test_sets" / "set_01",
    PROJECT_ROOT / "data" / "test_sets" / "set_02",
    PROJECT_ROOT / "data" / "test_sets" / "set_03",
    PROJECT_ROOT / "data" / "test_sets" / "set_04",
    PROJECT_ROOT / "data" / "test_sets" / "set_05",
]

@pytest.mark.parametrize("set_dir", TEST_SETS)
def test_denoise_original_preserves_signals_and_reduces_noise(set_dir: Path):
    src_path = set_dir / "original.csv"
    ann_path = set_dir / "annotations.csv"

    src = load_spectrum(src_path, mass_min=100, mass_max=1000)
    ann = pd.read_csv(ann_path)

    ann_orig = ann[ann["spectrum_type"] == "original"].copy()
    ann_signal = ann_orig[ann_orig["is_signal"]].copy()
    ann_noise = ann_orig[~ann_orig["is_signal"]].copy()

    rel_error_ppm = 0.5

    denoised = denoise(src, force = 10.0, intensity = 100, quantile = None)  # текущий denoise из nomspectra / твоей обёртки
    den_df = denoised.table.copy()

    # 1. recall по сигналам
    kept_signals = 0
    for _, row in ann_signal.iterrows():
        mass_obs = row["mass_obs"]
        diff_ppm = (den_df["mass"] - mass_obs) / mass_obs * 1e6
        if any(diff_ppm.abs() <= rel_error_ppm + 1e-6):
            kept_signals += 1

    signal_recall = kept_signals / len(ann_signal)

    # 2. retention по шуму
    kept_noise = 0
    for _, row in ann_noise.iterrows():
        mass_obs = row["mass_obs"]
        diff_ppm = (den_df["mass"] - mass_obs) / mass_obs * 1e6
        if any(diff_ppm.abs() <= rel_error_ppm + 1e-6):
            kept_noise += 1

    noise_retention = kept_noise / len(ann_noise)

    print(f"[{set_dir.name}] signal_recall={signal_recall:.3f}")
    print(f"[{set_dir.name}] noise_retention={noise_retention:.3f}")

    assert signal_recall >= 0.90, f"{set_dir.name}: denoise удаляет слишком много сигналов"
    assert noise_retention <= 0.80, f"{set_dir.name}: denoise слишком слабо чистит шум"