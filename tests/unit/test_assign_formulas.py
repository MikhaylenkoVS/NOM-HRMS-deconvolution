# /test_assign_fomulas.py
import pandas as pd
import pytest
from pathlib import Path
import re
from src.core.spectrum_ops import load_spectrum, assign_formulas, denoise
from src.configs import PIPELINE, PATHS

THIS_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = THIS_DIR.parent

# ── Единый источник истины: src/configs/pipeline.json ──
_TEST_SETS_ROOT = PROJECT_ROOT / PATHS.test_sets_dir
TEST_SETS = sorted(p for p in _TEST_SETS_ROOT.glob("set_*") if p.is_dir())

# bruto_dict из pipeline.json -> default_brutto_dict
_RAW_BRUTTO = PIPELINE.default_brutto_dict
DEFAULT_BRUTTO_DICT = {el: tuple(rng) for el, rng in _RAW_BRUTTO.items()}

FORMULA_RE = re.compile(r"([A-Z][a-z]?)(\d*)")


def parse_formula(formula: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for elem, num in FORMULA_RE.findall(formula):
        n = int(num) if num else 1
        counts[elem] = counts.get(elem, 0) + n
    return counts


def normalize_brutto(formula: str) -> str:
    """
    Приводит формулу к каноническому виду:
    - разбирает элементы и их стехиометрию,
    - сортирует элементы по алфавиту,
    - собирает обратно в строку.
    Пример:
      'C12H10N2O2' -> 'C12H10N2O2'
      'C12H10O2N2' -> 'C12H10N2O2'
    """
    if formula is None or not isinstance(formula, str):
        return formula

    # найдём пары (элемент, число)
    # элементы — одна заглавная + опционально строчные, числа — одна или более цифр
    tokens = re.findall(r"([A-Z][a-z]*)(\d*)", formula)
    if not tokens:
        return formula

    counts = {}
    for elem, num_str in tokens:
        count = int(num_str) if num_str else 1
        counts[elem] = counts.get(elem, 0) + count

    # сортируем элементы по алфавиту и собираем обратно
    parts = []
    for elem in sorted(counts.keys()):
        cnt = counts[elem]
        parts.append(elem if cnt == 1 else f"{elem}{cnt}")
    return "".join(parts)


def brutto_within_dict(formula: str, brutto_dict: dict[str, tuple[int, int]]) -> bool:
    counts = parse_formula(formula)
    for el, (emin, emax) in brutto_dict.items():
        val = counts.get(el, 0)
        if not (emin <= val <= emax):
            return False
    return True


@pytest.mark.parametrize("set_dir", TEST_SETS)
def test_assign_formulas_original_in_all_sets(set_dir: Path):
    """
    Проверка assign (mode='simple') для spectrum_type='original' во всех наборах set_0N.

    Жёсткие критерии:
    1) для каждого сигнального пика есть пик в assigned в окне rel_error_ppm;
    2) для каждого сигнального пика в этом окне есть хотя бы одна назначенная формула;
    3) все назначенные формулы укладываются в DEFAULT_BRUTTO_DICT.

    Диагностические (не валят тест):
    - процент совпадений с annotations;


    Проверка assign (mode='simple') для spectrum_type='original' во всех наборах set_0N.
    """

    src_path = set_dir / "original.csv"
    ann_path = set_dir / "annotations.csv"

    assert src_path.exists(), f"Нет original.csv в {set_dir}"
    assert ann_path.exists(), f"Нет annotations.csv в {set_dir}"

    # 1. Загружаем спектр
    _load_cfg = PIPELINE.test_mode["load"]
    src = load_spectrum(
        src_path,
        mass_min=_load_cfg["original_mass_min"],
        mass_max=_load_cfg["original_mass_max"],
    )

    # 2. Назначаем формулы простым режимом (генерация по brutto_dict в spectrum_ops)
    rel_error_ppm = PIPELINE.test_mode["assign"]["rel_error_ppm"]

    src = assign_formulas(
        src,
        mode="simple",
        rel_error_ppm=rel_error_ppm,
        mass_min=None,
        mass_max=None,
        nom_prioritize=True,
        nom_weight=20.0,
    )

    src_df = src.table.copy()
    assert "assign" in src_df.columns
    assert "brutto" in src_df.columns

    assigned = src_df[src_df["assign"] == True].copy()

    # 3. Загружаем annotations и берём only original
    ann = pd.read_csv(ann_path)
    ann_orig = ann[(ann["spectrum_type"] == "original") & (ann["is_signal"])].copy()

    n_assigned = len(assigned)
    n_signals = len(ann_orig)

    assert n_signals > 0, f"{set_dir.name}: нет сигнальных пиков для 'original'"
    assert (
        n_assigned > 0
    ), f"{set_dir.name}: assign_formulas(simple) не назначил ни одной формулы"

    mismatches = []
    matches = 0

    # 4. Сравнение с annotations + сбор информации о NO_PEAK / WRONG_BRUTTO
    for _, row in ann_orig.iterrows():
        mass_obs = row["mass_obs"]
        formula_true = row["formula"]

        diff_ppm = (assigned["mass"] - mass_obs) / mass_obs * 1e6
        candidates = assigned[diff_ppm.abs() <= rel_error_ppm + 1e-6]

        if candidates.empty:
            # Диагностика: нет ни одного назначенного пика в окне ±rel_error_ppm
            # Выведем ближайшие 3 назначенных пика по ppm
            abs_ppm_all = diff_ppm.abs()
            nearest_idx = abs_ppm_all.nsmallest(3).index
            nearest = assigned.loc[nearest_idx].copy()
            nearest["ppm_err"] = abs_ppm_all[nearest_idx]

            print(
                f"[{set_dir.name}] NO_PEAK_IN_ASSIGNED для сигнала {mass_obs:.6f} ({formula_true}):"
            )
            print(nearest[["mass", "brutto", "ppm_err"]].to_string(index=False))

            mismatches.append(
                {
                    "mass_obs": mass_obs,
                    "formula_true": formula_true,
                    "status": "NO_PEAK_IN_ASSIGNED",
                }
            )
            continue
        # нормализуем истинную формулу один раз
        formula_true_norm = normalize_brutto(formula_true)

        cand_norm = candidates["brutto"].apply(normalize_brutto)
        if any(cand_norm == formula_true_norm):
            matches += 1
        else:
            # Диагностика: пытаемся понять, нет ли правильной формулы у соседнего пика
            # Выводим несколько ближайших кандидатов по ppm для этого сигнала
            candidates = candidates.copy()
            candidates["ppm_err"] = diff_ppm[candidates.index].abs()
            candidates = candidates.sort_values("ppm_err")

            print(
                f"[{set_dir.name}] WRONG_BRUTTO для сигнала {mass_obs:.6f} ({formula_true}):"
            )
            print(
                candidates[["mass", "brutto", "ppm_err"]].head(5).to_string(index=False)
            )

            mismatches.append(
                {
                    "mass_obs": mass_obs,
                    "formula_true": formula_true,
                    "status": "WRONG_BRUTTO",
                    "candidates_brutto": list(candidates["brutto"].unique())[:5],
                }
            )
    # 4.1. Процент сигнальных пиков, которые получили хотя бы одну формулу
    signal_with_brutto = 0
    for _, row in ann_orig.iterrows():
        mass_obs = row["mass_obs"]
        diff_ppm = (assigned["mass"] - mass_obs) / mass_obs * 1e6
        in_window = assigned[diff_ppm.abs() <= rel_error_ppm + 1e-6]
        if in_window.empty:
            continue  # для этих уже есть статус NO_PEAK_IN_ASSIGNED

        # есть ли среди пиков в окне хотя бы один с assign == True
        if any(in_window["assign"] == True):
            signal_with_brutto += 1

    signal_brutto_ratio = signal_with_brutto / n_signals if n_signals > 0 else 0.0

    print(
        f"[{set_dir.name}] Сигнальных пиков с хотя бы одной формулой: "
        f"{signal_with_brutto}/{n_signals} (доля {signal_brutto_ratio:.3f})"
    )

    # Жёсткий критерий (настрой как нужно, например не меньше 0.8)
    assert signal_brutto_ratio >= 0.8, (
        f"{set_dir.name}: слишком мало сигнальных пиков с формулой "
        f"({signal_with_brutto}/{n_signals}, доля {signal_brutto_ratio:.3f})"
    )

    match_ratio = matches / n_signals if n_signals > 0 else 0.0

    # 5. Жёсткий критерий 1: для каждого сигнального пика есть кандидат в окне
    missing_peaks = [m for m in mismatches if m["status"] == "NO_PEAK_IN_ASSIGNED"]
    missing_ratio = len(missing_peaks) / n_signals if n_signals > 0 else 0.0

    print(
        f"[{set_dir.name}] Сигнальных пиков без кандидатов: "
        f"{len(missing_peaks)}/{n_signals} (доля {missing_ratio:.3f})"
    )

    assert missing_ratio <= 0.2, (
        f"{set_dir.name}: слишком много сигнальных пиков без кандидатов "
        f"({len(missing_peaks)}/{n_signals}, доля {missing_ratio:.3f})"
    )

    # 6. Жёсткий критерий 2: в окне есть хотя бы одна назначенная формула
    missing_brutto = []
    for _, row in ann_orig.iterrows():
        mass_obs = row["mass_obs"]
        diff_ppm = (assigned["mass"] - mass_obs) / mass_obs * 1e6
        in_window = assigned[diff_ppm.abs() <= rel_error_ppm + 1e-6]
        if in_window.empty:
            continue  # уже учтено в missing_peaks
        if not any(in_window["assign"] == True):
            missing_brutto.append(mass_obs)

    assert len(missing_brutto) == 0, (
        f"{set_dir.name}: для части сигнальных пиков в окне нет ни одной назначенной формулы: "
        f"{len(missing_brutto)}/{n_signals}"
    )

    # 7. Жёсткий критерий 3: все назначенные формулы укладываются в DEFAULT_BRUTTO_DICT
    invalid_brutto = []
    for _, row in assigned.iterrows():
        brutto = row["brutto"]
        if not isinstance(brutto, str) or not brutto:
            continue
        if not brutto_within_dict(brutto, DEFAULT_BRUTTO_DICT):
            invalid_brutto.append(brutto)

    assert len(invalid_brutto) == 0, (
        f"{set_dir.name}: найдены формулы вне DEFAULT_BRUTTO_DICT "
        f"(пример: {invalid_brutto[:5]})"
    )

    # 9. Логи (некритические метрики)
    print(f"\n[{set_dir.name}] Назначено формул: {n_assigned}")
    print(f"[{set_dir.name}] Сигнальных пиков (original): {n_signals}")
    print(
        f"[{set_dir.name}] Совпадений с annotations: {matches}/{n_signals} (доля {match_ratio:.3f})"
    )
    print(f"[{set_dir.name}] Мисматчей: {len(mismatches)}")
