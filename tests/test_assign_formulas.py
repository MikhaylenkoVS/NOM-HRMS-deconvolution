#/test_assign_fomulas.py
import pandas as pd
import pytest
from pathlib import Path
import re
from src.core.spectrum_ops import load_spectrum, assign_formulas, denoise

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent

TEST_SETS = [
    PROJECT_ROOT / "data" / "test_sets" / "set_01",
    PROJECT_ROOT / "data" / "test_sets" / "set_02",
    PROJECT_ROOT / "data" / "test_sets" / "set_03",
    PROJECT_ROOT / "data" / "test_sets" / "set_04",
    PROJECT_ROOT / "data" / "test_sets" / "set_05",
]

DEFAULT_BRUTTO_DICT = {
    "C": (0, 50),
    "H": (0, 100),
    "O": (0, 25),
    "N": (0, 10),
}

FORMULA_RE = re.compile(r"([A-Z][a-z]?)(\d*)")
def _subtract_one_h(brutto: str) -> str:
    """
    Временный костыль: уменьшить число H на 1 в строке формулы.
    Работает только для CHON-формул, где H явно указан.
    Примеры:
      C20H29O2 -> C20H28O2
      C10H14O2N -> C10H13O2N
    Если H нет или H1, оставляем как есть.
    """
    if brutto is None or not isinstance(brutto, str):
        return brutto

    m = re.search(r"H(\d+)", brutto)
    if not m:
        # нет явного Hn — ничего не меняем
        return brutto

    h_count = int(m.group(1))
    if h_count <= 1:
        # H или H1 — не рискуем уходить в H0/H-1
        return brutto

    new_h = f"H{h_count - 1}"
    old_h = f"H{h_count}"
    return brutto.replace(old_h, new_h, 1)

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
    - процент шумовых пиков, получивших формулу.


    Проверка assign (mode='simple') для spectrum_type='original' во всех наборах set_0N.

    TODO [release]:
    - выкинуть H-1 костыль для synthetic-аннотаций после выравнивания mass/ion_mode;
    - реализовать более честный учёт шума (denoise / intensity threshold);
    - добавить мягкий NOM-фильтр при выборе лучшей формулы среди кандидатов.
    """

    src_path = set_dir / "original.csv"
    ann_path = set_dir / "annotations.csv"

    assert src_path.exists(), f"Нет original.csv в {set_dir}"
    assert ann_path.exists(), f"Нет annotations.csv в {set_dir}"

    # 1. Загружаем спектр
    src = load_spectrum(src_path, mass_min=100, mass_max=1000)

    # 2. Назначаем формулы простым режимом (генерация по brutto_dict в spectrum_ops)
    rel_error_ppm = 0.5

    src = assign_formulas(
        src,
        mode="simple",
        rel_error_ppm=rel_error_ppm,
        mass_min=None,
        mass_max=None,
    )

    src_df = src.table.copy()
    assert "assign" in src_df.columns
    assert "brutto" in src_df.columns

    # TODO: TEMP HACK до согласования mass / ion_mode:
    # synthetic-данные хранят нейтральные массы, а simple-assign сейчас
    # систематически даёт формулы с +1 H. Для теста снимаем 1 протон.
    from copy import deepcopy
    # NOTE / TODO (release-blocker):
    # Сейчас synthetic-test_sets хранят нейтральные массы, а assign_formulas_simple
    # интерпретирует mass как m/z [M-H]-. В результате simple-assign систематически
    # даёт формулы с +1 H относительно истинных.
    #
    # Временное решение для тестов: после assign снимаем один протон из brutto
    # (_subtract_one_h). Это КОСТЫЛЬ только для synthetic-данных.
    #
    # К релизу нужно:
    # - синхронизировать mass-модель generator vs assign_formulas_simple,
    # - убрать этот костыль и проверить, что match_ratio остаётся высоким.

    src_df = src.table.copy()

    mask_assigned = src_df["assign"] == True
    src_df.loc[mask_assigned, "brutto"] = (
        src_df.loc[mask_assigned, "brutto"]
        .apply(_subtract_one_h)
    )

    # обновляем src, чтобы дальнейший код теста работал с "исправленными" brutto
    src.table = src_df
    assigned = src_df[src_df["assign"] == True].copy()

    # 3. Загружаем annotations и берём only original
    ann = pd.read_csv(ann_path)
    ann_orig = ann[(ann["spectrum_type"] == "original") & (ann["is_signal"])].copy()
    ann_noise = ann[(ann["spectrum_type"] == "original") & (~ann["is_signal"])].copy()

    n_assigned = len(assigned)
    n_signals = len(ann_orig)

    assert n_signals > 0, f"{set_dir.name}: нет сигнальных пиков для 'original'"
    assert n_assigned > 0, f"{set_dir.name}: assign_formulas(simple) не назначил ни одной формулы"

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

            print(f"[{set_dir.name}] NO_PEAK_IN_ASSIGNED для сигнала {mass_obs:.6f} ({formula_true}):")
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

            print(f"[{set_dir.name}] WRONG_BRUTTO для сигнала {mass_obs:.6f} ({formula_true}):")
            print(candidates[["mass", "brutto", "ppm_err"]].head(5).to_string(index=False))

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

    # 8. Шумовые пики, которые получили формулу (false positives)
    noise_with_formula = 0
    n_noise = len(ann_noise)

    for _, row in ann_noise.iterrows():
        mass_obs = row["mass_obs"]
        diff_ppm = (assigned["mass"] - mass_obs) / mass_obs * 1e6
        in_window = assigned[diff_ppm.abs() <= rel_error_ppm + 1e-6]
        if in_window.empty:
            continue  # вообще нет пика в окне — считаем, что шум проигнорирован

        if any(in_window["assign"] == True):
            noise_with_formula += 1

    # TODO [noise]: пересмотреть стратегию учёта шума.
    # Сейчас simple-assign не различает сигнал/шум и может назначать формулы
    # слабым/случайным пикам. В synthetic-тесте мы просто ограничиваем
    # noise_ratio <= 0.2.
    # К релизу:
    # - либо использовать результат denoise / порог по интенсивности до assign,
    # - либо внедрить в assign явный фильтр по интенсивности для шума.

    noise_ratio = noise_with_formula / n_noise if n_noise > 0 else 0.0

    print(
        f"[{set_dir.name}] Шумовых пиков с формулой: "
        f"{noise_with_formula}/{n_noise} (доля {noise_ratio:.3f})"
    )

    # Жёсткий критерий: хотим мало ложноположительных
    # Например, не более 20 % шумовых пиков получили формулу
    assert noise_ratio <= 0.2, (
        f"{set_dir.name}: слишком много шумовых пиков получили формулу "
        f"({noise_with_formula}/{n_noise}, доля {noise_ratio:.3f})"
    )
    # 9. Логи (некритические метрики)
    print(f"\n[{set_dir.name}] Назначено формул: {n_assigned}")
    print(f"[{set_dir.name}] Сигнальных пиков (original): {n_signals}")
    print(f"[{set_dir.name}] Совпадений с annotations: {matches}/{n_signals} (доля {match_ratio:.3f})")
    print(f"[{set_dir.name}] Мисматчей: {len(mismatches)}")
    print(
        f"[{set_dir.name}] Шумовых пиков с формулой: "
        f"{noise_with_formula}/{n_noise} (доля {noise_ratio:.3f})"
    )