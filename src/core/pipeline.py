from __future__ import annotations
from .spectrum_ops import load_spectrum, denoise, find_series, DELTA_CD3, DELTA_CD3CO, assign_formulas, build_result_table, visualize_series
import pandas as pd

# TODO: синхронизировать mass_min/mass_max между приложением и тестами.
#  В интеграционных тестах assign_formulas стабильно работает только при
#  явном mass_min=0, mass_max=1000. Нужно:
#    - либо делать такие границы дефолтными для режима simple,
#    - либо прокидывать настройки из единого конфигурационного места,
#      чтобы поведение пайплайна и тестов не расходилось.
def run_pipeline(
    src_path,
    dmet_path,
    dacet_path,
    *,
    # Загрузка
    sep=",",
    mass_min=200.0,
    mass_max=700.0,
    # Шумоподавление
    noise_force=1.5,
    noise_intensity=None,
    noise_quantile=None,
    # Назначение формул
    brutto_dict=None,
    rel_error=1.0,
    sign='-',
    # Поиск серий
    ppm_tol=5.0,
    max_groups=20,
    allow_gaps=True,
    # Визуализация
    visualize=True,
    save_dmet=None,
    save_dacet=None,
    # Выходной файл
    output_csv=None,
):
    """
    Полный пайплайн определения числа -COOH и -OH групп.

    Параметры
    ---------
    src_path, dmet_path, dacet_path : str | Path
        Пути к CSV-файлам трёх спектров.
    src_mapper, deriv_mapper : dict, optional
        Словари переименования столбцов {'m/z': 'mass', 'I': 'intensity'}.
    sep : str
        Разделитель полей CSV.
    mass_min, mass_max : float
        Рабочий диапазон масс (Da).
    noise_force / noise_intensity / noise_quantile
        Параметры шумоподавления (см. denoise()).
    brutto_dict : dict, optional
        Элементы и диапазоны для назначения формул.
    rel_error : float
        Допустимая ошибка назначения (ppm).
    sign : {'-', '+', '0'}
        Режим ионизации.
    ppm_tol : float
        Допуск поиска пика в серии (ppm). Рекомендуется 3-5 ppm.
    max_groups : int
        Максимальное ожидаемое число групп на молекулу.
    allow_gaps : bool
        Разрешать ли пропуски внутри серии (рекомендуется True).
    visualize : bool
        Строить ли графики для серий с пропущенными пиками.
    save_dmet, save_dacet : str, optional
        Пути для сохранения графиков (PNG/PDF).
    output_csv : str, optional
        Сохранить итоговую таблицу в CSV.

    Возвращает
    ----------
    pd.DataFrame — итоговая таблица.
    """
    print('=' * 60)
    print('ШАГ 1: Загрузка спектров')
    print('=' * 60)
    _mapper = {"mass": "mass", "intensity": "intensity"}
    src   = load_spectrum(src_path,   mapper = _mapper, sep=sep,
                          mass_min=mass_min, mass_max=mass_max, metadata={'name': 'src'})
    dmet  = load_spectrum(dmet_path,  mapper = _mapper, sep=sep,
                          mass_min=mass_min, mass_max=mass_max, metadata={'name': 'dmet'})
    dacet = load_spectrum(dacet_path, mapper = _mapper, sep=sep,
                          mass_min=mass_min, mass_max=mass_max, metadata={'name': 'dacet'})
    print(f"  Загружено пиков:  src={len(src)},  dmet={len(dmet)},  dacet={len(dacet)}")

    print()
    print('=' * 60)
    print('ШАГ 2a: Шумоподавление')
    print('=' * 60)
    src   = denoise(src,   force=noise_force, intensity=noise_intensity, quantile=noise_quantile)
    dmet  = denoise(dmet,  force=noise_force, intensity=noise_intensity, quantile=noise_quantile)
    dacet = denoise(dacet, force=noise_force, intensity=noise_intensity, quantile=noise_quantile)
    print(f"  После шумоподавления: src={len(src)},  dmet={len(dmet)},  dacet={len(dacet)}")

    print()
    print('=' * 60)
    print('ШАГ 2b: Назначение брутто-формул исходному спектру')
    print('=' * 60)
    src = assign_formulas(src, brutto_dict=brutto_dict, rel_error=rel_error,
                          sign=sign, mass_min=mass_min, mass_max=mass_max)
    n_assigned = int(src.table.get('assign', pd.Series(dtype=bool)).sum())
    print(f"  Назначено формул: {n_assigned} из {len(src)} пиков")

    print()
    print('=' * 60)
    print('ШАГ 3: Серии дейтерометилирования (-> N_COOH)')
    print('=' * 60)

    print("DEBUG after assign_formulas")
    print("type(src):", type(src))
    print("src columns:", list(src.table.columns))
    print(src.table[["mass"] + [c for c in ["assign", "brutto"] if c in src.table.columns]].head())
    df_dmet = find_series(src, dmet, delta=DELTA_CD3,
                          ppm_tol=ppm_tol, max_groups=max_groups, allow_gaps=allow_gaps)
    print(f"  Соединений с сериями CD3: {len(df_dmet)}")
    if not df_dmet.empty:
        print(f"  Макс. N_COOH = {df_dmet['n_groups'].max()}")
        ng = df_dmet['missing'].apply(len).sum()
        if ng:
            print(f"  ВНИМАНИЕ: Внутренних пропусков в сериях: {ng}")

    print()
    print('=' * 60)
    print('ШАГ 4: Серии дейтероацилирования (-> N_OH_total)')
    print('=' * 60)
    df_dacet = find_series(src, dacet, delta=DELTA_CD3CO,
                           ppm_tol=ppm_tol, max_groups=max_groups, allow_gaps=allow_gaps)
    print(f"  Соединений с сериями CD3CO: {len(df_dacet)}")
    if not df_dacet.empty:
        print(f"  Макс. N_OH_total = {df_dacet['n_groups'].max()}")
        ng2 = df_dacet['missing'].apply(len).sum()
        if ng2:
            print(f"  ВНИМАНИЕ: Внутренних пропусков в сериях: {ng2}")

    print()
    print('=' * 60)
    print('ШАГ 5: Итоговая таблица N_COOH / N_OH')
    print('=' * 60)
    result = build_result_table(src, df_dmet, df_dacet)
    print(f"  Строк в таблице: {len(result)}")
    print(f"  Соединений с N_COOH > 0: {(result['N_COOH'] > 0).sum()}")
    print(f"  Соединений с N_OH   > 0: {(result['N_OH']   > 0).sum()}")

    if visualize:
        print()
        print('=' * 60)
        print('ШАГ 6: Визуализация пропущенных пиков')
        print('=' * 60)
        visualize_series(src, dmet, df_dmet,
                         delta=DELTA_CD3, label="дейтерометилирования",
                         ppm_tol=ppm_tol, save_path=save_dmet)
        visualize_series(src, dacet, df_dacet,
                         delta=DELTA_CD3CO, label="дейтероацилирования",
                         ppm_tol=ppm_tol, save_path=save_dacet)

    if output_csv:
        result.to_csv(output_csv, index=False, sep=';', encoding='utf-8-sig')
        print(f"\nИтоговая таблица сохранена: {output_csv}")

    return result
