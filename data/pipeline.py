"""
data/pipeline.py

Сборочный конвейер: FLUXNET + CHELSA → чистый датасет для кластеризации.

Использование:
    from data.pipeline import build_dataset
    df = build_dataset(config)
"""

import pandas as pd
import numpy as np
from pathlib import Path

from data.fluxnet_loader import load_fluxnet_sites
from data.chelsa_loader import extract_climate_features


def build_dataset(config: dict, force_rebuild: bool = False) -> pd.DataFrame:
    """
    Строит объединённый датасет.

    Returns
    -------
    df : DataFrame с колонками
         [site_id, lat, lon, nee_annual, igbp, bio1, ..., tree_cover]
         Готов к подаче в модели.
    """
    processed_path = Path(config["data"]["processed_path"])
    processed_path.parent.mkdir(parents=True, exist_ok=True)

    if processed_path.exists() and not force_rebuild:
        print(f"[pipeline] загружаем готовый датасет: {processed_path}")
        df = pd.read_csv(processed_path)
        print(f"[pipeline] {len(df)} станций, {df.shape[1]} колонок")
        return df

    print("[pipeline] строим датасет с нуля...")

    # 1. Станции FLUXNET
    print("\n→ Шаг 1: FLUXNET станции")
    stations = load_fluxnet_sites(config)

    # 2. Климатические фичи
    print("\n→ Шаг 2: климатические фичи (CHELSA)")
    features = extract_climate_features(
        stations[["site_id", "lat", "lon"]],
        config,
        cache_path=config["data"]["chelsa_path"],
    )

    # 3. Объединяем
    print("\n→ Шаг 3: объединяем")
    df = stations.merge(features, on=["site_id", "lat", "lon"], how="inner")

    # 4. Чистка
    df = _clean(df, config)

    # 5. Сохраняем
    df.to_csv(processed_path, index=False)
    print(f"\n[pipeline] ✓ сохранено {len(df)} станций → {processed_path}")
    _print_summary(df, config)

    return df


def _clean(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Удаляет выбросы, заполняет пропуски."""
    feature_cols = config["data"]["feature_cols"]
    target_col   = config["data"]["target_col"]

    # Удаляем строки с пропусками в фичах
    before = len(df)
    df = df.dropna(subset=feature_cols + [target_col])
    if len(df) < before:
        print(f"  [clean] удалено {before - len(df)} строк с NaN")

    # Удаляем экстремальные выбросы по z-score > 4
    for col in feature_cols:
        z = np.abs((df[col] - df[col].mean()) / df[col].std())
        outliers = (z > 4).sum()
        if outliers > 0:
            print(f"  [clean] {col}: {outliers} выбросов → заменяем медианой")
            df.loc[z > 4, col] = df[col].median()

    return df.reset_index(drop=True)


def _print_summary(df: pd.DataFrame, config: dict) -> None:
    """Печатает краткую сводку по датасету."""
    feature_cols = config["data"]["feature_cols"]
    target_col   = config["data"]["target_col"]

    print("\n" + "─" * 50)
    print("ДАТАСЕТ:")
    print(f"  Станций: {len(df)}")
    print(f"  Признаков: {len(feature_cols)}")
    print(f"  Целевая переменная ({target_col}):")
    print(f"    mean={df[target_col].mean():.1f}, "
          f"std={df[target_col].std():.1f}, "
          f"min={df[target_col].min():.1f}, "
          f"max={df[target_col].max():.1f}")
    if "igbp" in df.columns:
        print(f"  IGBP классы: {df['igbp'].value_counts().to_dict()}")
    print("─" * 50)
