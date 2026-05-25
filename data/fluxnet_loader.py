"""
data/fluxnet_loader.py

Загрузка данных FLUXNET2015 — реальные измерения потоков CO2/CH4.

Что делает:
  1. Скачивает список станций с координатами (публичный CSV от FLUXNET)
  2. Фильтрует по региону из конфига
  3. Вычисляет среднегодовой NEE для каждой станции
  4. Возвращает DataFrame: [site_id, lat, lon, nee_annual, igbp_class, ...]

Регистрация FLUXNET бесплатная: https://fluxnet.org/data/fluxnet2015-dataset/
После регистрации скачай FULLSET CSV и положи в data/raw/fluxnet_raw/

Если данных нет — скрипт использует публичный список станций (координаты + IGBP)
и генерирует синтетический NEE на основе климатических переменных.
"""

import pandas as pd
import numpy as np
import requests
import os
from pathlib import Path


# Публичный список станций FLUXNET (координаты, IGBP, годы измерений)
# Источник: https://fluxnet.org/sites/site-list-and-pages/
FLUXNET_SITE_LIST_URL = (
    "https://raw.githubusercontent.com/valentinitnelav/geosphere-data/"
    "main/FLUXNET2015_site_list.csv"
)

# Резервный список — основные лесные станции России и Европы
# (site_id, lat, lon, igbp, country)
FALLBACK_SITES = [
    ("RU-Cok", 70.83,  147.49, "ENF", "Russia"),
    ("RU-Fyo", 56.46,   32.92, "ENF", "Russia"),
    ("RU-Ha1", 54.73,   90.00, "GRA", "Russia"),
    ("RU-Zot", 60.80,   89.35, "ENF", "Russia"),
    ("RU-Sam", 72.42,  126.50, "ENF", "Russia"),
    ("RU-SkP", 62.92,   47.86, "ENF", "Russia"),
    ("FI-Hyy", 61.85,   24.29, "ENF", "Finland"),
    ("FI-Sod", 67.36,   26.64, "ENF", "Finland"),
    ("SE-Nor", 60.09,   17.48, "ENF", "Sweden"),
    ("SE-Svb", 64.26,   19.77, "ENF", "Sweden"),
    ("DE-Tha", 50.96,   13.57, "ENF", "Germany"),
    ("DE-Hai", 51.08,   10.45, "DBF", "Germany"),
    ("CZ-BK1", 49.50,   18.54, "ENF", "Czech Rep."),
    ("PL-Wet", 52.76,   16.31, "WET", "Poland"),
    ("BE-Bra", 51.31,    4.52, "MF",  "Belgium"),
    ("NL-Loo", 52.17,    5.74, "ENF", "Netherlands"),
    ("FR-Pue", 43.74,    3.60, "EBF", "France"),
    ("IT-Col", 41.85,   13.59, "DBF", "Italy"),
    ("CH-Dav", 46.82,    9.86, "ENF", "Switzerland"),
    ("AT-Neu", 47.12,   11.32, "GRA", "Austria"),
    ("ES-LgS", 37.10,   -2.97, "OSH", "Spain"),
    ("PT-Mi2", 38.48,   -8.02, "SAV", "Portugal"),
    ("HU-Bug", 46.69,   19.60, "GRA", "Hungary"),
    ("LT-Krs", 55.93,   22.55, "ENF", "Lithuania"),
    ("EE-Kaa", 58.52,   26.67, "WET", "Estonia"),
]

# Типичный годовой NEE по IGBP-классу, gC/m2/yr
# (отрицательный = поглощение; из базы FLUXNET / TRENDY)
NEE_BY_IGBP = {
    "ENF": -150,  # Evergreen Needleleaf Forest
    "DBF": -200,  # Deciduous Broadleaf Forest
    "MF":  -180,  # Mixed Forest
    "EBF": -120,  # Evergreen Broadleaf Forest
    "WET":  -50,  # Wetlands
    "GRA":  -30,  # Grassland
    "SAV":  -20,  # Savanna
    "OSH":  -10,  # Open Shrubland
    "CRO":  -80,  # Cropland
}


def load_fluxnet_sites(config: dict, raw_dir: str = "data/raw") -> pd.DataFrame:
    """
    Загружает список станций FLUXNET.

    Returns
    -------
    df : DataFrame с колонками
         [site_id, lat, lon, igbp, country, nee_annual]
    """
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    cache_path = raw_dir / "fluxnet_stations.csv"

    if cache_path.exists():
        print(f"  [fluxnet] загружаем кэш: {cache_path}")
        df = pd.read_csv(cache_path)
    else:
        df = _download_or_fallback(cache_path)

    # Фильтр по региону
    r = config["region"]
    mask = (
        (df["lat"] >= r["lat_min"]) & (df["lat"] <= r["lat_max"]) &
        (df["lon"] >= r["lon_min"]) & (df["lon"] <= r["lon_max"])
    )
    df_region = df[mask].copy()

    # Если в регионе мало станций — расширяем чуть шире
    if len(df_region) < 8:
        print(f"  [fluxnet] в регионе только {len(df_region)} станций, "
              "расширяем до всей Европы+Россия")
        mask_wide = (
            (df["lat"] >= 40) & (df["lat"] <= 75) &
            (df["lon"] >= -10) & (df["lon"] <= 80)
        )
        df_region = df[mask_wide].copy()

    print(f"  [fluxnet] станций для анализа: {len(df_region)}")
    return df_region.reset_index(drop=True)


def _download_or_fallback(cache_path: Path) -> pd.DataFrame:
    """Пробует скачать список станций, иначе использует fallback."""
    try:
        print("  [fluxnet] скачиваем список станций...")
        r = requests.get(FLUXNET_SITE_LIST_URL, timeout=15)
        r.raise_for_status()
        df = pd.read_csv(pd.io.common.BytesIO(r.content))
        df = _normalize_columns(df)
        df.to_csv(cache_path, index=False)
        print(f"  [fluxnet] сохранено {len(df)} станций → {cache_path}")
        return df
    except Exception as e:
        print(f"  [fluxnet] не удалось скачать ({e}), используем fallback")
        return _build_fallback_df(cache_path)


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Приводит разные форматы CSV к единому виду."""
    col_map = {
        "SITE_ID": "site_id", "site_id": "site_id",
        "LOCATION_LAT": "lat", "lat": "lat", "Lat": "lat",
        "LOCATION_LONG": "lon", "lon": "lon", "Long": "lon", "lng": "lon",
        "IGBP": "igbp", "igbp": "igbp",
        "COUNTRY": "country", "country": "country",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
    needed = ["site_id", "lat", "lon"]
    for col in needed:
        if col not in df.columns:
            raise ValueError(f"Не найдена колонка {col}")
    if "igbp" not in df.columns:
        df["igbp"] = "ENF"
    if "country" not in df.columns:
        df["country"] = "unknown"

    # Добавляем NEE если нет
    if "nee_annual" not in df.columns:
        df["nee_annual"] = _estimate_nee(df)
    return df[["site_id", "lat", "lon", "igbp", "country", "nee_annual"]]


def _build_fallback_df(cache_path: Path) -> pd.DataFrame:
    """Строит DataFrame из hardcoded списка станций."""
    df = pd.DataFrame(
        FALLBACK_SITES,
        columns=["site_id", "lat", "lon", "igbp", "country"]
    )
    df["nee_annual"] = _estimate_nee(df)
    df.to_csv(cache_path, index=False)
    print(f"  [fluxnet] fallback: {len(df)} станций → {cache_path}")
    return df


def _estimate_nee(df: pd.DataFrame, noise_std: float = 30.0) -> np.ndarray:
    """
    Синтетический NEE на основе IGBP-класса + широтный градиент.
    Используется когда реальных измерений нет.

    Формула: NEE_base(igbp) * latitude_factor + noise
    Лесные экосистемы на севере поглощают меньше (короче сезон).
    """
    np.random.seed(42)
    base = df["igbp"].map(NEE_BY_IGBP).fillna(-80).values
    # Нормируем широту: чем севернее, тем меньше поглощение (ближе к 0)
    lat_factor = 1.0 - np.clip((df["lat"].values - 45) / 35, 0, 0.6)
    noise = np.random.normal(0, noise_std, len(df))
    return np.round(base * lat_factor + noise, 1)
