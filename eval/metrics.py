"""
eval/metrics.py

Все метрики качества кластеризации в одном месте.

Метрики:
  - silhouette_score     : геометрическое качество кластеров
  - in_cluster_r2        : насколько хорошо NEE предсказывается внутри кластера
  - adjusted_rand_index  : совпадение с экспертной классификацией (IGBP)
  - inertia              : сумма квадратов расстояний (для elbow-plot)
  - cluster_stats        : размеры и характеристики кластеров
"""

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from typing import Optional


def compute_all_metrics(
    df: pd.DataFrame,
    labels: np.ndarray,
    X_scaled: np.ndarray,
    config: dict,
    model_name: str = "model",
) -> dict:
    """
    Считает все метрики и возвращает словарь.

    Parameters
    ----------
    df       : исходный датасет с колонками [igbp, nee_annual, ...]
    labels   : назначения кластеров, shape (N,)
    X_scaled : стандартизированные признаки
    config   : конфиг
    model_name: имя модели для логирования

    Returns
    -------
    metrics : dict с ключами silhouette, in_cluster_r2, ari, ...
    """
    target_col   = config["data"]["target_col"]
    feature_cols = config["data"]["feature_cols"]

    y = df[target_col].values
    n_clusters = len(np.unique(labels))

    print(f"\n  [{model_name}] метрики кластеризации:")

    metrics = {"model": model_name, "n_clusters": n_clusters}

    # 1. Silhouette score
    if n_clusters > 1 and len(np.unique(labels)) > 1:
        sil = silhouette_score(X_scaled, labels, sample_size=min(len(labels), 5000))
        metrics["silhouette"] = round(float(sil), 4)
        print(f"    Silhouette score:     {sil:.4f}")
    else:
        metrics["silhouette"] = None

    # 2. In-cluster R² (главная метрика)
    r2 = _in_cluster_r2(y, labels, df[feature_cols].values)
    metrics["in_cluster_r2"] = round(float(r2), 4)
    print(f"    In-cluster R²:        {r2:.4f}  ← главная метрика")

    # 3. Global R² (baseline: один регрессор на всё)
    r2_global = _global_r2(y, df[feature_cols].values)
    metrics["global_r2"] = round(float(r2_global), 4)
    print(f"    Global R²:            {r2_global:.4f}  (без разбиения)")
    metrics["r2_lift"] = round(float(r2 - r2_global), 4)
    print(f"    R² lift:              {r2 - r2_global:+.4f}  ← прирост от разбиения")

    # 4. ARI против IGBP-классов (если есть)
    if "igbp" in df.columns:
        ari = adjusted_rand_score(df["igbp"], labels)
        metrics["ari_vs_igbp"] = round(float(ari), 4)
        print(f"    ARI vs IGBP:          {ari:.4f}")

    # 5. Статистика по кластерам
    stats = _cluster_stats(df, labels, target_col)
    metrics["cluster_stats"] = stats
    metrics["min_cluster_size"] = int(stats["size"].min())
    metrics["max_cluster_size"] = int(stats["size"].max())
    print(f"    Кластеры: мин={metrics['min_cluster_size']}, "
          f"макс={metrics['max_cluster_size']} станций")

    return metrics


def _in_cluster_r2(
    y: np.ndarray,
    labels: np.ndarray,
    X: np.ndarray,
) -> float:
    """
    Средний взвешенный R² внутри кластеров.

    Для каждого кластера k обучаем линейную регрессию y ~ X
    на станциях кластера. R² взвешивается по размеру кластера.

    Если в кластере < 3 станций — используем константную модель (R²=0).
    """
    unique_labels = np.unique(labels)
    r2_list, weights = [], []

    for k in unique_labels:
        mask = labels == k
        n_k = mask.sum()

        if n_k < 3:
            r2_list.append(0.0)
            weights.append(n_k)
            continue

        X_k, y_k = X[mask], y[mask]

        # Простая линейная регрессия: NEE ~ climate features
        # Используем только 2 первых признака во избежание переобучения
        # (у нас мало данных на кластер)
        X_k_simple = X_k[:, :2] if X_k.shape[1] >= 2 else X_k

        try:
            if n_k >= 5:
                # Cross-validated R²
                reg = LinearRegression()
                cv_r2 = cross_val_score(
                    reg, X_k_simple, y_k, cv=min(3, n_k), scoring="r2"
                )
                r2_k = float(np.clip(cv_r2.mean(), -1, 1))
            else:
                # Просто fit/predict
                reg = LinearRegression().fit(X_k_simple, y_k)
                r2_k = float(np.clip(reg.score(X_k_simple, y_k), -1, 1))
        except Exception:
            r2_k = 0.0

        r2_list.append(r2_k)
        weights.append(n_k)

    weights = np.array(weights, dtype=float)
    weights /= weights.sum()
    return float(np.dot(r2_list, weights))


def _global_r2(y: np.ndarray, X: np.ndarray) -> float:
    """R² линейной регрессии без разбиения на кластеры."""
    X_simple = X[:, :2] if X.shape[1] >= 2 else X
    try:
        reg = LinearRegression().fit(X_simple, y)
        return float(np.clip(reg.score(X_simple, y), -1, 1))
    except Exception:
        return 0.0


def _cluster_stats(
    df: pd.DataFrame,
    labels: np.ndarray,
    target_col: str,
) -> pd.DataFrame:
    """Статистика по кластерам: размер, mean/std NEE."""
    df = df.copy()
    df["cluster"] = labels
    stats = df.groupby("cluster").agg(
        size=(target_col, "count"),
        nee_mean=(target_col, "mean"),
        nee_std=(target_col, "std"),
    ).reset_index()
    stats["nee_mean"] = stats["nee_mean"].round(1)
    stats["nee_std"]  = stats["nee_std"].round(1)
    return stats


def compare_models(results: list[dict]) -> pd.DataFrame:
    """
    Создаёт сравнительную таблицу метрик для нескольких моделей.

    Parameters
    ----------
    results : список dict от compute_all_metrics

    Returns
    -------
    df_compare : DataFrame для вывода/экспорта
    """
    rows = []
    for r in results:
        rows.append({
            "Модель":          r["model"],
            "Кластеров":       r["n_clusters"],
            "Silhouette":      r.get("silhouette", "—"),
            "In-cluster R²":   r.get("in_cluster_r2", "—"),
            "Global R²":       r.get("global_r2", "—"),
            "R² lift":         r.get("r2_lift", "—"),
            "ARI vs IGBP":     r.get("ari_vs_igbp", "—"),
        })
    return pd.DataFrame(rows).set_index("Модель")
