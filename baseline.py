"""
models/baseline.py

KMeans baseline — воспроизводит подход Kharitonova et al. (2025).
Равные веса всех признаков, стандартная кластеризация.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BaselineResult:
    labels: np.ndarray
    scaler: StandardScaler
    kmeans: KMeans
    feature_cols: list
    weights: np.ndarray        # для совместимости с WeightedResult
    X_scaled: np.ndarray


def fit_baseline(
    df: pd.DataFrame,
    config: dict,
) -> BaselineResult:
    """
    Обучает KMeans с равными весами (baseline).

    Parameters
    ----------
    df     : датасет (уже очищенный)
    config : словарь из config.yaml

    Returns
    -------
    BaselineResult
    """
    feature_cols = config["data"]["feature_cols"]
    n_clusters   = config["clustering"]["n_clusters"]
    seed         = config["experiment"]["seed"]

    X = df[feature_cols].values

    # Стандартизация (обязательна для KMeans)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # KMeans
    kmeans = KMeans(
        n_clusters=n_clusters,
        n_init=config["clustering"]["n_init"],
        max_iter=config["clustering"]["max_iter"],
        random_state=seed,
    )
    labels = kmeans.fit_predict(X_scaled)

    print(f"  [baseline] KMeans: {n_clusters} кластеров, "
          f"инерция = {kmeans.inertia_:.1f}")

    return BaselineResult(
        labels=labels,
        scaler=scaler,
        kmeans=kmeans,
        feature_cols=feature_cols,
        weights=np.ones(len(feature_cols)),  # равные веса
        X_scaled=X_scaled,
    )
