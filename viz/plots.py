"""
viz/plots.py

Визуализация результатов: карты экорегионов, веса признаков, обучение.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from typing import Optional


PALETTE = [
    "#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#3B1F2B",
    "#44BBA4", "#E94F37", "#393E41", "#F5A623", "#7B2D8B",
    "#00A878", "#D62246",
]


def plot_comparison_maps(
    df: pd.DataFrame,
    labels_baseline: np.ndarray,
    labels_weighted: np.ndarray,
    output_dir: str = "results/",
    title_suffix: str = "",
) -> None:
    """Карты экорегионов: baseline vs weighted, рядом."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, labels, title in zip(
        axes,
        [labels_baseline, labels_weighted],
        ["KMeans baseline", "Weighted clustering (наш метод)"],
    ):
        sc = ax.scatter(
            df["lon"], df["lat"],
            c=labels,
            cmap="tab20",
            s=120,
            alpha=0.85,
            edgecolors="white",
            linewidths=0.5,
        )
        ax.set_xlabel("Долгота (°E)", fontsize=11)
        ax.set_ylabel("Широта (°N)", fontsize=11)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.grid(True, alpha=0.2)
        plt.colorbar(sc, ax=ax, label="Кластер")

        # Подписи станций
        if "site_id" in df.columns and len(df) <= 50:
            for _, row in df.iterrows():
                ax.annotate(
                    row["site_id"],
                    (row["lon"], row["lat"]),
                    fontsize=6,
                    alpha=0.7,
                    xytext=(3, 3),
                    textcoords="offset points",
                )

    plt.suptitle(
        f"Экорегионы лесных ландшафтов{title_suffix}",
        fontsize=14, y=1.02,
    )
    plt.tight_layout()
    path = Path(output_dir) / "comparison_maps.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  [viz] карты сохранены → {path}")


def plot_feature_weights(
    feature_cols: list,
    weights_baseline: np.ndarray,
    weights_weighted: np.ndarray,
    output_dir: str = "results/",
) -> None:
    """Горизонтальный barplot: важность признаков в двух моделях."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Нормируем для сравнения
    wb = weights_baseline / (weights_baseline.sum() + 1e-8)
    ww = weights_weighted / (weights_weighted.sum() + 1e-8)

    idx = np.argsort(ww)[::-1]  # сортируем по весу в weighted модели
    names = [feature_cols[i] for i in idx]
    wb_sorted = wb[idx]
    ww_sorted = ww[idx]

    fig, ax = plt.subplots(figsize=(10, 7))
    y_pos = np.arange(len(names))
    bar_h = 0.35

    ax.barh(y_pos + bar_h / 2, wb_sorted, bar_h,
            label="Baseline (равные)", color="#ADB5BD", alpha=0.8)
    ax.barh(y_pos - bar_h / 2, ww_sorted, bar_h,
            label="Weighted (наш метод)", color="#2E86AB", alpha=0.9)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel("Нормированный вес признака", fontsize=11)
    ax.set_title("Важность признаков для районирования", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, axis="x", alpha=0.3)
    ax.axvline(1 / len(names), color="gray", linestyle="--",
                alpha=0.5, label="Равный вес")

    plt.tight_layout()
    path = Path(output_dir) / "feature_weights.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  [viz] веса признаков → {path}")


def plot_training_curve(
    loss_history: list,
    weight_history: list,
    feature_cols: list,
    output_dir: str = "results/",
) -> None:
    """График обучения: loss и эволюция весов по эпохам."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curve
    ax1.plot(loss_history, color="#C73E1D", linewidth=2)
    ax1.set_xlabel("Эпоха", fontsize=11)
    ax1.set_ylabel("Loss (−R²)", fontsize=11)
    ax1.set_title("Кривая обучения", fontsize=13)
    ax1.grid(True, alpha=0.3)

    # Эволюция весов
    weight_history = np.array(weight_history)  # (n_epochs, D)
    D = weight_history.shape[1]
    # Показываем топ-5 признаков по финальному весу
    final_weights = weight_history[-1]
    top_idx = np.argsort(final_weights)[::-1][:5]

    for i in top_idx:
        name = feature_cols[i] if i < len(feature_cols) else f"f{i}"
        ax2.plot(weight_history[:, i], linewidth=2,
                 label=name, alpha=0.85)

    ax2.set_xlabel("Эпоха", fontsize=11)
    ax2.set_ylabel("Вес признака", fontsize=11)
    ax2.set_title("Эволюция весов (топ-5)", fontsize=13)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = Path(output_dir) / "training_curve.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  [viz] кривая обучения → {path}")


def plot_metrics_table(
    df_compare: pd.DataFrame,
    output_dir: str = "results/",
) -> None:
    """Красивая таблица метрик как matplotlib figure."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, max(3, len(df_compare) + 1.5)))
    ax.axis("off")

    col_widths = [0.22] + [0.13] * (len(df_compare.columns))
    table = ax.table(
        cellText=df_compare.reset_index().values,
        colLabels=["Модель"] + list(df_compare.columns),
        cellLoc="center",
        loc="center",
        colWidths=col_widths,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2.0)

    # Подсветить строку с нашим методом
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#2E86AB")
            cell.set_text_props(color="white", fontweight="bold")
        elif "Weighted" in str(df_compare.reset_index().iloc[row - 1, 0]):
            cell.set_facecolor("#E8F4FD")
        else:
            cell.set_facecolor("#F8F9FA")

    plt.title("Сравнение моделей", fontsize=14, fontweight="bold", pad=20)
    path = Path(output_dir) / "metrics_table.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  [viz] таблица метрик → {path}")


def plot_nee_by_cluster(
    df: pd.DataFrame,
    labels_baseline: np.ndarray,
    labels_weighted: np.ndarray,
    target_col: str = "nee_annual",
    output_dir: str = "results/",
) -> None:
    """Boxplot NEE по кластерам — показывает насколько кластеры однородны."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

    for ax, labels, title in zip(
        [ax1, ax2],
        [labels_baseline, labels_weighted],
        ["Baseline (KMeans)", "Weighted (наш метод)"],
    ):
        data_by_cluster = [
            df[target_col][labels == k].values
            for k in sorted(np.unique(labels))
        ]
        ax.boxplot(data_by_cluster, patch_artist=True,
                   boxprops=dict(facecolor="#ADB5BD", alpha=0.7))
        ax.set_xlabel("Кластер", fontsize=11)
        ax.set_ylabel(f"{target_col} (gC/m²/yr)", fontsize=11)
        ax.set_title(title, fontsize=13)
        ax.grid(True, axis="y", alpha=0.3)
        ax.axhline(0, color="red", linestyle="--", alpha=0.4, linewidth=0.8)

    plt.suptitle(
        "Распределение NEE по кластерам\n"
        "(меньше разброс внутри = лучше)",
        fontsize=13,
    )
    plt.tight_layout()
    path = Path(output_dir) / "nee_by_cluster.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  [viz] boxplot NEE → {path}")
