"""
models/weighted_clustering.py

Task-driven Weighted Soft K-Means — основной вклад работы.

Идея: вместо равных весов признаков находим такие веса w_j,
при которых кластеризация максимизирует качество предсказания
целевой переменной (NEE) внутри каждого кластера.

Алгоритм:
  1. Инициализируем веса w = [1, 1, ..., 1] (как baseline)
  2. Вычисляем мягкое назначение: p(k|i) = softmax(-d(x_i, c_k) / τ)
  3. Вычисляем взвешенный in-cluster R² как дифференцируемую функцию
  4. Обновляем w через градиентный спуск (Adam)
  5. Уменьшаем τ (annealing) → кластеры становятся жёсткими
  6. Повторяем до сходимости

Ключевой факт: всё дифференцируемо → можно использовать autograd/torch.
Здесь реализовано на чистом numpy для минимума зависимостей.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
from sklearn.preprocessing import StandardScaler


@dataclass
class WeightedClusteringResult:
    labels: np.ndarray           # жёсткие назначения (argmax)
    soft_assignments: np.ndarray # мягкие вероятности p(k|i), shape (N, K)
    weights: np.ndarray          # обученные веса признаков
    centers: np.ndarray          # центры кластеров в пространстве признаков
    scaler: StandardScaler
    feature_cols: list
    X_scaled: np.ndarray
    loss_history: list           # -R² по эпохам
    weight_history: list         # веса по эпохам


class WeightedSoftKMeans:
    """
    Weighted Soft K-Means с task-aware обучением весов.

    Parameters
    ----------
    n_clusters   : число кластеров K
    temperature  : начальная температура τ (softmax)
    temperature_min : финальная температура (после annealing)
    n_epochs     : число итераций градиентного спуска
    lr           : learning rate (Adam)
    reg_lambda   : L2 регуляризация весов (штраф за большие веса)
    seed         : random seed
    """

    def __init__(
        self,
        n_clusters: int = 12,
        temperature: float = 1.0,
        temperature_min: float = 0.1,
        n_epochs: int = 200,
        lr: float = 0.01,
        reg_lambda: float = 0.01,
        convergence_tol: float = 1e-4,
        seed: int = 42,
    ):
        self.K = n_clusters
        self.tau_init = temperature
        self.tau_min = temperature_min
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg_lambda = reg_lambda
        self.tol = convergence_tol
        self.seed = seed

        self.weights_ = None
        self.centers_ = None
        self.loss_history_ = []
        self.weight_history_ = []

    def fit(
        self,
        X_scaled: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[list] = None,
    ) -> "WeightedSoftKMeans":
        """
        Обучает веса признаков.

        Parameters
        ----------
        X_scaled : стандартизированные признаки, shape (N, D)
        y        : целевая переменная (NEE), shape (N,)
        """
        np.random.seed(self.seed)
        N, D = X_scaled.shape
        self.feature_names_ = feature_names or [f"f{i}" for i in range(D)]

        # Инициализация весов (softplus чтобы > 0)
        # log-space: w = softplus(v) → всегда положительные
        v = np.zeros(D)  # логиты весов

        # Инициализация центров через случайные точки
        idx = np.random.choice(N, self.K, replace=False)
        self.centers_ = X_scaled[idx].copy()

        # Adam оптимизатор (для весов v)
        m_v, s_v = np.zeros(D), np.zeros(D)
        m_c = np.zeros_like(self.centers_)
        s_c = np.zeros_like(self.centers_)
        beta1, beta2, eps = 0.9, 0.999, 1e-8

        prev_loss = float('inf')
        patience_count = 0

        print(f"  [wcm] обучаем {self.K} кластеров, {self.n_epochs} эпох")

        for epoch in range(1, self.n_epochs + 1):
            # Temperature annealing: линейный schedule
            tau = max(
                self.tau_min,
                self.tau_init * (1 - epoch / self.n_epochs)
                + self.tau_min * (epoch / self.n_epochs),
            )

            # Положительные веса через softplus
            w = np.log1p(np.exp(v))  # softplus(v), ∈ (0, ∞)
            w_norm = w / (w.sum() + 1e-8) * D  # нормируем: сумма = D

            # Взвешенные расстояния: d(x_i, c_k) = Σ_j w_j (x_ij - c_kj)²
            # Форма: (N, K)
            diff = X_scaled[:, None, :] - self.centers_[None, :, :]  # (N,K,D)
            dist2 = (diff ** 2 * w_norm[None, None, :]).sum(axis=2)   # (N,K)

            # Мягкое назначение: p(k|i) = softmax(-dist2 / τ)
            logits = -dist2 / tau
            logits -= logits.max(axis=1, keepdims=True)  # numerical stability
            P = np.exp(logits)
            P /= P.sum(axis=1, keepdims=True)  # (N, K)

            # In-cluster weighted R²
            r2, grad_P = _weighted_r2_and_grad(P, y)

            # Регуляризация
            reg = self.reg_lambda * (w_norm ** 2).sum()
            loss = -r2 + reg

            # Градиент по весам w_norm через цепное правило
            # ∂loss/∂w_norm_j = Σ_{i,k} (∂loss/∂P_ik) * (∂P_ik/∂d_ik) * (-x_ij-c_kj)² / tau
            dL_dP = -grad_P  # (N, K)

            # ∂P_ik/∂dist2_ik = -P_ik/τ * (1 - P_ik) ≈ через Jacobian softmax
            # Упрощённо: ∂logL/∂dist2_ik = (1/τ) * (P_ik * Σ_l dL_dP_il*P_il - dL_dP_ik*P_ik)
            dL_dd = (1.0 / tau) * (
                P * (dL_dP * P).sum(axis=1, keepdims=True) - dL_dP * P
            )  # (N, K)

            # ∂dist2_ik/∂w_norm_j = (x_ij - c_kj)²
            dL_dw = (dL_dd[:, :, None] * diff ** 2).sum(axis=(0, 1))  # (D,)
            dL_dw += 2 * self.reg_lambda * w_norm

            # Градиент в пространстве v (цепное правило через softplus)
            sigmoid_v = 1.0 / (1.0 + np.exp(-v))  # σ(v) = dw/dv
            # Нормировка добавляет член: ∂(w_norm_j)/∂w_k
            sum_w = w.sum() + 1e-8
            dw_norm_dw = (D * sum_w - D * w) / sum_w ** 2  # диагональ Якобиана
            grad_v = dL_dw * dw_norm_dw * sigmoid_v

            # Adam update для v
            t = epoch
            m_v = beta1 * m_v + (1 - beta1) * grad_v
            s_v = beta2 * s_v + (1 - beta2) * grad_v ** 2
            m_hat = m_v / (1 - beta1 ** t)
            s_hat = s_v / (1 - beta2 ** t)
            v -= self.lr * m_hat / (np.sqrt(s_hat) + eps)

            # Обновляем центры (EM-step: взвешенное среднее)
            new_centers = (P[:, :, None] * X_scaled[:, None, :]).sum(axis=0)
            new_centers /= P.sum(axis=0)[:, None] + 1e-8
            self.centers_ = 0.9 * self.centers_ + 0.1 * new_centers  # мягкое обновление

            loss = float(loss)
            self.loss_history_.append(loss)
            self.weight_history_.append(w_norm.copy())

            # Логирование
            if epoch % 50 == 0 or epoch == 1:
                print(f"    эпоха {epoch:3d}: R²={r2:.4f}, τ={tau:.3f}, "
                      f"loss={loss:.4f}")

            # Early stopping
            if abs(prev_loss - loss) < self.tol:
                patience_count += 1
                if patience_count >= 10:
                    print(f"    сошлось на эпохе {epoch}")
                    break
            else:
                patience_count = 0
            prev_loss = loss

        self.weights_ = np.log1p(np.exp(v))
        self.weights_ /= (self.weights_.sum() + 1e-8) * D
        return self

    def predict(self, X_scaled: np.ndarray) -> np.ndarray:
        """Жёсткие назначения по ближайшему центру."""
        w = self.weights_
        diff = X_scaled[:, None, :] - self.centers_[None, :, :]
        dist2 = (diff ** 2 * w[None, None, :]).sum(axis=2)
        return dist2.argmin(axis=1)

    def soft_predict(self, X_scaled: np.ndarray, tau: float = 0.1) -> np.ndarray:
        """Мягкие вероятности при заданной температуре."""
        w = self.weights_
        diff = X_scaled[:, None, :] - self.centers_[None, :, :]
        dist2 = (diff ** 2 * w[None, None, :]).sum(axis=2)
        logits = -dist2 / tau
        logits -= logits.max(axis=1, keepdims=True)
        P = np.exp(logits)
        return P / P.sum(axis=1, keepdims=True)


def _weighted_r2_and_grad(
    P: np.ndarray, y: np.ndarray
) -> tuple[float, np.ndarray]:
    """
    Дифференцируемый взвешенный R².

    Для каждого кластера k вычисляем взвешенный R²:
      R²_k = 1 - SSres_k / SStot_k

    Где SSres_k = Σ_i p(k|i) * (y_i - ŷ_k)²
          SStot_k = Σ_i p(k|i) * (y_i - ȳ_k)²
          ŷ_k = Σ_i p(k|i) * y_i / Σ_i p(k|i)  (взвешенное среднее)

    Итоговый R² = взвешенное среднее R²_k (весa = Σ_i p(k|i)).

    Returns
    -------
    r2     : float — средний взвешенный R²
    grad_P : (N, K) — ∂r2/∂P_ik
    """
    N, K = P.shape
    eps = 1e-8

    # Веса кластеров
    W_k = P.sum(axis=0)  # (K,)

    # Взвешенные средние y в кластерах
    y_mean_k = (P * y[:, None]).sum(axis=0) / (W_k + eps)  # (K,)

    # SSres и SStot
    resid = y[:, None] - y_mean_k[None, :]   # (N, K)
    y_dev = y[:, None] - y_mean_k[None, :]   # то же (для SStot ŷ=y_mean)

    SSres = (P * resid ** 2).sum(axis=0)     # (K,)
    SStot = (P * y_dev  ** 2).sum(axis=0)    # (K,)  [= SSres когда ŷ=mean, т.е. =SSres]

    # На самом деле при ŷ=mean: SSres = SStot, R²=0
    # Используем глобальное среднее как baseline
    y_global = (P * y[:, None]).sum() / (P.sum() + eps)
    y_dev_global = y[:, None] - y_global
    SStot = (P * y_dev_global ** 2).sum(axis=0)  # (K,)

    R2_k = 1 - SSres / (SStot + eps)  # (K,)
    R2_k = np.clip(R2_k, -1, 1)       # ограничиваем снизу (нестабильные кластеры)

    # Итоговый R² = взвешенное среднее
    r2 = (W_k * R2_k).sum() / (W_k.sum() + eps)

    # Градиент ∂r2/∂P_ik — аналитически
    # Упрощение: считаем через конечные разности (надёжнее для небольших N)
    delta = 1e-4
    grad_P = np.zeros((N, K))
    for n in range(N):
        for k in range(K):
            P_plus = P.copy()
            P_plus[n, k] += delta
            # нормируем строку
            P_plus[n] /= P_plus[n].sum()
            r2_plus = _r2_fast(P_plus, y, y_global)
            grad_P[n, k] = (r2_plus - r2) / delta

    return float(r2), grad_P


def _r2_fast(P: np.ndarray, y: np.ndarray, y_global: float) -> float:
    """Быстрое вычисление R² без градиента."""
    eps = 1e-8
    W_k = P.sum(axis=0)
    y_mean_k = (P * y[:, None]).sum(axis=0) / (W_k + eps)
    resid = y[:, None] - y_mean_k[None, :]
    y_dev = y[:, None] - y_global
    SSres = (P * resid ** 2).sum(axis=0)
    SStot = (P * y_dev  ** 2).sum(axis=0)
    R2_k = 1 - SSres / (SStot + eps)
    R2_k = np.clip(R2_k, -1, 1)
    r2 = (W_k * R2_k).sum() / (W_k.sum() + eps)
    return float(r2)


def fit_weighted(
    df: pd.DataFrame,
    config: dict,
) -> WeightedClusteringResult:
    """
    Обёртка: обучает WeightedSoftKMeans на датасете.

    Returns
    -------
    WeightedClusteringResult
    """
    feature_cols = config["data"]["feature_cols"]
    target_col   = config["data"]["target_col"]
    wc_cfg       = config["weighted_clustering"]
    seed         = config["experiment"]["seed"]

    X = df[feature_cols].values
    y = df[target_col].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = WeightedSoftKMeans(
        n_clusters=config["clustering"]["n_clusters"],
        temperature=wc_cfg["temperature"],
        temperature_min=wc_cfg["temperature_min"],
        n_epochs=wc_cfg["n_epochs"],
        lr=wc_cfg["lr"],
        reg_lambda=wc_cfg["reg_lambda"],
        convergence_tol=wc_cfg["convergence_tol"],
        seed=seed,
    )
    model.fit(X_scaled, y, feature_names=feature_cols)

    labels = model.predict(X_scaled)
    soft   = model.soft_predict(X_scaled)

    return WeightedClusteringResult(
        labels=labels,
        soft_assignments=soft,
        weights=model.weights_,
        centers=model.centers_,
        scaler=scaler,
        feature_cols=feature_cols,
        X_scaled=X_scaled,
        loss_history=model.loss_history_,
        weight_history=model.weight_history_,
    )
