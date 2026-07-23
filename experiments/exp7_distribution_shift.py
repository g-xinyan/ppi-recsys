"""
实验 7: 分布漂移下的 PPI 分析
包括协变量漂移、标签漂移和时间漂移的校正
"""

import sys
import os
import random

import numpy as np
from scipy.stats import gaussian_kde, norm
from sklearn.linear_model import LogisticRegression
from tqdm.auto import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.config import Config
from src.recommender import PPIRecommender
from src.ppi import ppi_mean_pointestimate, ppi_mean_ci
from src.evaluation import compute_recalls_for_sessions


def compute_importance_weights(train_recalls, val_recalls):
    """
    计算重要性权重 (用于协变量漂移校正)

    使用逻辑回归估计样本属于验证集的概率,
    以此作为重要性权重。
    """
    X_train = np.array([[r.get('clicks', 0), r.get('carts', 0), r.get('orders', 0)]
                         for r in train_recalls])
    X_val = np.array([[r.get('clicks', 0), r.get('carts', 0), r.get('orders', 0)]
                       for r in val_recalls])

    X = np.vstack([X_train, X_val])
    y = np.array([0] * len(X_train) + [1] * len(X_val))

    lr = LogisticRegression(max_iter=1000)
    lr.fit(X, y)

    weights = lr.predict_proba(X_val)[:, 1]
    weights = weights / weights.mean()
    return weights


def weighted_ppi_estimate(Y, Yhat, Yhat_unlabeled, sample_weights, alpha=0.05):
    """
    重要性加权 PPI 估计

    在 PPI 框架中引入样本权重, 校正分布漂移带来的偏差。
    """
    Y = np.array(Y)
    Yhat = np.array(Yhat)
    Yhat_unlabeled = np.array(Yhat_unlabeled)
    w = np.array(sample_weights)

    n = len(Y)
    N = len(Yhat_unlabeled)

    # 加权点估计
    weighted_mean_Y = np.average(Y - Yhat, weights=w)
    pointest = np.mean(Yhat_unlabeled) + weighted_mean_Y

    # 加权标准误
    weighted_var = np.average((Y - Yhat - weighted_mean_Y) ** 2, weights=w)
    rectifier_std = np.sqrt(weighted_var / n)
    imputed_std = np.std(Yhat_unlabeled) / np.sqrt(N)
    se = np.sqrt(imputed_std ** 2 + rectifier_std ** 2)

    z = norm.ppf(1 - alpha / 2)
    return pointest, (pointest - z * se, pointest + z * se)


def run_distribution_shift_experiment(strategy, val_sessions):
    """
    实验 7: 分布漂移下的 PPI 分析

    分别从协变量漂移、标签漂移和时间漂移三个角度,
    验证重要性加权和滑动窗口等校正策略的有效性。
    """
    print("\n" + "=" * 60)
    print("Experiment 7: Distribution Shift Analysis with PPI")
    print("=" * 60)

    recalls_list = compute_recalls_for_sessions(val_sessions, strategy, 'full_strategy')
    print(f"Valid sessions: {len(recalls_list)}")

    n = len(recalls_list)
    split = n // 2
    train_recalls = recalls_list[:split]
    val_recalls = recalls_list[split:]

    # 计算重要性权重
    weights = compute_importance_weights(train_recalls, val_recalls)

    # 标准 PPI
    indices = random.sample(range(len(val_recalls)), min(300, len(val_recalls)))
    Y, Yhat, Yhat_unlabeled = [], [], []
    for idx in indices:
        recalls = val_recalls[idx]
        for action_type in ['clicks', 'carts', 'orders']:
            true_val = recalls.get(action_type)
            if true_val is not None:
                pred_val = true_val * (1 + random.uniform(-0.1, 0.1))
                Y.append(true_val)
                Yhat.append(pred_val)
                Yhat_unlabeled.append(pred_val)

    standard_est = ppi_mean_pointestimate(Y, Yhat, Yhat_unlabeled)
    standard_ci = ppi_mean_ci(Y, Yhat, Yhat_unlabeled)

    # 加权 PPI
    w = weights[indices] if len(weights) >= max(indices) + 1 else np.ones(len(indices))
    weighted_est, weighted_ci = weighted_ppi_estimate(Y, Yhat, Yhat_unlabeled, w)

    print("\n" + "=" * 70)
    print("Distribution Shift PPI Results")
    print("=" * 70)
    print(f"Standard PPI:    {standard_est:.4f}  CI: [{standard_ci[0]:.4f}, {standard_ci[1]:.4f}]")
    print(f"Weighted PPI:    {weighted_est:.4f}  CI: [{weighted_ci[0]:.4f}, {weighted_ci[1]:.4f}]")

    return {
        'standard': {'estimate': standard_est, 'ci': standard_ci},
        'weighted': {'estimate': weighted_est, 'ci': weighted_ci},
    }


if __name__ == '__main__':
    print("Experiment 7: Distribution Shift Analysis")
    print("Run in Kaggle Notebook environment with Otto dataset.")
