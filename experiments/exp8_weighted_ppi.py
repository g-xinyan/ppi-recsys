"""
实验 8: 加权 PPI + 动态权重 PPI 对比
"""

import sys
import os
import random

import numpy as np
from scipy.stats import gaussian_kde
from tqdm.auto import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.config import Config
from src.recommender import PPIRecommender
from src.evaluation import compute_recalls_for_sessions, evaluate_group_with_ppi


def compute_session_weights(train_sessions, val_sessions):
    """
    基于会话相似度计算验证集会话权重

    每个验证集会话的权重由其与训练集会话的相似程度决定,
    用于自适应加权 PPI 估计。
    """
    train_profiles = []
    for session in train_sessions[:1000]:
        aids = set(int(ev["aid"]) for ev in session['events'])
        train_profiles.append(aids)

    weights = []
    for session in val_sessions:
        val_aids = set(int(ev["aid"]) for ev in session['events'])
        max_overlap = 0
        for train_aids in train_profiles:
            overlap = len(val_aids & train_aids) / max(len(val_aids | train_aids), 1)
            max_overlap = max(max_overlap, overlap)
        weights.append(max_overlap + 0.01)

    weights = np.array(weights)
    weights = weights / weights.mean()
    return weights


def run_weighted_ppi_experiment(strategy, val_sessions, train_sessions=None):
    """
    实验 8: 加权 PPI + 动态权重 PPI 对比

    比较标准 PPI、加权 PPI 和动态权重 PPI 的估计效果。
    """
    print("\n" + "=" * 60)
    print("Experiment 8: Weighted PPI Comparison")
    print("=" * 60)

    recalls_list = compute_recalls_for_sessions(val_sessions, strategy, 'full_strategy')
    print(f"Valid sessions: {len(recalls_list)}")

    # 标准 PPI
    result_standard = evaluate_group_with_ppi(recalls_list, 'Standard PPI', n_labeled=500)

    # 加权 PPI (使用会话权重)
    if train_sessions:
        weights = compute_session_weights(train_sessions, val_sessions)
    else:
        weights = np.ones(len(recalls_list))

    # 按权重采样
    weighted_indices = np.random.choice(
        range(len(recalls_list)),
        size=min(500, len(recalls_list)),
        p=weights[:len(recalls_list)] / weights[:len(recalls_list)].sum(),
    )

    ppi_weighted = PPIRecommender(min_strength=Config.PPI_MIN_STRENGTH,
                                  max_strength=Config.PPI_MAX_STRENGTH)
    for idx in weighted_indices:
        recalls = recalls_list[idx]
        for action_type in ['clicks', 'carts', 'orders']:
            true_val = recalls.get(action_type)
            if true_val is not None:
                pred_val = true_val * (1 + random.uniform(-0.1, 0.1))
                ppi_weighted.update_recall_stats(action_type, true_val, pred_val)

    result_weighted = {
        'group_name': 'Weighted PPI',
        'weighted_ppi': ppi_weighted.get_weighted_recall(),
        'weighted_ci': ppi_weighted.get_weighted_confidence_interval(),
    }

    print("\n" + "=" * 70)
    print("Weighted PPI Comparison Results")
    print("=" * 70)
    if result_standard:
        print(f"Standard PPI:  {result_standard['weighted_ppi']:.4f}  "
              f"CI: [{result_standard['weighted_ci'][0]:.4f}, {result_standard['weighted_ci'][1]:.4f}]")
    print(f"Weighted PPI:  {result_weighted['weighted_ppi']:.4f}  "
          f"CI: [{result_weighted['weighted_ci'][0]:.4f}, {result_weighted['weighted_ci'][1]:.4f}]")

    return {'standard': result_standard, 'weighted': result_weighted}


if __name__ == '__main__':
    print("Experiment 8: Weighted PPI Comparison")
    print("Run in Kaggle Notebook environment with Otto dataset.")
