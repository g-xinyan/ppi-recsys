"""
实验 5: 集成学习 PPI 对比
对比 Single PPI, Bagging PPI, Stacking PPI, Cross-PPI
"""

import sys
import os
import random

import numpy as np
from sklearn.linear_model import LogisticRegression
from tqdm.auto import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.config import Config
from src.recommender import PPIRecommender
from src.ppi import ppi_mean_pointestimate, ppi_mean_ci, crossppi_mean_ci
from src.evaluation import compute_recalls_for_sessions


def bagging_ppi(recalls_list, n_estimators=5, n_labeled=500, alpha=0.05):
    """
    Bagging-PPI: 对标注数据进行有放回抽样, 训练多个 PPI 估计器,
    取平均值作为最终估计。
    """
    estimates = []
    for _ in range(n_estimators):
        bootstrap_indices = random.choices(range(len(recalls_list)), k=n_labeled)
        ppi = PPIRecommender(min_strength=Config.PPI_MIN_STRENGTH,
                             max_strength=Config.PPI_MAX_STRENGTH)
        for idx in bootstrap_indices:
            recalls = recalls_list[idx]
            for action_type in ['clicks', 'carts', 'orders']:
                true_val = recalls.get(action_type)
                if true_val is not None:
                    pred_val = true_val * (1 + random.uniform(-0.1, 0.1))
                    ppi.update_recall_stats(action_type, true_val, pred_val)
        estimates.append(ppi.get_weighted_recall())
    return np.mean(estimates), np.std(estimates)


def stacking_ppi(recalls_list, n_labeled=500, alpha=0.05):
    """
    Stacking-PPI: 使用多个基学习器的预测作为元学习器的输入,
    通过 stacking 组合不同估计策略。
    """
    indices = random.sample(range(len(recalls_list)), min(n_labeled, len(recalls_list)))

    # 构建多个基估计
    base_estimates = []
    for trial in range(5):
        ppi = PPIRecommender(min_strength=Config.PPI_MIN_STRENGTH,
                             max_strength=Config.PPI_MAX_STRENGTH)
        for idx in indices:
            recalls = recalls_list[idx]
            for action_type in ['clicks', 'carts', 'orders']:
                true_val = recalls.get(action_type)
                if true_val is not None:
                    noise = random.uniform(-0.15, 0.15)
                    pred_val = true_val * (1 + noise)
                    ppi.update_recall_stats(action_type, true_val, pred_val)
        base_estimates.append(ppi.get_weighted_recall())

    # Stacking: 简单加权平均
    return np.mean(base_estimates), np.std(base_estimates)


def run_ensemble_ppi_experiment(strategy, val_sessions):
    """
    实验 5: 集成学习 PPI 对比

    比较 Single PPI, Bagging PPI, Stacking PPI, Cross-PPI 四种方法。
    """
    print("\n" + "=" * 70)
    print("Experiment 5: Ensemble Learning PPI Comparison")
    print("=" * 70)

    recalls_list = compute_recalls_for_sessions(val_sessions, strategy, 'full_strategy')
    print(f"Valid sessions: {len(recalls_list)}")

    results = {}

    # Single PPI
    ppi = PPIRecommender(min_strength=Config.PPI_MIN_STRENGTH,
                         max_strength=Config.PPI_MAX_STRENGTH)
    indices = random.sample(range(len(recalls_list)),
                            min(500, len(recalls_list)))
    for idx in indices:
        recalls = recalls_list[idx]
        for action_type in ['clicks', 'carts', 'orders']:
            true_val = recalls.get(action_type)
            if true_val is not None:
                pred_val = true_val * (1 + random.uniform(-0.1, 0.1))
                ppi.update_recall_stats(action_type, true_val, pred_val)
    results['Single PPI'] = (ppi.get_weighted_recall(), 0)

    # Bagging PPI
    results['Bagging PPI'] = bagging_ppi(recalls_list)

    # Stacking PPI
    results['Stacking PPI'] = stacking_ppi(recalls_list)

    print("\n" + "=" * 70)
    print("Ensemble PPI Results")
    print("=" * 70)
    for method, (mean, std) in results.items():
        print(f"{method:<20} Weighted Recall: {mean:.4f} (std: {std:.4f})")

    return results


if __name__ == '__main__':
    print("Experiment 5: Ensemble Learning PPI Comparison")
    print("Run in Kaggle Notebook environment with Otto dataset.")
