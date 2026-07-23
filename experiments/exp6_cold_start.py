"""
实验 6: 冷启动场景下的 PPI 分析
"""

import sys
import os
import random
from collections import defaultdict

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.config import Config
from src.recommender import PPIRecommender
from src.evaluation import compute_recalls_for_sessions, evaluate_group_with_ppi


def extract_cold_start_sessions(sessions, min_events=5):
    """
    提取冷启动会话 (事件数较少的会话)

    Parameters
    ----------
    sessions : list
        会话列表
    min_events : int
        最少事件数阈值

    Returns
    -------
    list
        冷启动会话列表
    """
    cold_sessions = []
    for session in sessions:
        if len(session['events']) <= min_events:
            cold_sessions.append(session)
    return cold_sessions


def run_cold_start_experiment(strategy, val_sessions):
    """
    实验 6: 冷启动场景下的 PPI 分析

    分析 PPI 在冷启动 (短会话) 场景中的表现,
    对比不同冷启动程度下 PPI 的效果差异。
    """
    print("\n" + "=" * 60)
    print("Experiment 6: Cold-Start PPI Analysis")
    print("=" * 60)

    # 按会话长度分组
    cold_very = [s for s in val_sessions if len(s['events']) <= 5]
    cold_mild = [s for s in val_sessions if 5 < len(s['events']) <= 10]
    warm = [s for s in val_sessions if len(s['events']) > 10]

    print(f"Very cold (<=5 events): {len(cold_very)}")
    print(f"Mild cold (6-10 events): {len(cold_mild)}")
    print(f"Warm (>10 events): {len(warm)}")

    groups = {
        'Very Cold (<=5)': cold_very,
        'Mild Cold (6-10)': cold_mild,
        'Warm (>10)': warm,
    }

    results = []
    for name, group_sessions in groups.items():
        if len(group_sessions) < 10:
            print(f"\nSkipping {name}: too few sessions")
            continue
        recalls_list = compute_recalls_for_sessions(group_sessions, strategy, 'full_strategy')
        result = evaluate_group_with_ppi(recalls_list, name, n_labeled=min(300, len(recalls_list)))
        if result:
            results.append(result)

    print("\n" + "=" * 70)
    print("Cold-Start PPI Results")
    print("=" * 70)
    for r in results:
        print(f"{r['group_name']:<25} Weighted: {r['weighted_ppi']:.4f}  "
              f"CI: [{r['weighted_ci'][0]:.4f}, {r['weighted_ci'][1]:.4f}]")

    return results


if __name__ == '__main__':
    print("Experiment 6: Cold-Start PPI Analysis")
    print("Run in Kaggle Notebook environment with Otto dataset.")
