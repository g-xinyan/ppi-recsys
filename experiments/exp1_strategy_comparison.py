"""
实验 1-3: 召回策略对比、会话长度分析、时间窗口分析
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.config import Config
from src.data import load_sessions, split_sessions, build_multiple_co_visitation_matrices
from src.evaluation import (
    compute_recalls_for_sessions,
    evaluate_group_with_ppi,
)


def run_strategy_comparison(strategy, val_sessions):
    """
    实验 1: 不同召回策略的 PPI 评估对比

    比较 history_only, neighbor_only, transfer_only, category_only,
    long_term_only, full_strategy 六种策略的加权召回率。
    """
    print("\n" + "=" * 60)
    print("Experiment 1: PPI Evaluation of Different Recall Strategies")
    print("=" * 60)

    results = []
    for strategy_key, strategy_name in Config.RECALL_STRATEGIES.items():
        print(f"\nEvaluating strategy: {strategy_name}")
        recalls_list = compute_recalls_for_sessions(val_sessions, strategy, strategy_key)
        print(f"Valid sessions: {len(recalls_list)}")

        result = evaluate_group_with_ppi(recalls_list, strategy_name, n_labeled=500)
        if result:
            results.append(result)

    print("\n" + "=" * 70)
    print("Recall Strategy PPI Evaluation Results (n=500)")
    print("=" * 70)
    print(f"{'Strategy':<20} {'Clicks':<12} {'Carts':<12} {'Orders':<12} {'Weighted':<12}")
    print("-" * 70)
    for r in results:
        print(f"{r['group_name']:<20} {r['clicks_ppi']:.4f}       "
              f"{r['carts_ppi']:.4f}       {r['orders_ppi']:.4f}       "
              f"{r['weighted_ppi']:.4f}")
    print("-" * 70)

    best = max(results, key=lambda x: x['weighted_ppi'])
    print(f"\nBest strategy: {best['group_name']} "
          f"(Weighted Recall: {best['weighted_ppi']:.4f})")

    return results


def run_session_length_comparison(strategy, val_sessions):
    """
    实验 2: 不同会话长度的 PPI 评估对比

    将验证集会话按事件数量分为 short(<10), medium(10-30), long(>30) 三组,
    分别进行 PPI 评估并分析置信区间重叠。
    """
    print("\n" + "=" * 60)
    print("Experiment 2: PPI Evaluation by Session Length")
    print("=" * 60)

    grouped = {'short': [], 'medium': [], 'long': []}
    for session in val_sessions:
        n_events = len(session['events'])
        if n_events < 10:
            grouped['short'].append(session)
        elif n_events < 30:
            grouped['medium'].append(session)
        else:
            grouped['long'].append(session)

    for k, v in grouped.items():
        print(f"{Config.SESSION_LENGTH_GROUPS[k][2]}: {len(v)} sessions")

    results = []
    for group_key, group_sessions in grouped.items():
        if not group_sessions:
            continue
        group_name = Config.SESSION_LENGTH_GROUPS[group_key][2]
        recalls_list = compute_recalls_for_sessions(group_sessions, strategy, 'full_strategy')
        result = evaluate_group_with_ppi(recalls_list, group_name, n_labeled=300)
        if result:
            results.append(result)

    print("\n" + "=" * 70)
    print("Session Length PPI Evaluation Results")
    print("=" * 70)
    for r in results:
        print(f"{r['group_name']:<20} Weighted: {r['weighted_ppi']:.4f}  "
              f"CI: [{r['weighted_ci'][0]:.4f}, {r['weighted_ci'][1]:.4f}]")

    return results


def run_time_window_comparison(strategy, val_sessions):
    """
    实验 3: 不同时间窗口的 PPI 评估对比

    将验证集按时间顺序分为 early/middle/late 三段,
    分析时间漂移对 PPI 评估的影响。
    """
    print("\n" + "=" * 60)
    print("Experiment 3: PPI Evaluation by Time Window")
    print("=" * 60)

    sorted_sessions = sorted(val_sessions, key=lambda x: x['idx'])
    n = len(sorted_sessions)

    results = []
    for window_key, (start_ratio, end_ratio, window_name) in Config.TIME_WINDOWS.items():
        start_idx = int(n * start_ratio)
        end_idx = int(n * end_ratio)
        window_sessions = sorted_sessions[start_idx:end_idx]

        print(f"\nEvaluating window: {window_name} (sessions {start_idx}-{end_idx})")
        recalls_list = compute_recalls_for_sessions(window_sessions, strategy, 'full_strategy')
        result = evaluate_group_with_ppi(recalls_list, window_name, n_labeled=300)
        if result:
            results.append(result)

    print("\n" + "=" * 70)
    print("Time Window PPI Evaluation Results")
    print("=" * 70)
    for r in results:
        print(f"{r['group_name']:<20} Weighted: {r['weighted_ppi']:.4f}  "
              f"CI: [{r['weighted_ci'][0]:.4f}, {r['weighted_ci'][1]:.4f}]")

    return results


if __name__ == '__main__':
    print("Experiments 1-3 require the full Otto dataset and recall strategy implementation.")
    print("Please run in Kaggle Notebook environment or provide data path.")
    print("See notebooks/otto-interval.ipynb for the complete runnable version.")
