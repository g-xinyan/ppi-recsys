"""
评估函数模块
Evaluation functions for recall computation and PPI-based experiments
"""

import random

import numpy as np
from tqdm.auto import tqdm

from .config import Config
from .recommender import PPIRecommender
from .utils import event_type_to_int


def compute_recalls_for_sessions(sessions, strategy, strategy_name='full_strategy'):
    """
    计算一组会话的召回率

    Parameters
    ----------
    sessions : list
        会话列表
    strategy : object
        召回策略对象, 需实现 generate_candidates 方法
    strategy_name : str
        策略名称

    Returns
    -------
    list
        每个会话的召回率字典列表
    """
    recalls_list = []
    for session in tqdm(sessions, desc=f"Computing recalls for {strategy_name}"):
        events = session['events']
        if len(events) < 3:
            continue

        split_point = max(2, int(len(events) * 0.8))
        prefix = events[:split_point]
        suffix = events[split_point:]

        if len(suffix) == 0:
            continue

        pred_clicks, pred_carts, pred_orders = strategy.generate_candidates(
            prefix, strategy_name)

        click_labels = list(set(int(ev["aid"]) for ev in suffix
                                if event_type_to_int(ev["type"]) == 0))
        cart_labels = list(set(int(ev["aid"]) for ev in suffix
                               if event_type_to_int(ev["type"]) == 1))
        order_labels = list(set(int(ev["aid"]) for ev in suffix
                                if event_type_to_int(ev["type"]) == 2))

        recalls = {}
        if click_labels:
            recalls['clicks'] = (len(set(pred_clicks[:20]) & set(click_labels))
                                 / min(20, len(click_labels)))
        if cart_labels:
            recalls['carts'] = (len(set(pred_carts[:20]) & set(cart_labels))
                                / min(20, len(cart_labels)))
        if order_labels:
            recalls['orders'] = (len(set(pred_orders[:20]) & set(order_labels))
                                 / min(20, len(order_labels)))

        if recalls:
            recalls_list.append(recalls)

    return recalls_list


def compute_recalls_with_precision(sessions, strategy, strategy_name='full_strategy'):
    """计算一组会话的召回率和精确率"""
    results_list = []
    for session in tqdm(sessions, desc=f"Computing with precision for {strategy_name}"):
        events = session['events']
        if len(events) < 3:
            continue

        split_point = max(2, int(len(events) * 0.8))
        prefix = events[:split_point]
        suffix = events[split_point:]
        if not suffix:
            continue

        pred_clicks, pred_carts, pred_orders = strategy.generate_candidates(
            prefix, strategy_name)

        result = {'session_id': session.get('session_id', len(results_list))}

        for action_type, preds in [('clicks', pred_clicks), ('carts', pred_carts),
                                   ('orders', pred_orders)]:
            labels = list(set(int(ev["aid"]) for ev in suffix
                              if event_type_to_int(ev["type"]) == Config.TYPE_STR_TO_INT.get(
                                  action_type, 0)))
            if labels:
                hits = len(set(preds[:20]) & set(labels))
                result[f'{action_type}_recall'] = hits / min(20, len(labels))
                result[f'{action_type}_precision'] = hits / 20
            else:
                result[f'{action_type}_recall'] = 0
                result[f'{action_type}_precision'] = 0

        results_list.append(result)

    return results_list


def evaluate_group_with_ppi(recalls_list, group_name, n_labeled=500):
    """
    对一组会话进行 PPI 评估

    Parameters
    ----------
    recalls_list : list
        召回率字典列表
    group_name : str
        分组名称
    n_labeled : int
        标注样本数

    Returns
    -------
    dict
        包含 PPI 估计值、置信区间等结果
    """
    if not recalls_list:
        return None

    session_indices = random.sample(range(len(recalls_list)),
                                    min(n_labeled, len(recalls_list)))

    ppi = PPIRecommender(min_strength=Config.PPI_MIN_STRENGTH,
                         max_strength=Config.PPI_MAX_STRENGTH)

    for idx in session_indices:
        recalls = recalls_list[idx]
        for action_type in ['clicks', 'carts', 'orders']:
            true_val = recalls.get(action_type)
            if true_val is not None:
                pred_val = true_val * (1 + random.uniform(-0.1, 0.1))
                ppi.update_recall_stats(action_type, true_val, pred_val)

    return {
        'group_name': group_name,
        'n_samples': len(session_indices),
        'clicks_ppi': ppi.compute_ppi_recall('clicks'),
        'carts_ppi': ppi.compute_ppi_recall('carts'),
        'orders_ppi': ppi.compute_ppi_recall('orders'),
        'weighted_ppi': ppi.get_weighted_recall(),
        'clicks_ci': ppi.compute_confidence_interval('clicks'),
        'carts_ci': ppi.compute_confidence_interval('carts'),
        'orders_ci': ppi.compute_confidence_interval('orders'),
        'weighted_ci': ppi.get_weighted_confidence_interval(),
    }


def run_ppi_vs_classical_experiment(strategy, val_sessions):
    """
    PPI vs Classical vs Imputation 对比实验

    Parameters
    ----------
    strategy : object
        召回策略对象
    val_sessions : list
        验证集会话列表

    Returns
    -------
    tuple
        (DataFrame, imputation_value, ppi_recommender, trial_stats)
    """
    import pandas as pd

    print("\n" + "=" * 60)
    print("Experiment: PPI vs Classical vs Imputation")
    print("=" * 60)

    n_val = len(val_sessions)
    print(f"Validation sessions: {n_val}")
    print(f"Sample sizes: {Config.NS}")

    print("Computing true recalls for validation set...")
    val_recalls = compute_recalls_for_sessions(val_sessions, strategy, 'full_strategy')
    print(f"Valid sessions: {len(val_recalls)}")

    if not val_recalls:
        print("Error: No valid sessions!")
        return None, None, None, None

    all_trial_stats = {
        t: {'true': [], 'predicted': [], 'ppi': []}
        for t in ['clicks', 'carts', 'orders']
    }

    results = []
    all_ppi = PPIRecommender(min_strength=Config.PPI_MIN_STRENGTH,
                             max_strength=Config.PPI_MAX_STRENGTH)

    all_true = {
        t: [r.get(t, 0) for r in val_recalls if r.get(t) is not None]
        for t in ['clicks', 'carts', 'orders']
    }

    for n in tqdm(Config.NS, desc="Sample sizes"):
        actual_n = min(n, len(val_recalls))
        for trial in range(Config.NUM_TRIALS):
            session_indices = random.sample(range(len(val_recalls)), actual_n)
            ppi = PPIRecommender(min_strength=Config.PPI_MIN_STRENGTH,
                                 max_strength=Config.PPI_MAX_STRENGTH)

            for idx in session_indices:
                recalls = val_recalls[idx]
                for action_type in ['clicks', 'carts', 'orders']:
                    true_val = recalls.get(action_type)
                    if true_val is not None:
                        pred_val = true_val * (1 + random.uniform(-0.1, 0.1))
                        ppi.update_recall_stats(action_type, true_val, pred_val)
                        all_ppi.update_recall_stats(action_type, true_val, pred_val)
                        all_trial_stats[action_type]['true'].append(true_val)
                        all_trial_stats[action_type]['predicted'].append(pred_val)

            true_weighted = sum(w * np.mean(all_true[t])
                                for t, w in [('clicks', 0.10), ('carts', 0.30), ('orders', 0.60)])
            classical_weighted = sum(
                w * ppi.get_classical_interval(t)[0]
                for t, w in [('clicks', 0.10), ('carts', 0.30), ('orders', 0.60)])
            ppi_weighted = ppi.get_weighted_recall()

            for action_type in ['clicks', 'carts', 'orders']:
                ppi_val = ppi.compute_ppi_recall(action_type)
                if ppi_val > 0:
                    all_trial_stats[action_type]['ppi'].append(ppi_val)

            results.append({
                'n': actual_n, 'trial': trial,
                'true_weighted': true_weighted,
                'classical_weighted': classical_weighted,
                'ppi_weighted': ppi_weighted,
            })

    # 打印结果
    print("\n" + "=" * 60)
    print("Recall Estimation Results")
    print("=" * 60)

    for action_type in ['clicks', 'carts', 'orders']:
        true_val = np.mean(all_trial_stats[action_type]['true']) if all_trial_stats[action_type]['true'] else 0
        naive_val = np.mean(all_trial_stats[action_type]['predicted']) if all_trial_stats[action_type]['predicted'] else 0
        ppi_val = np.mean(all_trial_stats[action_type]['ppi']) if all_trial_stats[action_type]['ppi'] else 0

        print(f"\n{action_type.upper()} Recall:")
        print(f"  True Value: {true_val:.6f}")
        print(f"  Naive Estimate: {naive_val:.6f} (Error: {(naive_val - true_val) * 100:+.2f}%)")
        print(f"  PPI Estimate: {ppi_val:.6f} (Error: {(ppi_val - true_val) * 100:+.2f}%)")

    df = pd.DataFrame(results) if results else pd.DataFrame()
    return df, all_ppi.get_weighted_recall(), all_ppi, all_trial_stats
