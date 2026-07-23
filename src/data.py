"""
数据加载与预处理模块
Data loading, session splitting, and co-visitation matrix construction
"""

import json
import random
from collections import Counter, defaultdict

from tqdm.auto import tqdm

from .config import Config
from .utils import (
    add_pairs,
    counterdict_to_topneighbors,
    event_type_to_int,
    extract_item_category,
    get_recent_unique,
)


def load_sessions(max_sessions):
    """
    加载会话数据

    Parameters
    ----------
    max_sessions : int
        最大加载会话数

    Returns
    -------
    list
        会话列表, 每个会话包含 session_id, events, idx
    """
    print(f"Loading {max_sessions} sessions to cache...")
    sessions = []
    with open(Config.TRAIN_PATH, "r") as f:
        for idx, line in enumerate(tqdm(f, desc="Loading sessions")):
            if idx >= max_sessions:
                break
            obj = json.loads(line)
            events = sorted(obj["events"], key=lambda x: x["ts"])
            sessions.append({'session_id': obj['session'], 'events': events, 'idx': idx})
    print(f"Loaded {len(sessions)} sessions")
    return sessions


def split_sessions(sessions, train_ratio=0.6):
    """
    划分训练集和验证集

    Parameters
    ----------
    sessions : list
        会话列表
    train_ratio : float
        训练集比例

    Returns
    -------
    tuple
        (train_sessions, val_sessions)
    """
    n = len(sessions)
    indices = list(range(n))
    random.shuffle(indices)
    train_size = int(n * train_ratio)
    train_sessions = [sessions[i] for i in indices[:train_size]]
    val_sessions = [sessions[i] for i in indices[train_size:]]
    print(f"Train sessions: {len(train_sessions)}")
    print(f"Validation sessions: {len(val_sessions)}")
    return train_sessions, val_sessions


def build_multiple_co_visitation_matrices(sessions):
    """
    构建多种共现矩阵

    包括 click/cart/order 共现矩阵、click->cart/cart->order 转移矩阵、
    类别映射和长期兴趣。

    Parameters
    ----------
    sessions : list
        会话列表

    Returns
    -------
    dict
        包含所有矩阵和映射的字典
    """
    print("Building multiple co-visitation matrices...")
    matrices = {}

    print("Building click co-visitation matrix...")
    matrices['click'] = _build_matrix_by_type(sessions, [0])

    print("Building cart co-visitation matrix...")
    matrices['cart'] = _build_matrix_by_type(sessions, [1])

    print("Building order co-visitation matrix...")
    matrices['order'] = _build_matrix_by_type(sessions, [2])

    print("Building click->cart transition matrix...")
    matrices['click_to_cart'] = _build_transition_matrix(sessions, [0], [1])

    print("Building cart->order transition matrix...")
    matrices['cart_to_order'] = _build_transition_matrix(sessions, [1], [2])

    print("Building category map...")
    matrices['category_map'] = _build_category_map(sessions)

    print("Building long term interest...")
    matrices['long_term_interest'] = _build_long_term_interest(sessions)

    return matrices


def _build_matrix_by_type(sessions, types):
    """按行为类型构建共现矩阵"""
    counter = defaultdict(Counter)
    for session in tqdm(sessions, desc="Building by type"):
        events = session['events']
        aids = [int(ev["aid"]) for ev in events]
        ev_types = [event_type_to_int(ev["type"]) for ev in events]
        filtered = [(aid, t) for aid, t in zip(aids, ev_types) if t in types]
        if len(filtered) < 2:
            continue
        filtered_aids = [aid for aid, _ in filtered]
        recent_all = get_recent_unique(filtered_aids, None, None, Config.MAX_SESSION_AIDS)
        add_pairs(counter, recent_all, Config.CLICK_WINDOW)
    return counterdict_to_topneighbors(counter, topk=Config.TOPN_NEIGHBORS)


def _build_transition_matrix(sessions, source_types, target_types):
    """构建行为转移矩阵"""
    counter = defaultdict(Counter)
    for session in tqdm(sessions, desc="Building transition matrix"):
        events = session['events']
        aids = [int(ev["aid"]) for ev in events]
        ev_types = [event_type_to_int(ev["type"]) for ev in events]
        source_items = [aid for aid, t in zip(aids, ev_types) if t in source_types]
        target_items = [aid for aid, t in zip(aids, ev_types) if t in target_types]
        if len(source_items) < 2 or len(target_items) < 1:
            continue
        recent_source = get_recent_unique(source_items, None, None, Config.MAX_SESSION_AIDS)
        recent_target = get_recent_unique(target_items, None, None, Config.MAX_SESSION_AIDS)
        for i, src in enumerate(recent_source):
            for j, tgt in enumerate(recent_target[:Config.TRANSFER_WINDOW]):
                if src == tgt:
                    continue
                counter[src][tgt] += 1 / (abs(i - j) + 1)
    return counterdict_to_topneighbors(counter, topk=Config.TOPN_NEIGHBORS // 2)


def _build_category_map(sessions):
    """构建商品类别映射"""
    category_map = defaultdict(list)
    for session in tqdm(sessions, desc="Building category map"):
        events = session['events']
        aids = [int(ev["aid"]) for ev in events]
        for aid in set(aids):
            cat = extract_item_category(aid)
            category_map[cat].append(aid)
    for cat in category_map:
        category_map[cat] = list(set(category_map[cat]))[:200]
    return category_map


def _build_long_term_interest(sessions):
    """构建长期兴趣列表"""
    long_term_counter = Counter()
    for session in tqdm(sessions, desc="Building long term interest"):
        events = session['events']
        aids = [int(ev["aid"]) for ev in events]
        types = [event_type_to_int(ev["type"]) for ev in events]
        for aid, t in zip(aids, types):
            long_term_counter[aid] += Config.TYPE_WEIGHT.get(t, 1)
    return [aid for aid, _ in long_term_counter.most_common(200)]
