"""
工具函数模块
Utility functions for data processing and scoring
"""

import math
from collections import Counter, defaultdict

from tqdm.auto import tqdm

from .config import Config


def event_type_to_int(t):
    """将事件类型字符串转换为整数编码"""
    if isinstance(t, str):
        return Config.TYPE_STR_TO_INT[t]
    return int(t)


def count_lines(path):
    """统计文件行数"""
    cnt = 0
    with open(path, "r") as f:
        for _ in f:
            cnt += 1
    return cnt


def extract_item_category(aid):
    """从 aid 提取商品类别"""
    return aid // 10000


def get_recent_unique(aids, types=None, allowed_types=None, max_len=30):
    """获取最近的唯一商品列表"""
    seen = set()
    result = []
    if types is None:
        iterator = zip(reversed(aids), [None] * len(aids))
    else:
        iterator = zip(reversed(aids), reversed(types))
    for aid, t in iterator:
        if allowed_types is not None and t not in allowed_types:
            continue
        if aid in seen:
            continue
        seen.add(aid)
        result.append(aid)
        if len(result) >= max_len:
            break
    return result


def add_pairs(counter_dict, items, window):
    """添加共现对到计数器"""
    n = len(items)
    for i in range(n):
        a = items[i]
        for j in range(i + 1, min(n, i + 1 + window)):
            b = items[j]
            if a == b:
                continue
            counter_dict[a][b] += 1
            counter_dict[b][a] += 1


def counterdict_to_topneighbors(counter_dict, topk=120):
    """将共现计数器转换为 top-k 邻居字典"""
    out = {}
    for aid, ctr in tqdm(counter_dict.items(), desc="Trim neighbors"):
        top_items = ctr.most_common(topk * 2)
        weighted_items = []
        for i, (nbr, weight) in enumerate(top_items):
            decay = 1 - (i / (topk * 2)) * 0.5
            weighted_items.append((nbr, weight * decay))
        weighted_items.sort(key=lambda x: -x[1])
        out[aid] = [x[0] for x in weighted_items[:topk]]
    return out


def add_self_scores_improved(scores, aids, types, type_weights):
    """添加历史行为自评分"""
    for pos, (aid, t) in enumerate(zip(reversed(aids), reversed(types))):
        recency = math.exp(-pos / 8)
        position_weight = 1 + (1 - pos / max(1, len(aids))) * 0.5
        weight = type_weights.get(t, 1.0) if isinstance(type_weights, dict) else type_weights
        scores[aid] += recency * weight * position_weight


def add_neighbor_scores_improved(scores, anchors, neighbor_map, mult, levels=3):
    """添加多阶邻居评分"""
    current_anchors = anchors
    for level in range(levels):
        decay = mult * (0.6 ** level)
        new_anchors = []
        for a_pos, aid in enumerate(current_anchors):
            nbrs = neighbor_map.get(aid, [])
            for n_pos, nbr in enumerate(nbrs[:15]):
                if nbr == aid:
                    continue
                level_weight = 1 / ((a_pos + 1) * (n_pos + 1) * (level + 1))
                scores[nbr] += decay * level_weight
                new_anchors.append(nbr)
        current_anchors = new_anchors[:Config.MULTI_LEVEL_ANCHORS]


def add_category_scores(scores, aids, category_map):
    """添加同类商品评分"""
    recent_aids = aids[-8:] if len(aids) >= 8 else aids
    recent_categories = [extract_item_category(aid) for aid in recent_aids]
    for cat in set(recent_categories):
        similar_items = category_map.get(cat, [])
        for aid in similar_items[:80]:
            scores[aid] += Config.CATEGORY_WEIGHT


def add_long_term_scores(scores, long_term_items):
    """添加长期兴趣评分"""
    for aid in long_term_items[:50]:
        scores[aid] += Config.LONG_TERM_WEIGHT


def topk_from_scores(scores, fallback, k=20):
    """从评分字典中提取 top-k 商品，带类别多样性约束"""
    result = []
    seen = set()
    category_count = defaultdict(int)
    sorted_items = sorted(scores.items(), key=lambda x: (-x[1], x[0]))

    for aid, score in sorted_items:
        if aid in seen:
            continue
        cat = extract_item_category(aid)
        if category_count[cat] >= 3:
            continue
        seen.add(aid)
        category_count[cat] += 1
        result.append(aid)
        if len(result) == k:
            return result

    for aid in fallback:
        if aid in seen:
            continue
        seen.add(aid)
        result.append(aid)
        if len(result) == k:
            return result
    return result[:k]
