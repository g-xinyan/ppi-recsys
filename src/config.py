"""
配置参数模块
Configuration for Otto Recommender System with PPI
"""

from pathlib import Path


class Config:
    """全局配置参数"""

    # ==================== 数据路径 ====================
    DATA_DIR = Path("/kaggle/input/competitions/otto-recommender-system")
    TRAIN_PATH = DATA_DIR / "train.jsonl"
    TEST_PATH = DATA_DIR / "test.jsonl"
    OUT_PATH = Path("/kaggle/working/submission.csv")

    # ==================== 召回参数 ====================
    MAX_TRAIN_SESSIONS_FOR_COVISIT = 300_000
    MAX_SESSION_AIDS = 30
    CLICK_WINDOW = 5
    BUY_WINDOW = 5
    TRANSFER_WINDOW = 10
    TOPN_NEIGHBORS = 120
    TOP_POPULAR = 150

    # ==================== 锚点参数 ====================
    ANCHORS_PER_SESSION = 8
    BUY_ANCHORS_PER_SESSION = 8
    MULTI_LEVEL_ANCHORS = 4

    # ==================== 权重分配 ====================
    HISTORY_WEIGHTS = {0: 1.0, 1: 6.0, 2: 5.0}
    NEIGHBOR_WEIGHTS = {'click': 0.8, 'cart': 0.6, 'order': 0.4}
    TRANSFER_WEIGHTS = {'click_to_cart': 0.5, 'cart_to_order': 0.7}
    LONG_TERM_WEIGHT = 0.4
    CATEGORY_WEIGHT = 0.3

    # ==================== PPI 参数 ====================
    PPI_MIN_STRENGTH = 0.02
    PPI_MAX_STRENGTH = 0.35
    USE_PPI_FOR_SCORING = True

    # ==================== 实验参数 ====================
    COVIST_SESSIONS = 8000
    TRAIN_SESSIONS = 6000
    VAL_SESSIONS = 4000
    NUM_TRIALS = 8
    NS = [100, 200, 500, 1000, 2000, 3000]

    ALPHA = 0.05
    POWER_TARGET = 0.8
    NULL_HYPOTHESIS = {'clicks': 0.35, 'carts': 0.45, 'orders': 0.55}

    RANDOM_SEED = 42
    TYPE_STR_TO_INT = {"clicks": 0, "carts": 1, "orders": 2}
    TYPE_WEIGHT = {0: 1, 1: 6, 2: 5}

    # ==================== 召回策略配置 ====================
    RECALL_STRATEGIES = {
        'history_only': 'History Only',
        'neighbor_only': 'Neighbor Only',
        'transfer_only': 'Transfer Only',
        'category_only': 'Category Only',
        'long_term_only': 'Long Term Only',
        'full_strategy': 'Full Strategy',
    }

    # ==================== 会话长度分组 ====================
    SESSION_LENGTH_GROUPS = {
        'short': (0, 10, 'Short (<10)'),
        'medium': (10, 30, 'Medium (10-30)'),
        'long': (30, float('inf'), 'Long (>30)'),
    }

    # ==================== 时间窗口配置 ====================
    TIME_WINDOWS = {
        'early': (0, 0.33, 'Early 33%'),
        'middle': (0.33, 0.66, 'Middle 33%'),
        'late': (0.66, 1.0, 'Late 33%'),
    }
