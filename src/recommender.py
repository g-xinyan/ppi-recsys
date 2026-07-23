"""
PPI 推荐器模块
Dynamic PPI Recommender with adaptive strength and bias correction
"""

import numpy as np
import scipy.stats as stats

from .ppi import ppi_mean_pointestimate, ppi_mean_ci, ppi_optimal_lam


class PPIRecommender:
    """
    动态 PPI 推荐器

    通过在线积累标注/预测数据对，动态调整 PPI 强度参数，
    实现偏差校正和置信区间估计。

    Parameters
    ----------
    min_strength : float
        PPI 强度下界
    max_strength : float
        PPI 强度上界
    """

    def __init__(self, min_strength=0.02, max_strength=0.35):
        self.min_strength = min_strength
        self.max_strength = max_strength
        self.storage = {
            'clicks': {'Y': [], 'Yhat': [], 'Yhat_unlabeled': []},
            'carts': {'Y': [], 'Yhat': [], 'Yhat_unlabeled': []},
            'orders': {'Y': [], 'Yhat': [], 'Yhat_unlabeled': []},
        }
        self.lam = {'clicks': 1.0, 'carts': 1.0, 'orders': 1.0}
        self.recall_stats = {
            'clicks': {'labeled': [], 'predicted': [], 'strength': []},
            'carts': {'labeled': [], 'predicted': [], 'strength': []},
            'orders': {'labeled': [], 'predicted': [], 'strength': []},
        }
        self.score_correction = {'clicks': 0.0, 'carts': 0.0, 'orders': 0.0}

    def update_recall_stats(self, action_type, labeled_recall, predicted_recall,
                            predicted_unlabeled=None):
        """更新召回统计信息"""
        self.recall_stats[action_type]['labeled'].append(labeled_recall)
        self.recall_stats[action_type]['predicted'].append(predicted_recall)

        self.storage[action_type]['Y'].append(labeled_recall)
        self.storage[action_type]['Yhat'].append(predicted_recall)

        if predicted_unlabeled is not None:
            self.storage[action_type]['Yhat_unlabeled'].append(predicted_unlabeled)
        else:
            self.storage[action_type]['Yhat_unlabeled'].append(predicted_recall)

        # 积累足够数据后更新 lambda
        if len(self.storage[action_type]['Y']) >= 10:
            Y = np.array(self.storage[action_type]['Y'])
            Yhat = np.array(self.storage[action_type]['Yhat'])
            Yhat_unlabeled = np.array(self.storage[action_type]['Yhat_unlabeled'])
            self.lam[action_type] = ppi_optimal_lam(Y, Yhat, Yhat_unlabeled)

            strength = self._compute_dynamic_strength(action_type)
            self.recall_stats[action_type]['strength'].append(strength)

            bias = (np.mean(self.recall_stats[action_type]['predicted'])
                    - np.mean(self.recall_stats[action_type]['labeled']))
            self.score_correction[action_type] = bias * strength

    def _compute_dynamic_strength(self, action_type):
        """计算动态 PPI 强度"""
        labeled = np.array(self.recall_stats[action_type]['labeled'])
        predicted = np.array(self.recall_stats[action_type]['predicted'])

        if len(labeled) < 10:
            return self.min_strength

        errors = predicted - labeled
        bias = np.mean(errors)
        bias_std = np.std(errors)
        correlation = np.corrcoef(labeled, predicted)[0, 1] if len(labeled) > 1 else 0
        correlation_factor = max(0, min(1, correlation))
        stability_factor = 1 / (1 + bias_std) if bias_std > 0 else 1

        strength = (self.min_strength
                    + (self.max_strength - self.min_strength)
                    * correlation_factor * stability_factor)

        if abs(bias) > 0.2:
            strength *= 0.5

        return np.clip(strength, self.min_strength, self.max_strength)

    def get_current_strength(self, action_type):
        """获取当前 PPI 强度"""
        if self.recall_stats[action_type]['strength']:
            return self.recall_stats[action_type]['strength'][-1]
        return self.min_strength

    def get_correction(self, action_type):
        """获取偏差校正值"""
        return self.score_correction.get(action_type, 0.0)

    def correct_score(self, action_type, original_score):
        """校正评分"""
        correction = self.get_correction(action_type)
        return max(0, original_score - correction)

    def compute_ppi_recall(self, action_type):
        """计算 PPI 召回率点估计"""
        if len(self.storage[action_type]['Y']) < 5:
            if self.recall_stats[action_type]['predicted']:
                return np.mean(self.recall_stats[action_type]['predicted'])
            return 0

        Y = np.array(self.storage[action_type]['Y'])
        Yhat = np.array(self.storage[action_type]['Yhat'])
        Yhat_unlabeled = np.array(self.storage[action_type]['Yhat_unlabeled'])

        return ppi_mean_pointestimate(Y, Yhat, Yhat_unlabeled,
                                      lam=self.lam[action_type])

    def compute_confidence_interval(self, action_type, alpha=0.05):
        """计算 PPI 置信区间"""
        if len(self.storage[action_type]['Y']) < 10:
            return 0, 1

        Y = np.array(self.storage[action_type]['Y'])
        Yhat = np.array(self.storage[action_type]['Yhat'])
        Yhat_unlabeled = np.array(self.storage[action_type]['Yhat_unlabeled'])

        lower, upper = ppi_mean_ci(Y, Yhat, Yhat_unlabeled,
                                   alpha=alpha, lam=self.lam[action_type])
        return np.clip(lower, 0, 1), np.clip(upper, 0, 1)

    def get_classical_interval(self, action_type, alpha=0.05):
        """计算经典置信区间 (仅使用标注数据)"""
        labeled = np.array(self.recall_stats[action_type]['labeled'])
        if len(labeled) < 10:
            return 0, 1
        mean = np.mean(labeled)
        se = np.std(labeled) / np.sqrt(len(labeled))
        z_score = stats.norm.ppf(1 - alpha / 2)
        lower = mean - z_score * se
        upper = mean + z_score * se
        return np.clip(lower, 0, 1), np.clip(upper, 0, 1)

    def get_weighted_recall(self):
        """计算加权召回率 (clicks:0.10, carts:0.30, orders:0.60)"""
        return (0.10 * self.compute_ppi_recall('clicks')
                + 0.30 * self.compute_ppi_recall('carts')
                + 0.60 * self.compute_ppi_recall('orders'))

    def get_weighted_confidence_interval(self, alpha=0.05):
        """计算加权召回率的置信区间 (Delta 方法)"""
        clicks_ci = self.compute_confidence_interval('clicks', alpha)
        carts_ci = self.compute_confidence_interval('carts', alpha)
        orders_ci = self.compute_confidence_interval('orders', alpha)

        weights = {'clicks': 0.10, 'carts': 0.30, 'orders': 0.60}

        lower = (weights['clicks'] * clicks_ci[0]
                 + weights['carts'] * carts_ci[0]
                 + weights['orders'] * orders_ci[0])
        upper = (weights['clicks'] * clicks_ci[1]
                 + weights['carts'] * carts_ci[1]
                 + weights['orders'] * orders_ci[1])

        return (lower, upper)
