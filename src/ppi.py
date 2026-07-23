"""
PPI 核心算法模块
Prediction-Powered Inference core functions

实现:
- PPI 点估计
- PPI 置信区间构造
- 最优 lambda 计算 (PPI++)
- 交叉 PPI (Cross-PPI)
"""

import numpy as np
import scipy.stats as stats


def ppi_mean_pointestimate(Y, Yhat, Yhat_unlabeled, lam=1):
    """
    PPI 点估计

    Parameters
    ----------
    Y : array-like
        标注数据上的真实值 (labeled ground truth)
    Yhat : array-like
        标注数据上的预测值 (predictions on labeled data)
    Yhat_unlabeled : array-like
        未标注数据上的预测值 (predictions on unlabeled data)
    lam : float
        权重参数

    Returns
    -------
    float
        PPI 点估计值
    """
    Y = np.array(Y)
    Yhat = np.array(Yhat)
    Yhat_unlabeled = np.array(Yhat_unlabeled)
    return np.mean(Yhat_unlabeled * lam) + np.mean(Y - lam * Yhat)


def ppi_mean_ci(Y, Yhat, Yhat_unlabeled, alpha=0.05, lam=1, alternative='two-sided'):
    """
    PPI 置信区间

    Parameters
    ----------
    Y : array-like
        标注数据上的真实值
    Yhat : array-like
        标注数据上的预测值
    Yhat_unlabeled : array-like
        未标注数据上的预测值
    alpha : float
        显著性水平 (默认 0.05 对应 95% 置信区间)
    lam : float
        权重参数
    alternative : str
        'two-sided', 'larger', 或 'smaller'

    Returns
    -------
    tuple
        (lower, upper) 置信区间边界
    """
    Y = np.array(Y)
    Yhat = np.array(Yhat)
    Yhat_unlabeled = np.array(Yhat_unlabeled)

    n = len(Y)
    N = len(Yhat_unlabeled)

    pointest = ppi_mean_pointestimate(Y, Yhat, Yhat_unlabeled, lam)

    imputed_std = np.std(lam * Yhat_unlabeled) / np.sqrt(N) if N > 0 else 0
    rectifier_std = np.std(Y - lam * Yhat) / np.sqrt(n) if n > 0 else 0
    se = np.sqrt(imputed_std**2 + rectifier_std**2)

    if alternative == 'two-sided':
        z = stats.norm.ppf(1 - alpha / 2)
        return pointest - z * se, pointest + z * se
    elif alternative == 'larger':
        z = stats.norm.ppf(1 - alpha)
        return pointest - z * se, np.inf
    else:
        z = stats.norm.ppf(1 - alpha)
        return -np.inf, pointest + z * se


def ppi_optimal_lam(Y, Yhat, Yhat_unlabeled):
    """
    计算最优 lambda (PPI++)

    通过最小化渐近方差来确定最优权重 lambda

    Parameters
    ----------
    Y : array-like
        标注数据上的真实值
    Yhat : array-like
        标注数据上的预测值
    Yhat_unlabeled : array-like
        未标注数据上的预测值

    Returns
    -------
    float
        最优 lambda 值, 裁剪到 [0, 1]
    """
    Y = np.array(Y)
    Yhat = np.array(Yhat)
    Yhat_unlabeled = np.array(Yhat_unlabeled)

    n = len(Y)
    N = len(Yhat_unlabeled)

    if n < 2 or N < 2:
        return 1.0

    var_Yhat = np.var(Yhat)
    var_Yhat_unlabeled = np.var(Yhat_unlabeled)
    cov_Y_Yhat = np.cov(Y, Yhat)[0, 1]

    if var_Yhat + var_Yhat_unlabeled * n / N > 0:
        lam = cov_Y_Yhat / (var_Yhat + var_Yhat_unlabeled * n / N)
    else:
        lam = 1.0
    return np.clip(lam, 0, 1)


def crossppi_mean_ci(Y, Yhat_matrix, Yhat_unlabeled_matrix, alpha=0.05):
    """
    交叉 PPI (Cross-PPI) 置信区间

    使用多个预测模型的平均值进行 PPI 推断

    Parameters
    ----------
    Y : array-like
        标注数据上的真实值
    Yhat_matrix : array-like, shape (n_models, n_samples)
        多个模型在标注数据上的预测值
    Yhat_unlabeled_matrix : array-like, shape (n_models, n_unlabeled)
        多个模型在未标注数据上的预测值
    alpha : float
        显著性水平

    Returns
    -------
    tuple
        (lower, upper) 置信区间边界
    """
    Y = np.array(Y)
    Yhat_matrix = np.array(Yhat_matrix)
    Yhat_unlabeled_matrix = np.array(Yhat_unlabeled_matrix)

    Yhat_avg = np.mean(Yhat_matrix, axis=1)
    Yhat_unlabeled_avg = np.mean(Yhat_unlabeled_matrix, axis=1)

    return ppi_mean_ci(Y, Yhat_avg, Yhat_unlabeled_avg, alpha=alpha, lam=1)
