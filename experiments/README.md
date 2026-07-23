# 实验说明

本目录包含论文中所有实验的独立脚本, 对应 Kaggle notebook `notebooks/otto-interval.ipynb` 中的各个实验模块。

## 实验列表

| 实验 | 文件 | 说明 |
|------|------|------|
| 1 | `exp1_strategy_comparison.py` | 六种召回策略的 PPI 评估对比 |
| 2 | `exp1_strategy_comparison.py` | 不同会话长度的 PPI 评估对比 |
| 3 | `exp1_strategy_comparison.py` | 不同时间窗口的 PPI 评估对比 |
| 4 | `exp4_ppi_vs_classical.py` | PPI vs Classical vs Imputation 对比 |
| 5 | `exp5_ensemble_ppi.py` | 集成学习 PPI 对比 (Bagging/Stacking/Cross) |
| 6 | `exp6_cold_start.py` | 冷启动场景下的 PPI 分析 |
| 7 | `exp7_distribution_shift.py` | 分布漂移下的 PPI 分析 |
| 8 | `exp8_weighted_ppi.py` | 加权 PPI + 动态权重 PPI 对比 |

## 运行说明

所有实验依赖 Otto Multi-Behavior Recommendation Dataset。推荐在 Kaggle Notebook 环境中运行, 或直接使用 `notebooks/otto-interval.ipynb` 获取完整的可运行版本。
