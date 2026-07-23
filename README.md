# Prediction-Powered Inference for Recommendation Systems

**预测驱动推断在推荐系统中的应用研究**

[中文说明](#中文说明) | [English](#english)

---

## 中文说明

### 项目简介

本项目是四川大学统计学（数据科学与大数据技术方向）2022级本科毕业论文的研究代码与论文全文。

研究围绕 **预测驱动推断（Prediction-Powered Inference, PPI）** 在推荐系统召回评估中的应用展开。PPI 是一种新兴的统计推断框架，通过小规模标注数据对大规模未标注数据上的预测结果进行偏差校正，从而平衡统计可靠性与推断效率。

### 主要工作

- 将 PPI 方法应用于 Otto 推荐系统的召回策略评估，通过构造置信区间量化估计的不确定性
- 从数据特征、模型结构、应用场景三个维度分析影响 PPI 效果的关键因素
- 提出集成 PPI 框架（Bagging-PPI、Stacking-PPI、Cross-PPI）降低单一预测器偏差风险
- 针对协变量漂移、标签漂移、时间漂移提出校正方法
- 分析了冷启动场景下 PPI 的表现与局限性

### 项目结构

```
ppi-recsys/
├── README.md                 # 项目说明
├── requirements.txt          # Python 依赖
├── thesis/
│   ├── thesis.pdf            # 毕业论文全文 (中文)
│   └── latex/                # LaTeX 源码
│       ├── main.tex
│       ├── scuthesis.sty
│       └── images/
├── src/                      # 核心代码
│   ├── config.py             # 配置参数
│   ├── ppi.py                # PPI 核心算法
│   ├── recommender.py        # PPI 推荐器
│   ├── utils.py              # 工具函数
│   ├── data.py               # 数据加载与处理
│   ├── model.py              # 双塔模型 (PyTorch)
│   └── evaluation.py         # 评估函数
├── experiments/              # 实验脚本
│   ├── exp1_strategy_comparison.py
│   ├── exp2_session_length.py
│   ├── exp3_time_window.py
│   ├── exp4_ppi_vs_classical.py
│   ├── exp5_ensemble_ppi.py
│   ├── exp6_cold_start.py
│   ├── exp7_distribution_shift.py
│   └── exp8_weighted_ppi.py
├── notebooks/
│   └── otto-interval.ipynb   # Kaggle 原始 notebook
├── figures/                  # 实验图表输出
└── output/                   # 运行结果输出
```

### 数据集

本项目使用 [Kaggle - Otto Multi-Behavior Recommendation Dataset](https://www.kaggle.com/competitions/otto-recommender-system/data)。请下载数据集并放置到对应路径，或在 `src/config.py` 中修改 `DATA_DIR` 路径。

### 运行环境

- Python >= 3.8
- PyTorch >= 1.12
- 推荐在 Kaggle Notebook 环境中运行（含 GPU 支持）

### 快速开始

```bash
pip install -r requirements.txt

# 运行单个实验
python experiments/exp1_strategy_comparison.py

# 或使用 Jupyter Notebook
jupyter notebook notebooks/otto-interval.ipynb
```

### 引用

```bibtex
@thesis{guo2026ppi,
  title={预测驱动推断在推荐系统中的应用研究},
  author={郭芯妍},
  school={四川大学数学学院},
  year={2026},
  advisor={周杰}
}
```

---

## English

### Overview

This repository contains the research code and full text of a bachelor's thesis from Sichuan University, focusing on **Prediction-Powered Inference (PPI)** applied to recall evaluation in recommendation systems.

PPI is a statistical inference framework that uses a small amount of labeled data to correct systematic biases in predictions made on large-scale unlabeled data, balancing statistical reliability with inference efficiency.

### Key Contributions

- Applied PPI to recall strategy evaluation on the Otto recommendation dataset, constructing confidence intervals to quantify estimation uncertainty
- Analyzed key factors affecting PPI performance from three dimensions: data characteristics, model structure, and application scenarios
- Proposed an ensemble PPI framework (Bagging-PPI, Stacking-PPI, Cross-PPI) to reduce single-predictor bias risk
- Developed correction methods for covariate shift, label shift, and temporal drift
- Investigated PPI performance and limitations in cold-start scenarios

### Dataset

This project uses the [Kaggle Otto Multi-Behavior Recommendation Dataset](https://www.kaggle.com/competitions/otto-recommender-system/data). Download the dataset and update `DATA_DIR` in `src/config.py`.

### Quick Start

```bash
pip install -r requirements.txt
python experiments/exp1_strategy_comparison.py
```

### Citation

```bibtex
@thesis{guo2026ppi,
  title={Prediction-Powered Inference for Recommendation Systems},
  author={Xinyan Guo},
  school={School of Mathematics, Sichuan University},
  year={2026},
  advisor={Jie Zhou}
}
```

### License

This project is licensed under the MIT License.
