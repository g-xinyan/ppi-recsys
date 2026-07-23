"""
实验 4: PPI vs Classical vs Imputation 对比
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.config import Config
from src.evaluation import run_ppi_vs_classical_experiment


if __name__ == '__main__':
    print("Experiment 4: PPI vs Classical vs Imputation Comparison")
    print("=" * 60)
    print("This experiment compares three estimation methods:")
    print("  - PPI (Prediction-Powered Inference)")
    print("  - Classical (only labeled data)")
    print("  - Imputation (only predictions on unlabeled data)")
    print()
    print("Run in Kaggle Notebook environment with Otto dataset.")
    print("See notebooks/otto-interval.ipynb for the complete version.")
