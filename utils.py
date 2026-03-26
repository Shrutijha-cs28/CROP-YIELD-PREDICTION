# ============================================================
# FILE: src/utils.py
# ============================================================
"""Utility functions."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

def calculate_metrics(y_true, y_pred):
    """Calculate evaluation metrics.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Dictionary of metrics
    """
    y_pred = np.array(y_pred).flatten()
    
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    metrics = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2_Score': r2,
        'MAPE': mape
    }
    
    return metrics

def plot_predictions_comparison(y_true, y_pred, title='Yield Predictions'):
    """Plot predictions comparison.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        title: Plot title
    """
    y_pred = np.array(y_pred).flatten()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Actual vs Predicted
    axes[0, 0].scatter(y_true, y_pred, alpha=0.6, s=50)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    axes[0, 0].set_xlabel('Actual Yield')
    axes[0, 0].set_ylabel('Predicted Yield')
    axes[0, 0].set_title('Actual vs Predicted')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Residuals
    residuals = y_true - y_pred
    axes[0, 1].scatter(y_pred, residuals, alpha=0.6, s=50)
    axes[0, 1].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[0, 1].set_xlabel('Predicted Yield')
    axes[0, 1].set_ylabel('Residuals')
    axes[0, 1].set_title('Residual Plot')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Residuals Distribution
    axes[1, 0].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    axes[1, 0].set_xlabel('Residuals')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Distribution of Residuals')
    axes[1, 0].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Time series comparison
    indices = np.arange(len(y_true))
    axes[1, 1].plot(indices, y_true, label='Actual', linewidth=2, marker='o', markersize=4)
    axes[1, 1].plot(indices, y_pred, label='Predicted', linewidth=2, marker='s', markersize=4)
    axes[1, 1].set_xlabel('Sample Index')
    axes[1, 1].set_ylabel('Yield')
    axes[1, 1].set_title('Time Series Comparison')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

def print_metrics_report(metrics):
    """Print formatted metrics report.
    
    Args:
        metrics: Dictionary of metrics
    """
    print("\n" + "="*50)
    print("MODEL EVALUATION METRICS")
    print("="*50)
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name:.<35} {metric_value:>10.4f}")
    print("="*50 + "\n")

def save_scaler(scaler, filepath):
    """Save scaler object.
    
    Args:
        scaler: StandardScaler object
        filepath: Path to save scaler
    """
    joblib.dump(scaler, filepath)
    print(f"Scaler saved to {filepath}")

def load_scaler(filepath):
    """Load scaler object.
    
    Args:
        filepath: Path to scaler file
        
    Returns:
        Loaded scaler
    """
    scaler = joblib.load(filepath)
    print(f"Scaler loaded from {filepath}")
    return scaler
