"""
Visualization module for model evaluation results
Generates research-grade plots for ROC curves and confusion matrices
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_curve, confusion_matrix, auc
from pathlib import Path

# Set style for research-quality plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def plot_roc_curve(y_test, y_test_prob, save_path='artifacts/roc_curve.png'):
    """
    Generate ROC curve visualization

    Args:
        y_test: True labels
        y_test_prob: Predicted probabilities for positive class
        save_path: Output file path

    Returns:
        Figure object
    """
    # Compute ROC curve points
    fpr, tpr, thresholds = roc_curve(y_test, y_test_prob)
    roc_auc = auc(fpr, tpr)

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot ROC curve
    ax.plot(fpr, tpr, color='blue', lw=2,
            label=f'ROC Curve (AUC = {roc_auc:.3f})')

    # Plot diagonal (random classifier)
    ax.plot([0, 1], [0, 1], color='gray', lw=1,
            linestyle='--', label='Random Classifier')

    # Styling
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate (Recall)', fontsize=12)
    ax.set_title('ROC Curve - Breast Cancer Classification', fontsize=14, fontweight='bold')
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)

    # Add text annotation with key metrics
    ax.text(0.6, 0.2, f'AUC = {roc_auc:.4f}\nExcellent discrimination',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved ROC curve to {save_path}")

    return fig


def plot_confusion_matrix(y_test, y_test_pred, save_path='artifacts/confusion_matrix.png'):
    """
    Generate confusion matrix heatmap

    Args:
        y_test: True labels
        y_test_pred: Predicted labels (hard predictions)
        save_path: Output file path

    Returns:
        Figure object
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)

    # Calculate percentages
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                xticklabels=['Benign', 'Malignant'],
                yticklabels=['Benign', 'Malignant'],
                cbar_kws={'label': 'Count'})
    ax1.set_xlabel('Predicted Label', fontsize=12)
    ax1.set_ylabel('True Label', fontsize=12)
    ax1.set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold')

    # Plot 2: Percentages
    sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Greens', ax=ax2,
                xticklabels=['Benign', 'Malignant'],
                yticklabels=['Benign', 'Malignant'],
                cbar_kws={'label': 'Percentage (%)'})
    ax2.set_xlabel('Predicted Label', fontsize=12)
    ax2.set_ylabel('True Label', fontsize=12)
    ax2.set_title('Confusion Matrix (Percentages)', fontsize=14, fontweight='bold')

    # Add clinical interpretation
    tn, fp, fn, tp = cm.ravel()
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)

    interpretation = (
        f"Clinical Metrics:\n"
        f"Recall (Sensitivity): {recall:.1%}\n"
        f"  → Caught {tp}/{tp + fn} malignant cases\n"
        f"Precision (PPV): {precision:.1%}\n"
        f"  → {tp}/{tp + fp} positive predictions correct"
    )

    fig.text(0.5, -0.05, interpretation, ha='center', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved confusion matrix to {save_path}")

    return fig


def plot_feature_importance(model, feature_names, top_n=15, save_path='artifacts/feature_importance.png'):
    """
    Plot top N most important features based on coefficient magnitude

    Args:
        model: Trained logistic regression model
        feature_names: List of feature names
        top_n: Number of top features to display
        save_path: Output file path

    Returns:
        Figure object
    """
    # Extract coefficients
    coefficients = model.named_steps['lr'].coef_[0]

    # Create DataFrame for sorting
    import pandas as pd
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'coefficient': coefficients,
        'abs_coefficient': np.abs(coefficients)
    }).sort_values('abs_coefficient', ascending=False).head(top_n)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Color by sign
    colors = ['red' if c < 0 else 'green' for c in feature_importance['coefficient']]

    # Horizontal bar plot
    y_pos = np.arange(len(feature_importance))
    ax.barh(y_pos, feature_importance['coefficient'], color=colors, alpha=0.7)

    # Styling
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feature_importance['feature'])
    ax.invert_yaxis()  # Top feature at top
    ax.set_xlabel('Coefficient Value', fontsize=12)
    ax.set_title(f'Top {top_n} Most Important Features\n(Logistic Regression Coefficients)',
                 fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    ax.grid(axis='x', alpha=0.3)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.7, label='Positive coefficient (increases malignancy risk)'),
        Patch(facecolor='red', alpha=0.7, label='Negative coefficient (decreases malignancy risk)')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved feature importance to {save_path}")

    return fig


def plot_cv_performance(cv_results, save_path='artifacts/cv_performance.png'):
    """
    Visualize cross-validation performance across different C values

    Args:
        cv_results: Dictionary with C values as keys and (train_mean, train_std, val_mean, val_std) as values
        save_path: Output file path

    """
    C_values = list(cv_results.keys())
    train_means = [cv_results[c][0] for c in C_values]
    train_stds = [cv_results[c][1] for c in C_values]
    val_means = [cv_results[c][2] for c in C_values]
    val_stds = [cv_results[c][3] for c in C_values]

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(C_values))
    width = 0.35

    # Plot bars with error bars
    ax.bar(x - width / 2, train_means, width, yerr=train_stds,
           label='Train ROC-AUC', alpha=0.7, capsize=5)
    ax.bar(x + width / 2, val_means, width, yerr=val_stds,
           label='Validation ROC-AUC', alpha=0.7, capsize=5)

    # Styling
    ax.set_xlabel('Regularization Parameter (C)', fontsize=12)
    ax.set_ylabel('ROC-AUC Score', fontsize=12)
    ax.set_title('Cross-Validation Performance Across Hyperparameters',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'C={c}' for c in C_values])
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0.98, 1.0])  # Zoom in to see differences

    # Highlight best C
    best_idx = np.argmax(val_means)
    ax.axvline(x=best_idx, color='red', linestyle='--', alpha=0.5, linewidth=2)
    ax.text(best_idx, 0.981, f'Best C={C_values[best_idx]}',
            ha='center', fontsize=10, color='red', fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved CV performance to {save_path}")

    # return figures
    return fig


def generate_all_visualizations(pipe, X_train, X_test, y_train, y_test, cv_results=None):
    """
    Generate complete set of visualizations for research documentation

    Args:
        pipe: Trained pipeline
        X_train, X_test, y_train, y_test: Data splits
        cv_results: Optional dictionary of CV results
    """
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)

    # Get predictions
    y_test_prob = pipe.predict_proba(X_test)[:, 1]
    y_test_pred = pipe.predict(X_test)

    plot_roc_curve(y_test, y_test_prob)
    plot_confusion_matrix(y_test, y_test_pred)
    plot_feature_importance(pipe, X_train.columns)

    if cv_results:
        plot_cv_performance(cv_results)

    print("\n✓ All visualizations saved to artifacts/")
    print("=" * 60)

