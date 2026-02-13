"""
Exploratory Data Analysis module
Generates statistical summaries and data quality reports
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def exploratory_data_analysis(df, save_dir='artifacts'):
    """
    Perform comprehensive EDA on breast cancer dataset

    Args:
        df: Input DataFrame
        save_dir: Directory to save plots

    Returns:
        Dictionary with summary statistics
    """
    print("\n" + "=" * 60)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    # Ensure save directory exists
    Path(save_dir).mkdir(exist_ok=True)

    # 1. Dataset Overview
    print("\n1. DATASET OVERVIEW:")
    print(f"   Total samples: {len(df)}")
    print(f"   Total features: {len(df.columns)}")
    print(f"   Memory usage: {df.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB")

    # 2. Class Distribution
    print("\n2. CLASS DISTRIBUTION:")
    if 'diagnosis' in df.columns:
        class_counts = df['diagnosis'].value_counts()
        print(f"   Benign (0): {class_counts.get(0, 0)} ({class_counts.get(0, 0) / len(df) * 100:.1f}%)")
        print(f"   Malignant (1): {class_counts.get(1, 0)} ({class_counts.get(1, 0) / len(df) * 100:.1f}%)")

        # Visualize class distribution
        fig, ax = plt.subplots(figsize=(8, 5))
        class_counts.plot(kind='bar', ax=ax, color=['lightblue', 'salmon'])
        ax.set_xticklabels(['Benign', 'Malignant'], rotation=0)
        ax.set_ylabel('Count')
        ax.set_title('Class Distribution')
        ax.grid(axis='y', alpha=0.3)
        for i, v in enumerate(class_counts):
            ax.text(i, v + 5, str(v), ha='center', fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{save_dir}/class_distribution.png', dpi=300)
        plt.close()
        print(f"   ✓ Saved class distribution plot")

    # 3. Data Quality Check
    print("\n3. DATA QUALITY:")
    missing = df.isnull().sum().sum()
    duplicates = df.duplicated().sum()
    print(f"   Missing values: {missing}")
    print(f"   Duplicate rows: {duplicates}")

    # Check for infinite values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    inf_count = np.isinf(df[numeric_cols]).sum().sum()
    print(f"   Infinite values: {inf_count}")

    # 4. Feature Statistics
    print("\n4. FEATURE STATISTICS (sample):")
    summary = df[numeric_cols].describe()
    print(summary.iloc[:, :5].to_string())  # Show first 5 features

    # 5. Feature Correlations with Target
    if 'diagnosis' in df.columns:
        print("\n5. TOP 10 FEATURES CORRELATED WITH DIAGNOSIS:")
        correlations = df[numeric_cols].corrwith(df['diagnosis']).abs().sort_values(ascending=False)
        print(correlations.head(10).to_string())

        # Visualize top correlations
        fig, ax = plt.subplots(figsize=(10, 6))
        top_corr = correlations.head(15)
        top_corr.plot(kind='barh', ax=ax, color='steelblue')
        ax.set_xlabel('Absolute Correlation with Diagnosis')
        ax.set_title('Top 15 Features by Correlation with Diagnosis')
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{save_dir}/feature_correlations.png', dpi=300)
        plt.close()
        print(f"   ✓ Saved correlation plot")

    # 6. Feature Distributions (sample)
    print("\n6. GENERATING DISTRIBUTION PLOTS...")
    if 'diagnosis' in df.columns:
        # Plot distributions for top 6 correlated features
        top_features = correlations.head(6).index
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()

        for idx, feature in enumerate(top_features):
            for diagnosis_val in [0, 1]:
                data = df[df['diagnosis'] == diagnosis_val][feature]
                label = 'Benign' if diagnosis_val == 0 else 'Malignant'
                axes[idx].hist(data, alpha=0.6, label=label, bins=30)

            axes[idx].set_xlabel(feature)
            axes[idx].set_ylabel('Frequency')
            axes[idx].set_title(f'Distribution: {feature}')
            axes[idx].legend()
            axes[idx].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{save_dir}/feature_distributions.png', dpi=300)
        plt.close()
        print(f"   ✓ Saved distribution plots")

    print("\n" + "=" * 60)
    print("EDA COMPLETE - All plots saved to artifacts/")
    print("=" * 60)

    # Return summary dictionary
    return {
        'total_samples': len(df),
        'total_features': len(numeric_cols),
        'class_balance': class_counts.to_dict() if 'diagnosis' in df.columns else None,
        'missing_values': missing,
        'duplicate_rows': duplicates,
        'top_correlations': correlations.head(10).to_dict() if 'diagnosis' in df.columns else None
    }