# data_analysis.py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def exploratory_data_analysis(df):
    """
    Perform EDA for research documentation
    """
    print("=" * 60)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    # Class distribution
    print("\n1. Class Distribution:")
    print(df['diagnosis'].value_counts())
    print(f"Malignant ratio: {df['diagnosis'].mean():.2%}")

    # Feature statistics
    print("\n2. Feature Summary Statistics:")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    summary = df[numeric_cols].describe()
    print(summary)

    # Correlation analysis
    print("\n3. Top 5 Features Correlated with Diagnosis:")
    correlations = df.corr()['diagnosis'].abs().sort_values(ascending=False)
    print(correlations.head(6))  # Top 5 + diagnosis itself

    # Missing values check
    print("\n4. Data Quality Check:")
    missing = df.isnull().sum()
    print(f"Missing values: {missing.sum()}")
    print(f"Duplicate rows: {df.duplicated().sum()}")

    # Save plots
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Class distribution
    df['diagnosis'].value_counts().plot(kind='bar', ax=axes[0])
    axes[0].set_title('Class Distribution')
    axes[0].set_xticklabels(['Benign', 'Malignant'])

    # Feature correlation heatmap (top features)
    top_features = correlations.head(11).index  # Top 10 + diagnosis
    sns.heatmap(df[top_features].corr(), annot=True, fmt='.2f',
                cmap='coolwarm', ax=axes[1])
    axes[1].set_title('Feature Correlation Matrix')

    plt.tight_layout()
    plt.savefig('eda_summary.png', dpi=300)
    print("\nSaved EDA visualizations")