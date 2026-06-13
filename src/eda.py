"""
eda.py
------
Exploratory Data Analysis for the Pima Indians Diabetes Dataset.
Generates and saves all visualization plots to the outputs/ directory.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_data, get_basic_info, FEATURE_COLUMNS


def run_eda(data_path: str, output_dir: str = 'outputs') -> None:
    """
    Full EDA pipeline: load data, print statistics, and generate plots.

    Parameters:
        data_path (str): Path to diabetes.csv
        output_dir (str): Directory to save output plots
    """
    os.makedirs(output_dir, exist_ok=True)

    # ── Load ─────────────────────────────────────────────────────────────
    df = load_data(data_path)
    info = get_basic_info(df)

    print("\n" + "═"*50)
    print("  EXPLORATORY DATA ANALYSIS")
    print("═"*50)
    print(f"  Rows:              {info['n_rows']}")
    print(f"  Columns:           {info['n_cols']}")
    print(f"  Missing values:    {info['missing_values']}")
    print(f"  Duplicate rows:    {info['duplicate_rows']}")
    print(f"  Class dist (0/1):  {info['class_distribution']}")
    print(f"  Imbalance ratio:   {info['class_balance_ratio']}:1")
    print("\nDescriptive Statistics:")
    print(df.describe().round(2).to_string())

    # ── Zero value analysis (biological impossibilities) ──────────────────
    zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    print("\nZero-value counts (biologically impossible):")
    for col in zero_cols:
        n = (df[col] == 0).sum()
        pct = n / len(df) * 100
        print(f"  {col:<28}: {n:>3} ({pct:.1f}%)")

    # ── Plot 1: Class Distribution ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 5))
    counts = df['Outcome'].value_counts()
    bars = ax.bar(['No Diabetes (0)', 'Diabetes (1)'],
                  counts.values,
                  color=['#3B82F6', '#EF4444'],
                  edgecolor='white', linewidth=1.5)
    for bar, count in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                f'{count}\n({count/len(df)*100:.1f}%)',
                ha='center', va='bottom', fontsize=11)
    ax.set_title('Target Class Distribution', fontsize=14, fontweight='bold')
    ax.set_ylabel('Count')
    ax.set_ylim(0, counts.max() * 1.2)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'eda_01_class_distribution.png'), dpi=150)
    plt.close()

    # ── Plot 2: Feature Distributions by Class ────────────────────────────
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    colors = {0: '#3B82F6', 1: '#EF4444'}
    labels = {0: 'No Diabetes', 1: 'Diabetes'}

    for i, col in enumerate(FEATURE_COLUMNS):
        for outcome in [0, 1]:
            subset = df[df['Outcome'] == outcome][col]
            axes[i].hist(subset, bins=25, alpha=0.6,
                         color=colors[outcome], label=labels[outcome],
                         edgecolor='none', density=True)
        axes[i].set_title(col, fontsize=11, fontweight='bold')
        axes[i].set_xlabel('Value', fontsize=9)
        axes[i].set_ylabel('Density', fontsize=9)
        axes[i].legend(fontsize=8)
        axes[i].grid(alpha=0.3)

    plt.suptitle('Feature Distributions by Diabetes Outcome',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'eda_02_feature_distributions.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # ── Plot 3: Correlation Heatmap ───────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 7))
    corr = df.corr(numeric_only=True)
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    corr_lower = corr.where(~mask)

    im = ax.imshow(corr_lower.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(len(corr.columns)))
    ax.set_yticklabels(corr.columns, fontsize=9)

    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            val = corr_lower.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f'{val:.2f}',
                        ha='center', va='center', fontsize=7.5,
                        color='white' if abs(val) > 0.5 else 'black')

    ax.set_title('Feature Correlation Matrix', fontsize=13, fontweight='bold', pad=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'eda_03_correlation_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # ── Plot 4: Boxplots (Outlier Check) ──────────────────────────────────
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for i, col in enumerate(FEATURE_COLUMNS):
        data_by_class = [
            df[df['Outcome'] == 0][col].values,
            df[df['Outcome'] == 1][col].values
        ]
        bp = axes[i].boxplot(data_by_class,
                              patch_artist=True,
                              labels=['No Diabetes', 'Diabetes'])
        bp['boxes'][0].set_facecolor('#3B82F680')
        bp['boxes'][1].set_facecolor('#EF444480')
        axes[i].set_title(col, fontsize=11, fontweight='bold')
        axes[i].grid(axis='y', alpha=0.3)

    plt.suptitle('Feature Boxplots by Diabetes Outcome',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'eda_04_boxplots.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # ── Plot 5: Correlation with Target ──────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    target_corr = df.corr(numeric_only=True)['Outcome'].drop('Outcome').sort_values()
    colors_bar = ['#EF4444' if x > 0 else '#3B82F6' for x in target_corr.values]
    bars = ax.barh(target_corr.index, target_corr.values, color=colors_bar, edgecolor='white')
    for bar, val in zip(bars, target_corr.values):
        ax.text(val + (0.003 if val >= 0 else -0.003), bar.get_y() + bar.get_height() / 2,
                f'{val:.3f}', va='center', ha='left' if val >= 0 else 'right', fontsize=9)
    ax.axvline(x=0, color='black', linewidth=0.8)
    ax.set_title('Feature Correlation with Target (Outcome)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Pearson Correlation')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'eda_05_target_correlation.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n[EDA] All plots saved to '{output_dir}/'")
    print("  - eda_01_class_distribution.png")
    print("  - eda_02_feature_distributions.png")
    print("  - eda_03_correlation_heatmap.png")
    print("  - eda_04_boxplots.png")
    print("  - eda_05_target_correlation.png")


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    run_eda(
        data_path=os.path.join(BASE_DIR, 'data', 'diabetes.csv'),
        output_dir=os.path.join(BASE_DIR, 'outputs')
    )
