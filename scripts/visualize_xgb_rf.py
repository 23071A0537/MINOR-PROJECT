"""
Visualization: XGBoost vs Random Forest Performance
====================================================
Creates comparison charts and detailed metrics visualization.
"""
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

BASE_DIR = Path(r"c:\Users\G.Monish Reddy\Desktop\MINOR PROJECT")

# Load data
with open(BASE_DIR / "random_forest_output" / "rf_results.json") as f:
    rf_results = json.load(f)

with open(BASE_DIR / "hybrid layer output.json") as f:
    hybrid = json.load(f)

class_names = ["NORMALL", "DoSD", "PROBE", "EXPLOIT", "MALWARE"]

# Extract metrics
rf_f1_per_class = [rf_results['f1_per_class'][cls] for cls in class_names]

# Extract ensemble metrics
ensemble_report = hybrid['metrics']['classification_report']
ensemble_f1_per_class = [ensemble_report[cls]['f1-score'] for cls in class_names]
ensemble_precision = [ensemble_report[cls]['precision'] for cls in class_names]
ensemble_recall = [ensemble_report[cls]['recall'] for cls in class_names]

# Create comparison visualizations
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('XGBoost + Random Forest Performance Analysis', fontsize=16, fontweight='bold')

# 1. F1 Score Comparison
ax = axes[0, 0]
x = np.arange(len(class_names))
width = 0.35
ax.bar(x - width/2, rf_f1_per_class, width, label='Random Forest', alpha=0.8, color='steelblue')
ax.bar(x + width/2, ensemble_f1_per_class, width, label='Ensemble (XGB+RF)', alpha=0.8, color='coral')
ax.set_ylabel('F1 Score', fontweight='bold')
ax.set_title('Per-Class F1 Score Comparison')
ax.set_xticks(x)
ax.set_xticklabels(class_names, rotation=45)
ax.legend()
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0, 1.0)

# Add value labels on bars
for i, (rf_f1, ens_f1) in enumerate(zip(rf_f1_per_class, ensemble_f1_per_class)):
    ax.text(i - width/2, rf_f1 + 0.02, f'{rf_f1:.3f}', ha='center', fontsize=9)
    ax.text(i + width/2, ens_f1 + 0.02, f'{ens_f1:.3f}', ha='center', fontsize=9)

# 2. Precision-Recall Trade-off (Ensemble)
ax = axes[0, 1]
x = np.arange(len(class_names))
width = 0.35
ax.bar(x - width/2, ensemble_precision, width, label='Precision', alpha=0.8, color='green')
ax.bar(x + width/2, ensemble_recall, width, label='Recall', alpha=0.8, color='purple')
ax.set_ylabel('Score', fontweight='bold')
ax.set_title('Ensemble: Precision vs Recall per Class')
ax.set_xticks(x)
ax.set_xticklabels(class_names, rotation=45)
ax.legend()
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0, 1.0)

# Add value labels
for i, (prec, rec) in enumerate(zip(ensemble_precision, ensemble_recall)):
    ax.text(i - width/2, prec + 0.02, f'{prec:.3f}', ha='center', fontsize=9)
    ax.text(i + width/2, rec + 0.02, f'{rec:.3f}', ha='center', fontsize=9)

# 3. Overall Metrics Comparison
ax = axes[1, 0]
metrics = ['Accuracy', 'F1 Macro', 'F1 Weighted']
rf_scores = [rf_results['accuracy'], rf_results['f1_macro'], rf_results['f1_weighted']]
ensemble_scores = [
    hybrid['metrics']['accuracy'],
    hybrid['metrics']['f1_macro'],
    hybrid['metrics']['f1_weighted']
]

x = np.arange(len(metrics))
width = 0.35
ax.bar(x - width/2, rf_scores, width, label='Random Forest', alpha=0.8, color='steelblue')
ax.bar(x + width/2, ensemble_scores, width, label='Ensemble (XGB+RF)', alpha=0.8, color='coral')
ax.set_ylabel('Score', fontweight='bold')
ax.set_title('Overall Performance Metrics')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0.97, 0.985)

# Add value labels
for i, (rf_s, ens_s) in enumerate(zip(rf_scores, ensemble_scores)):
    ax.text(i - width/2, rf_s + 0.0002, f'{rf_s:.4f}', ha='center', fontsize=9)
    ax.text(i + width/2, ens_s + 0.0002, f'{ens_s:.4f}', ha='center', fontsize=9)

# 4. F1 Improvement (Ensemble - RF)
ax = axes[1, 1]
improvements = [ens_f1 - rf_f1 for ens_f1, rf_f1 in zip(ensemble_f1_per_class, rf_f1_per_class)]
colors = ['green' if imp >= 0 else 'red' for imp in improvements]
ax.barh(class_names, improvements, color=colors, alpha=0.8)
ax.set_xlabel('F1 Score Improvement', fontweight='bold')
ax.set_title('Ensemble vs Random Forest F1 Gain')
ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
ax.grid(axis='x', alpha=0.3)

# Add value labels
for i, imp in enumerate(improvements):
    ax.text(imp + 0.005 if imp >= 0 else imp - 0.005, i, f'{imp:+.4f}', 
            va='center', ha='left' if imp >= 0 else 'right', fontsize=9)

plt.tight_layout()
plt.savefig(BASE_DIR / 'artifacts' / 'plots' / 'xgb_rf_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved comparison chart: artifacts/plots/xgb_rf_comparison.png")

# Create detailed metrics table
print("\n" + "="*80)
print("DETAILED COMPARISON TABLE")
print("="*80)

df = pd.DataFrame({
    'Class': class_names,
    'RF_F1': rf_f1_per_class,
    'Ensemble_F1': ensemble_f1_per_class,
    'F1_Gain': improvements,
    'Ensemble_Precision': ensemble_precision,
    'Ensemble_Recall': ensemble_recall,
})

print(df.to_string(index=False, float_format=lambda x: f'{x:.6f}'))

print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)
print(f"Random Forest:")
print(f"  Accuracy:     {rf_results['accuracy']:.6f}")
print(f"  F1 Macro:     {rf_results['f1_macro']:.6f}")
print(f"  F1 Weighted:  {rf_results['f1_weighted']:.6f}")
print(f"  OOB Score:    {rf_results['oob_score']:.6f}")

print(f"\nEnsemble (XGB 55% + RF 35% + VQC 10%):")
print(f"  Accuracy:     {hybrid['metrics']['accuracy']:.6f}")
print(f"  F1 Macro:     {hybrid['metrics']['f1_macro']:.6f}")
print(f"  F1 Weighted:  {hybrid['metrics']['f1_weighted']:.6f}")

print(f"\nImprovement (Ensemble - RF):")
print(f"  Accuracy:     {hybrid['metrics']['accuracy'] - rf_results['accuracy']:+.6f} ({(hybrid['metrics']['accuracy'] - rf_results['accuracy'])/rf_results['accuracy']*100:+.2f}%)")
print(f"  F1 Macro:     {hybrid['metrics']['f1_macro'] - rf_results['f1_macro']:+.6f} ({(hybrid['metrics']['f1_macro'] - rf_results['f1_macro'])/rf_results['f1_macro']*100:+.2f}%)")
print(f"  F1 Weighted:  {hybrid['metrics']['f1_weighted'] - rf_results['f1_weighted']:+.6f} ({(hybrid['metrics']['f1_weighted'] - rf_results['f1_weighted'])/rf_results['f1_weighted']*100:+.2f}%)")

print("\n✓ Analysis complete!")
