import json
from pathlib import Path

BASE_DIR = Path(r"c:\Users\G.Monish Reddy\Desktop\MINOR PROJECT")

# Load Random Forest results
with open(BASE_DIR / "random_forest_output" / "rf_results.json") as f:
    rf_results = json.load(f)

print("RANDOM FOREST PERFORMANCE")
print("=" * 60)
print(f"Accuracy:          {rf_results['accuracy']:.6f}")
print(f"F1 Macro:          {rf_results['f1_macro']:.6f}")
print(f"F1 Weighted:       {rf_results['f1_weighted']:.6f}")
print(f"OOB Score:         {rf_results['oob_score']:.6f}")
print(f"Training Time:     {rf_results['training_time_minutes']:.2f} min")
print(f"\nPer-Class F1 Scores:")
for cls, f1 in rf_results['f1_per_class'].items():
    print(f"  {cls:10s}: {f1:.6f}")

# Load hybrid layer output (which has XGBoost and RF ensemble)
with open(BASE_DIR / "hybrid layer output.json") as f:
    hybrid = json.load(f)

print("\n" + "=" * 60)
print("HYBRID LAYER (XGBoost + RF Ensemble)")
print("=" * 60)
print(f"Weights: XGBoost={hybrid['weights']['xgboost']}, RF={hybrid['weights']['random_forest']}")
print(f"Accuracy:          {hybrid['metrics']['accuracy']:.6f}")
print(f"F1 Macro:          {hybrid['metrics']['f1_macro']:.6f}")
print(f"F1 Weighted:       {hybrid['metrics']['f1_weighted']:.6f}")
print(f"\nPer-Class Performance (from ensemble):")
for cls, metrics in hybrid['metrics']['classification_report'].items():
    if cls not in ['accuracy', 'macro avg', 'weighted avg']:
        f1 = metrics.get('f1-score')
        prec = metrics.get('precision')
        rec = metrics.get('recall')
        print(f"  {cls:10s}: F1={f1:.6f}, Precision={prec:.6f}, Recall={rec:.6f}")

print("\n" + "=" * 60)
print("PERFORMANCE GAINS: Ensemble vs RF Only")
print("=" * 60)
print(f"Accuracy Gain:     {hybrid['metrics']['accuracy'] - rf_results['accuracy']:+.6f}")
print(f"F1 Macro Gain:     {hybrid['metrics']['f1_macro'] - rf_results['f1_macro']:+.6f}")
print(f"F1 Weighted Gain:  {hybrid['metrics']['f1_weighted'] - rf_results['f1_weighted']:+.6f}")

print("\nDone!")
