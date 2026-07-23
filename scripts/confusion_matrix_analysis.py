"""
Confusion Matrix Analysis: XGBoost + Random Forest
==================================================
Detailed analysis of misclassification patterns.
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(r"c:\Users\[USERNAME]\Desktop\MINOR PROJECT")

with open(BASE_DIR / "hybrid layer output.json") as f:
    hybrid = json.load(f)

cm = np.array(hybrid['metrics']['confusion_matrix'])
class_names = ["NORMALL", "DoSD", "PROBE", "EXPLOIT", "MALWARE"]

print("="*80)
print("CONFUSION MATRIX ANALYSIS: XGBoost + Random Forest Ensemble")
print("="*80)

print("\nRaw Confusion Matrix:")
print("Actual\\Predicted    ", end="")
for cls in class_names:
    print(f"{cls:>10}", end=" ")
print()
print("-"*80)

for i, actual_cls in enumerate(class_names):
    print(f"{actual_cls:>18}", end="")
    for j in range(len(class_names)):
        print(f"{cm[i, j]:>10}", end=" ")
    print()

# Calculate per-class metrics from confusion matrix
print("\n" + "="*80)
print("PER-CLASS MISCLASSIFICATION ANALYSIS")
print("="*80)

total_samples = cm.sum()
print(f"Total test samples: {total_samples:,}")

for i, actual_cls in enumerate(class_names):
    total_class = cm[i, :].sum()
    correct = cm[i, i]
    incorrect = total_class - correct
    error_rate = incorrect / total_class * 100 if total_class > 0 else 0
    
    print(f"\n{actual_cls} (n={total_class:,}):")
    print(f"  Correctly classified:   {correct:>7,} ({correct/total_class*100:>5.2f}%)")
    print(f"  Misclassified:          {incorrect:>7,} ({error_rate:>5.2f}%)")
    
    if incorrect > 0:
        print(f"  Top misclassifications:")
        # Find top misclassifications (excluding correct predictions)
        misclass = [(cm[i, j], class_names[j]) for j in range(len(class_names)) if j != i]
        misclass.sort(reverse=True)
        for count, pred_cls in misclass[:3]:
            if count > 0:
                pct = count / total_class * 100
                print(f"    → {pred_cls:12s}: {count:>6,} ({pct:>5.2f}%)")

# Analyze specific confusion patterns
print("\n" + "="*80)
print("NOTABLE CONFUSION PATTERNS")
print("="*80)

patterns = [
    (0, 2, "NORMALL misclassified as PROBE"),
    (0, 3, "NORMALL misclassified as EXPLOIT"),
    (1, 3, "DoSD misclassified as EXPLOIT"),
    (2, 3, "PROBE misclassified as EXPLOIT"),
    (3, 0, "EXPLOIT misclassified as NORMALL"),
    (4, 3, "MALWARE misclassified as EXPLOIT"),
]

for i, j, description in patterns:
    count = cm[i, j]
    total = cm[i, :].sum()
    pct = count / total * 100 if total > 0 else 0
    print(f"\n{description}:")
    print(f"  Count: {count:,} ({pct:.2f}% of {class_names[i]})")
    
    if pct > 0.5:
        print(f"  ⚠ Significant pattern - may indicate:")
        if i == 0 and j == 3:
            print(f"    - Normal traffic triggering false exploit alarms")
        elif i == 1 and j == 3:
            print(f"    - DoS patterns being confused with exploits")
        elif i == 3 and j == 0:
            print(f"    - Some exploits look benign to the model")
        elif i == 4 and j == 3:
            print(f"    - Malware patterns overlapping with exploits")

# Detection rates
print("\n" + "="*80)
print("ATTACK DETECTION RATES (Recall)")
print("="*80)

for i in range(len(class_names)):
    total = cm[i, :].sum()
    detected = cm[i, i]
    detection_rate = detected / total * 100 if total > 0 else 0
    print(f"{class_names[i]:>10}: {detection_rate:>6.2f}% detected ({detected:,}/{total:,})")

# False positive analysis
print("\n" + "="*80)
print("FALSE POSITIVE ANALYSIS")
print("="*80)

for j in range(len(class_names)):
    total_predicted = cm[:, j].sum()
    correct = cm[j, j]
    false_positives = total_predicted - correct
    false_positive_rate = false_positives / total_predicted * 100 if total_predicted > 0 else 0
    
    print(f"\nPredicted as {class_names[j]} (n={total_predicted:,}):")
    print(f"  True positives:   {correct:>7,} ({correct/total_predicted*100:>5.2f}%)")
    print(f"  False positives:  {false_positives:>7,} ({false_positive_rate:>5.2f}%)")
    
    if false_positives > 0:
        print(f"  False positive sources:")
        for i in range(len(class_names)):
            if i != j and cm[i, j] > 0:
                fp_count = cm[i, j]
                pct = fp_count / total_predicted * 100
                print(f"    ← {class_names[i]:12s}: {fp_count:>6,} ({pct:>5.2f}%)")

# Critical security metrics
print("\n" + "="*80)
print("SECURITY-RELEVANT METRICS")
print("="*80)

# What percentage of actual attacks are detected?
normal_total = cm[0, :].sum()
normal_detected = cm[0, 0]
normal_detection_rate = normal_detected / normal_total * 100

all_attacks = cm[1:, :].sum()
attacks_detected = sum(cm[i, i] for i in range(1, len(class_names)))
attack_detection_rate = attacks_detected / all_attacks * 100 if all_attacks > 0 else 0

print(f"\nNormal Traffic Detection:  {normal_detection_rate:.2f}%")
print(f"  → {normal_detected:,}/{normal_total:,} benign flows correctly identified")

print(f"\nAttack Detection (Overall): {attack_detection_rate:.2f}%")
print(f"  → {attacks_detected:,}/{all_attacks:,} attacks correctly identified")
print(f"  → {all_attacks - attacks_detected:,} attacks MISSED")

print(f"\nMissed Attacks Breakdown:")
for i in range(1, len(class_names)):
    total = cm[i, :].sum()
    missed = total - cm[i, i]
    missed_pct = missed / total * 100 if total > 0 else 0
    print(f"  {class_names[i]:10s}: {missed:>6,} missed ({missed_pct:>5.2f}%)")

# False alarm rate (what % of predictions are wrong?)
total_predictions = cm.sum()
correct_predictions = sum(cm[i, i] for i in range(len(class_names)))
false_alarm_rate = (total_predictions - correct_predictions) / total_predictions * 100

print(f"\nFalse Alarm Rate (False Positives): {false_alarm_rate:.2f}%")
print(f"  → {total_predictions - correct_predictions:,}/{total_predictions:,} incorrect predictions")

print("\n✓ Analysis complete!")
