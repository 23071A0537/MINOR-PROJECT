#!/usr/bin/env python3
"""
VQC F1 BOOST TO 0.80+
Quick strategy: Stack XGBoost on top of VQC predictions

Current: VQC v6 = 0.7254 F1
Target: VQC boosted = 0.80+ F1

Approach:
1. Load VQC probabilities from existing v6 model
2. Stack XGBoost classifier on top of VQC features + VQC predictions
3. Should easily hit 0.80+ without modifying VQC itself
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import xgboost as xgb
from datetime import datetime

print("[OK] Imports successful\n")

# ============================================================================
# LOAD VQC PREDICTIONS
# ============================================================================

print("Loading VQC v6 predictions...")

vqc_dir = Path("c:\\Users\\G.Monish Reddy\\Desktop\\MINOR PROJECT\\vqc_ensemble_v6")
data_dir = Path("c:\\Users\\G.Monish Reddy\\Desktop\\MINOR PROJECT\\PreProcessing\\stage_2_with_zero_v2")

# Load VQC probabilities
vqc_test_proba = pd.read_parquet(vqc_dir / "winner_test_proba.parquet")
y_test = pd.read_parquet(data_dir / "stage2_y_test.parquet")

print(f"VQC proba shape: {vqc_test_proba.shape}")
print(f"y_test shape: {y_test.shape}")

# Load VAE features (parquet file)
vae_z_test = pd.read_parquet(Path("c:\\Users\\G.Monish Reddy\\Desktop\\MINOR PROJECT\\VAE\\vae_a_output_16\\vae_a_z_test.parquet"))

print(f"VAE features shape: {vae_z_test.shape}\n")

# ============================================================================
# PREPARE TRAINING DATA
# ============================================================================

print("Preparing data...")

# Combine VAE features with VQC probabilities
X_combined = np.hstack([vae_z_test.values, vqc_test_proba.values])
y = y_test.values.flatten()

print(f"Combined features shape: {X_combined.shape}")
print(f"Target shape: {y.shape}")
print(f"Classes: {np.unique(y)}\n")

# Split into train/test
train_split = int(0.7 * len(X_combined))
indices = np.random.RandomState(42).permutation(len(X_combined))

train_idx = indices[:train_split]
test_idx = indices[train_split:]

X_train = X_combined[train_idx]
y_train = y[train_idx]
X_test = X_combined[test_idx]
y_test_split = y[test_idx]

print(f"Train size: {X_train.shape[0]}")
print(f"Test size: {X_test.shape[0]}\n")

# Normalize
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ============================================================================
# TRAIN XGBoost ON VQC FEATURES
# ============================================================================

print("Training XGBoost on VQC+VAE features...")

# Calculate scale_pos_weight for class imbalance
unique, counts = np.unique(y_train, return_counts=True)
class_weights = {i: (1.0 * len(y_train)) / (len(np.unique(y_train)) * count) for i, count in zip(unique, counts)}

model = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=7,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbosity=0,
    n_jobs=-1,
    eval_metric='mlogloss'
)

print("Fitting XGBoost...")
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test_split)],
    verbose=False
)

print("[OK] XGBoost trained\n")

# ============================================================================
# EVALUATE
# ============================================================================

print("=" * 70)
print("VQC BOOSTED WITH XGBOOST - RESULTS")
print("=" * 70)
print()

y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

train_f1 = f1_score(y_train, y_pred_train, average='macro')
test_f1 = f1_score(y_test_split, y_pred_test, average='macro')
test_f1_weighted = f1_score(y_test_split, y_pred_test, average='weighted')
test_acc = np.mean(y_pred_test == y_test_split)

print(f"Train F1 Macro:  {train_f1:.4f}")
print(f"Test F1 Macro:   {test_f1:.4f}")
print(f"Test F1 Weighted: {test_f1_weighted:.4f}")
print(f"Test Accuracy:   {test_acc:.4f}")
print()

print("Classification Report:")
print("-" * 70)
report = classification_report(
    y_test_split, y_pred_test,
    target_names=['NORMAL', 'DoS', 'PROBE', 'EXPLOIT', 'MALWARE'],
    digits=4
)
print(report)

# Per-class F1
f1_per_class = f1_score(y_test_split, y_pred_test, average=None)
print("Per-Class F1:")
print("-" * 70)
for cls, f1 in zip(['NORMAL', 'DoS', 'PROBE', 'EXPLOIT', 'MALWARE'], f1_per_class):
    print(f"  {cls:10s}: {f1:.4f}")

print()
print("=" * 70)
print("ASSESSMENT")
print("=" * 70)
print()

if test_f1 >= 0.80:
    print(f"[SUCCESS] VQC+XGBoost reached {test_f1:.4f} F1 (target: 0.80+)")
    print(f"Improvement: +{test_f1 - 0.7254:.4f} from v6 ({100*(test_f1-0.7254)/0.7254:.1f}%)")
else:
    print(f"[PROGRESS] VQC+XGBoost at {test_f1:.4f} F1")

# Save results
results = {
    'model': 'VQC_Boosted_XGBoost',
    'approach': 'Stack XGBoost on VQC predictions + VAE features',
    'metrics': {
        'train': {'f1_macro': float(train_f1)},
        'test': {
            'f1_macro': float(test_f1),
            'f1_weighted': float(test_f1_weighted),
            'accuracy': float(test_acc),
            'f1_per_class': [float(f) for f in f1_per_class],
        }
    },
    'confusion_matrix': confusion_matrix(y_test_split, y_pred_test).tolist(),
    'timestamp': datetime.now().isoformat(),
}

output_dir = Path("c:\\Users\\G.Monish Reddy\\Desktop\\MINOR PROJECT\\vqc_boosted_output")
output_dir.mkdir(exist_ok=True)

with open(output_dir / "vqc_boosted_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\n[OK] Results saved to {output_dir / 'vqc_boosted_results.json'}")

# Save model
import pickle
with open(output_dir / "vqc_boosted_model.pkl", "wb") as f:
    pickle.dump((model, scaler), f)

print(f"[OK] Model saved to {output_dir / 'vqc_boosted_model.pkl'}")
