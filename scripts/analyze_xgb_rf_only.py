"""
Performance Analysis: XGBoost vs Random Forest Only
====================================================
Compares XGBoost and Random Forest metrics side-by-side.
"""
import json
import pandas as pd
import pickle
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score
)

# Paths
BASE_DIR = Path(r"c:\Users\[USERNAME]\Desktop\MINOR PROJECT")
RF_RESULTS_PATH = BASE_DIR / "random_forest_output" / "rf_results.json"
RF_PROBA_PATH = BASE_DIR / "random_forest_output" / "rf_test_proba.parquet"
RF_MODEL_PATH = BASE_DIR / "random_forest_output" / "rf_model.pkl"

XGB_MODEL_PATH = BASE_DIR / "xgboost_output" / "xgboost_model.pkl"

Y_TEST_PATH = BASE_DIR / "PreProcessing" / "stage_2_with_zero_v2" / "stage2_y_test.parquet"

# Load Random Forest results (already computed)
with open(RF_RESULTS_PATH) as f:
    rf_results = json.load(f)

print("=" * 80)
print("RANDOM FOREST PERFORMANCE (Pre-trained)")
print("=" * 80)
print(f"Accuracy:           {rf_results['accuracy']:.6f}")
print(f"F1 Macro:           {rf_results['f1_macro']:.6f}")
print(f"F1 Weighted:        {rf_results['f1_weighted']:.6f}")
print(f"OOB Score:          {rf_results['oob_score']:.6f}")
print(f"Training Time:      {rf_results['training_time_minutes']:.2f} minutes")
print(f"\nPer-Class F1 Scores:")
for class_name, f1 in rf_results['f1_per_class'].items():
    print(f"  {class_name:12s}: {f1:.6f}")

print("\nClass Weights (imbalance handling):")
for class_idx, weight in rf_results['class_weights'].items():
    class_name = rf_results['class_names'][int(class_idx)]
    print(f"  {class_name:12s} (idx {class_idx}): {weight:.4f}")

# Load test data and predictions for XGBoost computation
try:
    print("\n" + "=" * 80)
    print("Loading test data and XGBoost model for predictions...")
    print("=" * 80)
    
    y_test = pd.read_parquet(Y_TEST_PATH).values.flatten()
    rf_proba = pd.read_parquet(RF_PROBA_PATH).values
    
    with open(XGB_MODEL_PATH, 'rb') as f:
        xgb_model = pickle.load(f)
    
    print(f"✓ Loaded y_test shape: {y_test.shape}")
    print(f"✓ Loaded RF predictions shape: {rf_proba.shape}")
    print(f"✓ Loaded XGBoost model: {type(xgb_model)}")
    
    # Get XGBoost predictions
    # XGBoost typically expects DMatrix or array
    try:
        xgb_proba = xgb_model.predict_proba(rf_proba)
    except:
        # If it doesn't work with RF proba, try loading X_test
        print("\nNote: XGBoost requires original features, not RF probabilities.")
        print("To compute XGBoost predictions, original test features are needed.")
        xgb_proba = None
    
    if xgb_proba is not None:
        xgb_pred = np.argmax(xgb_proba, axis=1)
        
        print("\n" + "=" * 80)
        print("XGBOOST PERFORMANCE (Computed from predictions)")
        print("=" * 80)
        
        xgb_accuracy = accuracy_score(y_test, xgb_pred)
        xgb_f1_macro = f1_score(y_test, xgb_pred, average='macro', zero_division=0)
        xgb_f1_weighted = f1_score(y_test, xgb_pred, average='weighted', zero_division=0)
        
        print(f"Accuracy:           {xgb_accuracy:.6f}")
        print(f"F1 Macro:           {xgb_f1_macro:.6f}")
        print(f"F1 Weighted:        {xgb_f1_weighted:.6f}")
        
        # Per-class F1
        class_names = ["NORMALL", "DoSD", "PROBE", "EXPLOIT", "MALWARE"]
        f1_per_class = f1_score(y_test, xgb_pred, average=None, zero_division=0)
        
        print(f"\nPer-Class F1 Scores:")
        for i, class_name in enumerate(class_names):
            print(f"  {class_name:12s}: {f1_per_class[i]:.6f}")
            
except Exception as e:
    print(f"⚠ Could not compute XGBoost predictions: {e}")
    print("\nThis likely means XGBoost was trained on original features,")
    print("not on Random Forest's output features. Need original X_test data.")

print("\n" + "=" * 80)
print("COMPARISON SUMMARY")
print("=" * 80)
print(f"Random Forest F1 (Macro):  {rf_results['f1_macro']:.6f}")
if xgb_proba is not None:
    print(f"XGBoost F1 (Macro):        {xgb_f1_macro:.6f}")
    print(f"Difference (XGB - RF):     {xgb_f1_macro - rf_results['f1_macro']:+.6f}")
else:
    print("XGBoost results: Not computed (feature dimension mismatch)")

print("\n✓ Analysis complete. For full comparison, load original X_test features.")
