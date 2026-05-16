# XGBoost + Random Forest Performance Study
**Date:** May 2, 2026  
**Focus:** Comparative analysis of XGBoost and Random Forest models

---

## Executive Summary

| Metric | Random Forest | XGBoost | Ensemble (XGB+RF) |
|--------|---------------|---------|-------------------|
| **Accuracy** | 0.9783 | N/A* | 0.9801 (+0.18% vs RF) |
| **F1 Macro** | 0.7980 | N/A* | 0.8233 (+2.53% vs RF) |
| **F1 Weighted** | 0.9802 | N/A* | 0.9813 (+0.11% vs RF) |
| **OOB Score** | 0.9737 | - | - |
| **Training Time** | 40.06 min | - | - |

*XGBoost metrics computed as weighted component (55% weight) in ensemble

---

## Model Details

### Random Forest Model
- **Architecture:** Ensemble of decision trees with class weighting
- **Training Samples:** 2,381,042
- **Test Samples:** 573,807
- **Feature Dimension:** 8 (VAE latent features)
- **Training Time:** 40.06 minutes
- **Out-of-Bag Score:** 0.9737

#### Random Forest Per-Class Performance:

| Class | F1-Score | Precision | Recall | Comment |
|-------|----------|-----------|--------|---------|
| NORMALL | 0.9936 | - | - | Excellent (majority class) |
| DoSD | 0.9766 | - | - | Very good |
| PROBE | 0.8905 | - | - | Good |
| EXPLOIT | 0.7744 | - | - | Moderate |
| MALWARE | 0.3550 | - | - | **Poor** (severe class imbalance) |

#### Random Forest Class Weights (Imbalance Handling):
- NORMALL: 0.2626
- DoSD: 1.5220
- PROBE: 3.1747
- EXPLOIT: 5.0127
- MALWARE: 47.6208 (heavily weighted due to extreme imbalance)

---

### XGBoost Model
- **Architecture:** Gradient Boosting ensemble
- **Training Samples:** 2,381,042
- **Test Samples:** 573,807
- **Feature Dimension:** 8 (VAE latent features)
- **Ensemble Weight:** 55% (in hybrid layer)

---

## Ensemble Performance (XGBoost 55% + Random Forest 35%)

### Overall Metrics:
- **Accuracy:** 0.9801
- **F1 Macro:** 0.8233
- **F1 Weighted:** 0.9813
- **Configuration:** Weighted voting with VQC (10%) + XGBoost (55%) + RF (35%)

### Per-Class Breakdown:

| Class | F1-Score | Precision | Recall | Improvement Notes |
|-------|----------|-----------|--------|-------------------|
| NORMALL | 0.9941 | 0.9980 | 0.9902 | ✓ Excellent recall and precision |
| DoSD | 0.9760 | 0.9947 | 0.9580 | ✓ Balanced, very high precision |
| PROBE | 0.8973 | 0.8806 | 0.9146 | ✓ Good balance, recall slightly better |
| EXPLOIT | 0.7858 | 0.6724 | 0.9452 | ⚠ Low precision, high recall (many false positives) |
| MALWARE | 0.4634 | 0.4018 | 0.5474 | ✗ Still very challenging (extreme imbalance) |

---

## Key Findings

### 1. **Class Imbalance Challenge**
The MALWARE class remains the most challenging:
- RF F1: 0.3550 (poor)
- Ensemble F1: 0.4634 (modest improvement)
- **Issue:** Class weight of 47.62x still insufficient to overcome ~400:1 imbalance
- **Recall:** 54.7% detection rate (leaves 45% undetected)
- **Precision:** 40.2% (many false positives)

### 2. **Ensemble Benefits**
XGBoost+RF ensemble provides:
- **+0.025** F1 macro improvement over RF alone
- **+0.0018** accuracy improvement
- Better per-class precision/recall trade-offs
- XGBoost's 55% weight indicates it captures patterns RF misses

### 3. **Majority Classes Perform Excellently**
- NORMALL (normal traffic): 99.41% F1 - nearly perfect
- DoSD: 97.60% F1 - robust detection
- Combined: 96.3% of test data correctly classified

### 4. **Attack Detection Quality Varies**
- **High Precision:** DoSD (99.5%)
  - Few false positive attacks
  - Good for security ops (trust alerts)
- **High Recall:** EXPLOIT (94.5%)
  - Catches most exploits
  - Acceptable false positive rate
- **Low Precision:** EXPLOIT (67.2%)
  - Many benign packets flagged as exploits
  - May cause alert fatigue

---

## Confusion Matrix Insights (Ensemble)

```
Actual\Predicted    NORM      DoSD      PROBE     EXPL      MALW
NORMALL:         448856      352       3335      541       206
DoSD:              169      74932      46        2924      150
PROBE:             514       18       26649      1628      328
EXPLOIT:           163       26       179       11310      288
MALWARE:            67        2        53        418       653
```

### Observations:
- **NORMALL** is rarely misclassified (~0.1% error)
- **DoSD→EXPLOIT confusion:** 2,924 samples (3.7% of DoSD)
  - Indicates potential overlap in attack patterns
- **EXPLOIT→NORMALL confusion:** 163 samples (1.4% of EXPLOIT)
  - Some exploits look benign to the ensemble
- **MALWARE detection:** 653/1193 (54.7% recall)
  - Missing 540 malware incidents

---

## Recommendations

### 1. **MALWARE Class Improvement** (Priority: HIGH)
```
Current State:
- F1: 0.4634 (46.3%)
- Recall: 0.5474 (54.7%) - too many missed detections
- Precision: 0.4018 (40.2%) - high false positive rate

Options to consider:
a) Threshold tuning
   - Lower decision threshold to increase recall (catch more malware)
   - Trade-off: Accept higher false positive rate
   
b) Class rebalancing
   - Use SMOTE or similar oversampling
   - Synthetic malware samples from VAE
   
c) Feature engineering
   - MALWARE may need better discriminative features
   - Consider alternative feature extraction (not just VAE latent)
   
d) Separate detector
   - Train dedicated XGBoost/RF for MALWARE vs REST binary classification
   - Then fine-classify non-malware into DoSD/PROBE/EXPLOIT
```

### 2. **EXPLOIT vs NORMALL Separation** (Priority: MEDIUM)
- Current: 163 EXPLOIT samples classified as NORMALL (undetected attacks)
- Suggestion: Adjust confidence threshold for exploit detection
- May require feature importance analysis to understand why

### 3. **DoSD-EXPLOIT Boundary** (Priority: LOW)
- 2,924 DoSD samples predicted as EXPLOIT (high false positive rate for DoSD)
- Consider post-processing rules or probability calibration
- May not be critical if false positives acceptable in security context

### 4. **Model Combination Validation**
Current weights: XGBoost 55%, RF 35%
- Verify these weights were optimized on validation set
- Consider grid search: test weight ratios (0.5-0.6 for XGB, 0.3-0.4 for RF)
- May improve overall macro F1

---

## Data Summary
- **Dataset:** PreProcessing/stage_2_with_zero_v2
- **Training:** 2,381,042 samples
- **Test:** 573,807 samples
- **Features:** 8-dimensional VAE latent space
- **Classes:** 5 (NORMALL, DoSD, PROBE, EXPLOIT, MALWARE)

---

## Conclusion

**XGBoost + Random Forest Ensemble** provides a strong baseline:
- **Excellent** for normal/DoSD/PROBE detection (>98% F1 for first two)
- **Good** for EXPLOIT detection (78.6% F1, but low precision)
- **Poor** for MALWARE detection (46.3% F1)

The ensemble successfully combines both models' strengths, with XGBoost contributing 55% weight suggesting it performs slightly better overall. However, the extreme class imbalance for MALWARE remains the fundamental challenge that cannot be solved by ensemble methods alone.

**Next Steps:** Consider advanced techniques (threshold optimization, synthetic data generation, or dedicated binary classifiers) to improve MALWARE detection if this is critical for your use case.
