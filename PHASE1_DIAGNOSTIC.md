# Phase 1 Training - Diagnostic Report

## Current Status

**Corrected Phase 1 training running** (v2 - without ADASYN resampling)  
Training script: `VQC/vqc_v7_phase1_train_corrected.py`  
Expected completion: In progress

## Issue Found & Fixed

### Problem with Initial Approach

- **First attempt F1:** 0.6976 ❌ (worse than v6's 0.7254)
- **Root cause:** ADASYN resampling created severe distribution mismatch

### Distribution Mismatch Analysis

```
                  Training (ADASYN)  Test (Original)   Ratio
NORMAL            16.7%              79.0%             4.7x mismatch
DoS               16.7%              13.6%             1.2x
PROBE             16.7%               5.1%             3.3x mismatch
EXPLOIT           23.3%               2.1%            11.1x SEVERE mismatch
MALWARE           26.7%               0.2%           130x SEVERE mismatch!
```

**Result:** Model learned "everything is a minority class" because training data was balanced, but test data was highly imbalanced.

### MALWARE Prediction Issue

- **Recall:** 0.9204 (perfect! Finds all MALWARE)
- **Precision:** 0.0598 (terrible! 98% false positives)
- **Reason:** Model predicts MALWARE for everything because ADASYN training made MALWARE 26.7% of data

### Solution: No Resampling

New approach (`vqc_v7_phase1_train_corrected.py`):

1. **Use original distribution** (no ADASYN)
2. **Increase class weights MUCH higher:**
   - MALWARE: 100x (was 30x)
   - EXPLOIT: 60x (was 5x)
   - PROBE: 15x (was 2x)
   - DoS: 6x (was 2x)
   - NORMAL: 1x
3. **Increase focal loss gamma:**
   - MALWARE γ=5.0 (was 4.0)
   - EXPLOIT γ=4.0 (was 3.0)
4. **Adjust training parameters:**
   - Larger batch size (256 vs 64) - better for imbalanced data
   - Lower LR (0.0005 vs 0.001) - more stable
   - More patience (50 vs 30 epochs) - need time to learn
   - More epochs (200 vs 150)

## Expected Improvement (v2)

Based on weight/gamma fixes:

- **Test distribution preserved** ✓
- **MALWARE precision should improve dramatically** (fix false positives)
- **Macro F1 should reach 0.77-0.78** (target)
- **MALWARE F1:** 0.36-0.40 or better

## Key Learnings

1. **Never resample without maintaining test distribution**
2. **Synthetic oversampling (ADASYN) creates distribution shift**
3. **Class weights + focal loss better than resampling for imbalanced data**
4. **High-weight classes can lead to false positives if not calibrated**

## Files

- `vqc_v7_phase1_train_pytorch.py` - First attempt (has distribution mismatch issue)
- `vqc_v7_phase1_train_corrected.py` - Fixed version (running now)
- Results will be in: `vqc_v7_phase1_trained_v2/phase1_results_v2.json`

## Next Steps When Corrected Training Completes

1. Verify test F1 macro ≥ 0.77
2. Verify MALWARE F1 ≥ 0.36
3. Check precision/recall balance for MALWARE
4. If successful → Phase 2 (quantum circuit expansion)
5. If limited → Further fine-tune weights/gamma

---

**Expected completion time:** 30-60 minutes from start of corrected training
