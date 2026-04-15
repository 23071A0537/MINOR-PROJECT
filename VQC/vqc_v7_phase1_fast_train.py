#!/usr/bin/env python3
"""
VQC v7 PHASE 1 FAST TRAINING (Using Precomputed Quantum Features)
Train neural head on precomputed 16-dim quantum features

This script is 100× faster than full circuit simulation because quantum
features are precomputed and cached.

Expected runtime: 60-90 minutes (vs 6-12 hours for full simulation)

Author: VQC v7 Phase 1 Implementation  
Date: 2026-04-01
"""

import os
import sys
import time
import json
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

# JAX imports
import jax
import jax.numpy as jnp
from jax import jit, value_and_grad

# ML imports
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Phase 1 training configuration"""
    
    # Paths
    PROJECT_ROOT = Path(__file__).parent.parent
    FEATURES_DIR = PROJECT_ROOT / "VQC" / "vqc_v7_quantum_features"
    OUTPUT_DIR = PROJECT_ROOT / "VQC" / "vqc_v7_phase1_trained"
    
    # Precomputed quantum features
    TRAIN_FEATURES = FEATURES_DIR / "quantum_features_train.npy"
    TRAIN_LABELS = FEATURES_DIR / "y_train.npy"
    TEST_FEATURES = FEATURES_DIR / "quantum_features_test.npy"
    TEST_LABELS = FEATURES_DIR / "y_test.npy"
    
    # Model parameters
    QNN_FEATURES = 16  # Precomputed quantum features dimension
    HEAD_DIMS = [128, 64]
    DROPOUT = 0.2
    
    # Training parameters
    NUM_CLASSES = 5
    CLASS_NAMES = ['NORMAL', 'DoSD', 'PROBE', 'EXPLOIT', 'MALWARE']
    
    # Phase 1 enhanced parameters
    CLASS_WEIGHTS = [1.0, 2.0, 2.0, 5.0, 30.0]  # MALWARE: 30×
    
    # Per-class focal gamma
    FOCAL_GAMMA = {
        0: 2.0,  # NORMAL
        1: 2.0,  # DoS
        2: 2.5,  # PROBE
        3: 3.0,  # EXPLOIT
        4: 4.0   # MALWARE (EXTREME focus)
    }
    
    # Training schedule
    BATCH_SIZE = 2000
    N_EPOCHS_PHASE1 = 150  # Head only
    N_EPOCHS_PHASE2 = 100  # Joint (quantum params frozen in precompute)
    LOG_EVERY = 5
    PHASE1_PATIENCE = 30
    PHASE2_PATIENCE = 20
    
    # Learning rates
    PHASE1_LR = 8e-3
    PHASE2_LR = 8e-4
    ADAM_LR_MIN = 1e-5
    ADAM_WARMUP = 5
    ADAM_B1 = 0.9
    ADAM_B2 = 0.999
    ADAM_EPS = 1e-8
    
    # Regularization
    L2_WEIGHT = 1e-5
    TEMPERATURE = 0.1
    
    # Validation split
    VAL_FRAC = 0.10
    
    # Random seed
    SEED = 42


# ============================================================================
# LOGGING
# ============================================================================

def setup_logging(output_dir):
    """Setup logging to both file and console"""
    os.makedirs(output_dir / "logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = output_dir / "logs" / f"training_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return log_file


# ============================================================================
# NEURAL HEAD (LayerNorm + Dropout)
# ============================================================================

def init_layer_norm(dim, seed):
    """Initialize LayerNorm parameters"""
    np.random.seed(seed)
    return {
        'gamma': np.ones(dim, dtype=np.float32),
        'beta': np.zeros(dim, dtype=np.float32)
    }


def apply_layer_norm(x, gamma, beta, eps=1e-5):
    """Apply LayerNorm"""
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    x_norm = (x - mean) / jnp.sqrt(var + eps)
    return gamma * x_norm + beta


def init_linear(in_dim, out_dim, seed):
    """Initialize linear layer with He initialization"""
    np.random.seed(seed)
    std = np.sqrt(2.0 / in_dim)
    W = np.random.randn(in_dim, out_dim).astype(np.float32) * std
    b = np.zeros(out_dim, dtype=np.float32)
    return W, b


def init_params(seed=Config.SEED):
    """Initialize neural head parameters (quantum features are precomputed)"""
    np.random.seed(seed)
    params = {}
    
    # Layer 0: QNN_FEATURES (16) -> HEAD_DIMS[0] (128)
    W0, b0 = init_linear(Config.QNN_FEATURES, Config.HEAD_DIMS[0], seed)
    params['W0'], params['b0'] = W0, b0
    ln0 = init_layer_norm(Config.HEAD_DIMS[0], seed)
    params['ln0_gamma'], params['ln0_beta'] = ln0['gamma'], ln0['beta']
    
    # Layer 1: HEAD_DIMS[0] (128) -> HEAD_DIMS[1] (64)
    W1, b1 = init_linear(Config.HEAD_DIMS[0], Config.HEAD_DIMS[1], seed + 1)
    params['W1'], params['b1'] = W1, b1
    ln1 = init_layer_norm(Config.HEAD_DIMS[1], seed + 1)
    params['ln1_gamma'], params['ln1_beta'] = ln1['gamma'], ln1['beta']
    
    # Output layer: HEAD_DIMS[1] (64) -> NUM_CLASSES (5)
    W2, b2 = init_linear(Config.HEAD_DIMS[1], Config.NUM_CLASSES, seed + 2)
    params['W2'], params['b2'] = W2, b2
    
    return params


def forward(params, qnn_features, training=False, dropout_key=None):
    """
    Forward pass through neural head
    qnn_features: (batch_size, 16) precomputed quantum features
    Returns: (batch_size, 5) logits
    """
    # Layer 0
    h0 = jnp.dot(qnn_features, params['W0']) + params['b0']
    h0 = apply_layer_norm(h0, params['ln0_gamma'], params['ln0_beta'])
    h0 = jax.nn.relu(h0)
    if training and dropout_key is not None:
        keep_prob = 1.0 - Config.DROPOUT
        mask = jax.random.bernoulli(dropout_key, keep_prob, h0.shape)
        h0 = jnp.where(mask, h0 / keep_prob, 0.0)
    
    # Layer 1
    h1 = jnp.dot(h0, params['W1']) + params['b1']
    h1 = apply_layer_norm(h1, params['ln1_gamma'], params['ln1_beta'])
    h1 = jax.nn.relu(h1)
    
    # Output logits
    logits = jnp.dot(h1, params['W2']) + params['b2']
    return logits


# ============================================================================
# PER-CLASS FOCAL LOSS
# ============================================================================

def per_class_focal_loss(params, qnn_features, y_oh, cw, gamma_dict, dropout_key=None, 
                         l2=Config.L2_WEIGHT, training=False):
    """
    Enhanced focal loss with per-class gamma values
    
    Key innovation: MALWARE gets gamma=4.0 (vs 2.5 in v6)
    """
    # Forward pass
    logits = forward(params, qnn_features, training=training, dropout_key=dropout_key)
    proba = jax.nn.softmax(logits, axis=1)
    proba_c = jnp.clip(proba, 1e-7, 1.0)
    
    # Compute pt (probability of true class)
    pt = jnp.sum(y_oh * proba_c, axis=1)  # (batch_size,)
    
    # Per-sample gamma based on true class
    class_indices = jnp.argmax(y_oh, axis=1)  # (batch_size,)
    gamma_array = jnp.array([gamma_dict[i] for i in range(Config.NUM_CLASSES)])
    gamma_per_sample = gamma_array[class_indices]  # (batch_size,)
    
    # Focal weight: (1 - pt)^gamma (per-sample gamma)
    focal_wt = (1.0 - pt) ** gamma_per_sample
    
    # Class weights (per-sample)
    sample_cw = jnp.sum(y_oh * cw[None, :], axis=1)
    
    # Cross-entropy
    ce = -jnp.sum(y_oh * jnp.log(proba_c), axis=1)
    
    # Weighted focal loss
    base_loss = jnp.mean(focal_wt * sample_cw * ce)
    
    # L2 regularization on weight matrices
    l2_loss = l2 * (jnp.sum(params['W0'] ** 2) + 
                    jnp.sum(params['W1'] ** 2) + 
                    jnp.sum(params['W2'] ** 2))
    
    return base_loss + l2_loss


# ============================================================================
# OPTIMIZATION
# ============================================================================

def cosine_lr(epoch, n_epochs, base_lr, min_lr=Config.ADAM_LR_MIN, warmup=Config.ADAM_WARMUP):
    """Cosine learning rate with warmup"""
    if epoch < warmup:
        return float(base_lr * (epoch + 1) / warmup)
    progress = (epoch - warmup) / max(1, n_epochs - warmup)
    return float(min_lr + 0.5 * (base_lr - min_lr) * (1 + np.cos(np.pi * progress)))


def adam_init(params):
    """Initialize Adam optimizer state"""
    return (jax.tree_util.tree_map(jnp.zeros_like, params),
            jax.tree_util.tree_map(jnp.zeros_like, params))


def adam_step(params, m, v, t, grads, lr, 
              b1=Config.ADAM_B1, b2=Config.ADAM_B2, eps=Config.ADAM_EPS):
    """Single Adam optimization step"""
    lr = jnp.float32(lr)
    t_n = t + jnp.float32(1.0)
    m_n = jax.tree_util.tree_map(lambda mi, gi: b1 * mi + (1 - b1) * gi, m, grads)
    v_n = jax.tree_util.tree_map(lambda vi, gi: b2 * vi + (1 - b2) * gi ** 2, v, grads)
    mh = jax.tree_util.tree_map(lambda mi: mi / (1 - b1 ** t_n), m_n)
    vh = jax.tree_util.tree_map(lambda vi: vi / (1 - b2 ** t_n), v_n)
    p_n = jax.tree_util.tree_map(lambda pi, mi, vi: pi - lr * mi / (jnp.sqrt(vi) + eps),
                                  params, mh, vh)
    return p_n, m_n, v_n, t_n


# ============================================================================
# PREDICTION
# ============================================================================

def predict_logits(params, qnn_features_np):
    """Predict logits for full dataset (inference mode)"""
    logits_list = []
    batch_size = Config.BATCH_SIZE
    n_samples = len(qnn_features_np)
    
    for i in range(0, n_samples, batch_size):
        X_batch = qnn_features_np[i:i + batch_size]
        X_jax = jnp.array(X_batch, dtype=jnp.float32)
        logits = forward(params, X_jax, training=False, dropout_key=None)
        logits_list.append(np.array(logits))
    
    return np.vstack(logits_list)


# ============================================================================
# DATA LOADING
# ============================================================================

def build_train_val(X_train_full, Y_train_full, val_frac=Config.VAL_FRAC, seed=Config.SEED):
    """Split training data into train/val with stratification"""
    np.random.seed(seed)
    n_total = len(Y_train_full)
    val_size = int(n_total * val_frac)
    
    # Stratified sampling
    train_idx, val_idx = [], []
    for c in range(Config.NUM_CLASSES):
        c_idx = np.where(Y_train_full == c)[0]
        np.random.shuffle(c_idx)
        n_val_c = int(len(c_idx) * val_frac)
        val_idx.extend(c_idx[:n_val_c])
        train_idx.extend(c_idx[n_val_c:])
    
    train_idx = np.array(train_idx)
    val_idx = np.array(val_idx)
    np.random.shuffle(train_idx)
    np.random.shuffle(val_idx)
    
    X_train = X_train_full[train_idx]
    Y_train = Y_train_full[train_idx]
    X_val = X_train_full[val_idx]
    Y_val = Y_train_full[val_idx]
    
    logging.info(f"Train/Val split: {len(X_train)} / {len(X_val)}")
    for c in range(Config.NUM_CLASSES):
        n_train_c = (Y_train == c).sum()
        n_val_c = (Y_val == c).sum()
        logging.info(f"  {Config.CLASS_NAMES[c]:<10}: train={n_train_c:5d}, val={n_val_c:4d}")
    
    return X_train, Y_train, X_val, Y_val


def load_data():
    """Load precomputed quantum features"""
    logging.info("="*70)
    logging.info("LOADING PRECOMPUTED QUANTUM FEATURES")
    logging.info("="*70)
    
    X_train_full = np.load(Config.TRAIN_FEATURES)
    Y_train_full = np.load(Config.TRAIN_LABELS)
    X_test = np.load(Config.TEST_FEATURES)
    Y_test = np.load(Config.TEST_LABELS)
    
    logging.info(f"Train features: {X_train_full.shape}")
    logging.info(f"Test features:  {X_test.shape}")
    logging.info(f"\nTraining class distribution:")
    for c in range(Config.NUM_CLASSES):
        count = (Y_train_full == c).sum()
        pct = 100 * count / len(Y_train_full)
        logging.info(f"  {Config.CLASS_NAMES[c]:<10}: {count:5d} ({pct:.1f}%)")
    
    # Create train/val split
    X_train, Y_train, X_val, Y_val = build_train_val(X_train_full, Y_train_full)
    
    return X_train, Y_train, X_val, Y_val, X_test, Y_test


# ============================================================================
# TRAINING
# ============================================================================

def train_model(tag, X_train_np, Y_train_np, X_val_np, Y_val_np, n_epochs, 
                base_lr, patience, seed=Config.SEED):
    """
    Single-phase training (quantum features are fixed)
    """
    np.random.seed(seed)
    dk_seq = jax.random.PRNGKey(seed)
    cw_jax = jnp.array(Config.CLASS_WEIGHTS, dtype=jnp.float32)
    gamma_dict = Config.FOCAL_GAMMA
    
    lb = LabelBinarizer().fit(np.arange(Config.NUM_CLASSES))
    Y_oh = lb.transform(Y_train_np).astype(np.float32)
    n_tr = len(X_train_np)
    n_bat = max(1, n_tr // Config.BATCH_SIZE)
    rng = np.random.RandomState(seed)
    params = init_params(seed)
    
    logging.info(f"\n{'='*70}")
    logging.info(f"  {tag} | seed={seed}")
    logging.info(f"  Per-class focal gamma: {gamma_dict}")
    logging.info(f"  Class weights: {Config.CLASS_WEIGHTS}")
    logging.info(f"  Training: {n_epochs} epochs, lr={base_lr:.2e}")
    logging.info(f"{'='*70}")
    
    # Training setup
    def loss_fn(params, X, y_oh, dropout_key):
        return per_class_focal_loss(params, X, y_oh, cw_jax, gamma_dict,
                                     dropout_key=dropout_key, training=True)
    
    grad_fn = jit(value_and_grad(loss_fn))
    m, v = adam_init(params)
    t = jnp.float32(0.0)
    
    best_vf1, best_params, no_imp = -1.0, params, 0
    t_start = time.time()
    
    for epoch in range(1, n_epochs + 1):
        lr_ep = cosine_lr(epoch - 1, n_epochs, base_lr)
        idx = rng.permutation(n_tr)
        Xs, Ys = X_train_np[idx], Y_oh[idx]
        ep_loss = 0.0
        
        for b in range(n_bat):
            s, e = b * Config.BATCH_SIZE, min((b + 1) * Config.BATCH_SIZE, n_tr)
            Xb = jnp.array(Xs[s:e], dtype=jnp.float32)
            Yb = jnp.array(Ys[s:e], dtype=jnp.float32)
            dk_seq, dk = jax.random.split(dk_seq)
            lv, grads = grad_fn(params, Xb, Yb, dk)
            params, m, v, t = adam_step(params, m, v, t, grads, lr_ep)
            ep_loss += float(lv)
        
        if epoch % Config.LOG_EVERY == 0 or epoch == n_epochs:
            val_log = predict_logits(params, X_val_np)
            vf1 = f1_score(Y_val_np, val_log.argmax(1), average='macro', zero_division=0)
            elapsed = time.time() - t_start
            logging.info(f'  EP {epoch:4d}/{n_epochs}  loss={ep_loss/n_bat:.4f}  '
                        f'val_F1={vf1:.4f}  lr={lr_ep:.2e}  {elapsed:.0f}s')
            if vf1 > best_vf1:
                best_vf1, best_params, no_imp = vf1, {k: np.array(v) for k, v in params.items()}, 0
            else:
                no_imp += Config.LOG_EVERY
                if no_imp >= patience:
                    logging.info(f'  Early stop at epoch {epoch}')
                    break
    
    logging.info(f'  Best val F1: {best_vf1:.4f}')
    logging.info(f'  Total training time: {(time.time() - t_start) / 60:.1f} minutes')
    
    return best_params, best_vf1


# ============================================================================
# EVALUATION & THRESHOLD OPTIMIZATION
# ============================================================================

def find_best_temperature(logits, y, T_range=None):
    """Find optimal temperature for calibration"""
    if T_range is None:
        T_range = np.concatenate([
            np.linspace(0.1, 1.0, 30),
            np.linspace(1.0, 5.0, 20),
        ])
    best_T, best_f1 = 1.0, -1.0
    for T in T_range:
        preds = np.argmax(logits / T, axis=1)
        f1 = f1_score(y, preds, average='macro', zero_division=0)
        if f1 > best_f1:
            best_f1, best_T = f1, T
    return best_T, best_f1


def find_per_class_thresholds(proba, y, n_classes=Config.NUM_CLASSES, thresh_range=None):
    """Find optimal per-class thresholds to maximize F1"""
    if thresh_range is None:
        thresh_range = np.linspace(0.05, 0.90, 35)
    
    thresholds = np.zeros(n_classes)
    for c in range(n_classes):
        best_t, best_f1c = 0.5, -1.0
        y_true_c = (y == c).astype(int)
        for t in thresh_range:
            y_pred_c = (proba[:, c] > t).astype(int)
            if y_pred_c.sum() == 0:
                continue
            tp = ((y_true_c == 1) & (y_pred_c == 1)).sum()
            fp = ((y_true_c == 0) & (y_pred_c == 1)).sum()
            fn = ((y_true_c == 1) & (y_pred_c == 0)).sum()
            prec = tp / (tp + fp + 1e-9)
            rec = tp / (tp + fn + 1e-9)
            f1c = 2 * prec * rec / (prec + rec + 1e-9)
            if f1c > best_f1c:
                best_f1c, best_t = f1c, t
        thresholds[c] = best_t
        logging.info(f'    {Config.CLASS_NAMES[c]:<10}: threshold={best_t:.2f}  F1={best_f1c:.4f}')
    return thresholds


def predict_with_thresholds(proba, thresholds):
    """Apply per-class thresholds for prediction"""
    margin = proba - thresholds[None, :]
    above = proba > thresholds[None, :]
    has_above = above.any(axis=1)
    masked_margin = np.where(above, margin, -np.inf)
    pred_thresh = masked_margin.argmax(axis=1)
    pred_fallback = proba.argmax(axis=1)
    return np.where(has_above, pred_thresh, pred_fallback)


def softmax_np(x):
    """Numpy softmax"""
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def evaluate_model(params, X_val, Y_val, X_test, Y_test, tag):
    """Complete evaluation pipeline"""
    logging.info(f"\n{'='*70}")
    logging.info(f"EVALUATING {tag}")
    logging.info(f"{'='*70}")
    
    # Validation predictions
    val_logits = predict_logits(params, X_val)
    
    # Temperature scaling
    logging.info("\nTemperature scaling...")
    best_T, _ = find_best_temperature(val_logits, Y_val)
    logging.info(f"  Best temperature: {best_T:.3f}")
    
    # Apply temperature
    val_logits_scaled = val_logits / best_T
    val_proba = softmax_np(val_logits_scaled)
    
    # Per-class threshold optimization
    logging.info("\nPer-class threshold optimization...")
    thresholds = find_per_class_thresholds(val_proba, Y_val)
    
    # Test evaluation
    logging.info("\nTest set evaluation...")
    test_logits = predict_logits(params, X_test)
    test_logits_scaled = test_logits / best_T
    test_proba = softmax_np(test_logits_scaled)
    
    # Predictions
    test_pred_argmax = test_proba.argmax(axis=1)
    test_pred_thresh = predict_with_thresholds(test_proba, thresholds)
    
    # Metrics
    f1_argmax = f1_score(Y_test, test_pred_argmax, average='macro', zero_division=0)
    f1_thresh = f1_score(Y_test, test_pred_thresh, average='macro', zero_division=0)
    
    logging.info(f"\nTest F1 (argmax):     {f1_argmax:.4f}")
    logging.info(f"Test F1 (threshold):  {f1_thresh:.4f}")
    logging.info(f"Improvement:          +{f1_thresh - f1_argmax:.4f}")
    
    # Detailed report
    logging.info("\nClassification Report (with thresholds):")
    report = classification_report(Y_test, test_pred_thresh, 
                                   target_names=Config.CLASS_NAMES,
                                   zero_division=0)
    logging.info(f"\n{report}")
    
    # Confusion matrix
    cm = confusion_matrix(Y_test, test_pred_thresh)
    logging.info("\nConfusion Matrix:")
    logging.info(f"\n{cm}")
    
    return {
        'temperature': float(best_T),
        'thresholds': thresholds.tolist(),
        'f1_argmax': float(f1_argmax),
        'f1_thresh': float(f1_thresh),
        'test_proba': test_proba,
        'confusion_matrix': cm.tolist(),
        'classification_report': classification_report(Y_test, test_pred_thresh,
                                                       target_names=Config.CLASS_NAMES,
                                                       output_dict=True,
                                                       zero_division=0)
    }


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Main training pipeline"""
    
    # Setup
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    log_file = setup_logging(Config.OUTPUT_DIR)
    
    logging.info("="*70)
    logging.info("VQC v7 PHASE 1 FAST TRAINING (Precomputed Features)")
    logging.info("="*70)
    logging.info(f"Output directory: {Config.OUTPUT_DIR}")
    logging.info(f"Log file: {log_file}")
    logging.info(f"\nConfiguration:")
    logging.info(f"  Class weights: {Config.CLASS_WEIGHTS}")
    logging.info(f"  Focal gamma: {Config.FOCAL_GAMMA}")
    logging.info(f"  Neural head: {Config.QNN_FEATURES} -> {Config.HEAD_DIMS} -> {Config.NUM_CLASSES}")
    logging.info(f"  Training: Phase1={Config.N_EPOCHS_PHASE1}ep, Phase2={Config.N_EPOCHS_PHASE2}ep")
    logging.info(f"  Batch size: {Config.BATCH_SIZE}")
    logging.info(f"  Device: {'GPU' if jax.devices()[0].platform == 'gpu' else 'CPU'}")
    
    # Check if quantum features exist
    if not Config.TRAIN_FEATURES.exists():
        logging.error(f"Quantum features not found at: {Config.TRAIN_FEATURES}")
        logging.error("Please run vqc_v7_precompute_features.py first!")
        return
    
    # Load data
    X_train, Y_train, X_val, Y_val, X_test, Y_test = load_data()
    
    # Train VQC-A
    logging.info("\n" + "="*70)
    logging.info("TRAINING VQC-A")
    logging.info("="*70)
    params_a, val_f1_a = train_model('VQC-A', X_train, Y_train, X_val, Y_val,
                                      Config.N_EPOCHS_PHASE1, Config.PHASE1_LR, 
                                      Config.PHASE1_PATIENCE, seed=Config.SEED)
    
    # Evaluate VQC-A
    results_a = evaluate_model(params_a, X_val, Y_val, X_test, Y_test, 'VQC-A')
    
    # Save VQC-A
    output_dir_a = Config.OUTPUT_DIR / "vqc_a"
    os.makedirs(output_dir_a, exist_ok=True)
    
    np.savez(output_dir_a / "model_params.npz", **params_a)
    np.save(output_dir_a / "test_proba.npy", results_a['test_proba'])
    
    meta_a = {
        'timestamp': datetime.now().isoformat(),
        'model': 'VQC-A',
        'phase': 'Phase 1 (Fast Training)',
        'seed': Config.SEED,
        'val_f1': float(val_f1_a),
        'test_f1_argmax': results_a['f1_argmax'],
        'test_f1_thresh': results_a['f1_thresh'],
        'temperature': results_a['temperature'],
        'thresholds': results_a['thresholds'],
        'class_weights': Config.CLASS_WEIGHTS,
        'focal_gamma': Config.FOCAL_GAMMA,
        'config': {
            'qnn_features': Config.QNN_FEATURES,
            'head_dims': Config.HEAD_DIMS,
            'epochs': Config.N_EPOCHS_PHASE1,
        }
    }
    
    with open(output_dir_a / "meta.json", 'w') as f:
        json.dump(meta_a, f, indent=2)
    
    logging.info(f"\nVQC-A saved to: {output_dir_a}")
    
    # Train VQC-B (different seed)
    logging.info("\n" + "="*70)
    logging.info("TRAINING VQC-B")
    logging.info("="*70)
    params_b, val_f1_b = train_model('VQC-B', X_train, Y_train, X_val, Y_val,
                                      Config.N_EPOCHS_PHASE1, Config.PHASE1_LR,
                                      Config.PHASE1_PATIENCE, seed=Config.SEED + 1)
    
    # Evaluate VQC-B
    results_b = evaluate_model(params_b, X_val, Y_val, X_test, Y_test, 'VQC-B')
    
    # Save VQC-B
    output_dir_b = Config.OUTPUT_DIR / "vqc_b"
    os.makedirs(output_dir_b, exist_ok=True)
    
    np.savez(output_dir_b / "model_params.npz", **params_b)
    np.save(output_dir_b / "test_proba.npy", results_b['test_proba'])
    
    meta_b = {
        'timestamp': datetime.now().isoformat(),
        'model': 'VQC-B',
        'phase': 'Phase 1 (Fast Training)',
        'seed': Config.SEED + 1,
        'val_f1': float(val_f1_b),
        'test_f1_argmax': results_b['f1_argmax'],
        'test_f1_thresh': results_b['f1_thresh'],
        'temperature': results_b['temperature'],
        'thresholds': results_b['thresholds'],
        'class_weights': Config.CLASS_WEIGHTS,
        'focal_gamma': Config.FOCAL_GAMMA,
        'config': {
            'qnn_features': Config.QNN_FEATURES,
            'head_dims': Config.HEAD_DIMS,
            'epochs': Config.N_EPOCHS_PHASE1,
        }
    }
    
    with open(output_dir_b / "meta.json", 'w') as f:
        json.dump(meta_b, f, indent=2)
    
    logging.info(f"\nVQC-B saved to: {output_dir_b}")
    
    # Ensemble evaluation
    logging.info("\n" + "="*70)
    logging.info("VQC ENSEMBLE (Simple Average)")
    logging.info("="*70)
    
    ensemble_proba = (results_a['test_proba'] + results_b['test_proba']) / 2
    ensemble_pred = ensemble_proba.argmax(axis=1)
    ensemble_f1 = f1_score(Y_test, ensemble_pred, average='macro', zero_division=0)
    
    logging.info(f"\nEnsemble F1: {ensemble_f1:.4f}")
    logging.info(f"VQC-A F1:    {results_a['f1_thresh']:.4f}")
    logging.info(f"VQC-B F1:    {results_b['f1_thresh']:.4f}")
    
    # Detailed ensemble report
    logging.info("\nEnsemble Classification Report:")
    report_ensemble = classification_report(Y_test, ensemble_pred, 
                                           target_names=Config.CLASS_NAMES,
                                           zero_division=0)
    logging.info(f"\n{report_ensemble}")
    
    # Save ensemble results
    ensemble_meta = {
        'timestamp': datetime.now().isoformat(),
        'model': 'VQC Ensemble (A+B average)',
        'phase': 'Phase 1 (Fast Training)',
        'ensemble_f1': float(ensemble_f1),
        'vqc_a_f1': results_a['f1_thresh'],
        'vqc_b_f1': results_b['f1_thresh'],
        'classification_report': classification_report(Y_test, ensemble_pred,
                                                       target_names=Config.CLASS_NAMES,
                                                       output_dict=True,
                                                       zero_division=0),
        'confusion_matrix': confusion_matrix(Y_test, ensemble_pred).tolist()
    }
    
    with open(Config.OUTPUT_DIR / "ensemble_results.json", 'w') as f:
        json.dump(ensemble_meta, f, indent=2)
    
    np.save(Config.OUTPUT_DIR / "ensemble_test_proba.npy", ensemble_proba)
    
    logging.info(f"\nEnsemble results saved to: {Config.OUTPUT_DIR / 'ensemble_results.json'}")
    
    # Final summary
    logging.info("\n" + "="*70)
    logging.info("PHASE 1 TRAINING COMPLETE!")
    logging.info("="*70)
    logging.info(f"\nResults Summary:")
    logging.info(f"  VQC-A F1:       {results_a['f1_thresh']:.4f}")
    logging.info(f"  VQC-B F1:       {results_b['f1_thresh']:.4f}")
    logging.info(f"  Ensemble F1:    {ensemble_f1:.4f}")
    logging.info(f"\nTarget: 0.77-0.78 F1 macro")
    if ensemble_f1 >= 0.77:
        logging.info("[SUCCESS] TARGET ACHIEVED!")
    else:
        logging.info(f"Gap to target: {0.77 - ensemble_f1:.4f}")
    
    logging.info(f"\nAll outputs saved to: {Config.OUTPUT_DIR}")
    logging.info("="*70)


if __name__ == "__main__":
    main()
