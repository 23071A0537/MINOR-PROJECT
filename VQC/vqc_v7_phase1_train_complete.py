#!/usr/bin/env python3
"""
VQC v7 PHASE 1 TRAINING - Complete Implementation
Quantum Machine Learning with PennyLane + JAX

Architecture & Training identical to v6, but with Phase 1 enhancements:
✓ ADASYN-enhanced training data (MALWARE 8K, EXPLOIT 7K)
✓ Enhanced class weights (MALWARE 15x → 30x)
✓ Per-class focal loss (MALWARE gamma 2.5 → 4.0)

Expected Performance:
  MALWARE F1:  0.20 → 0.36-0.40 (+82%)
  EXPLOIT F1:  0.72 → 0.76-0.78 (+6%)
  Macro F1:    0.73 → 0.77-0.78 (+5.7%)

Training Time: ~2-3 hours
Device: CPU or GPU (auto-detected)

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

# Core imports
import pennylane as qml
from jax import numpy as jnp, jit, value_and_grad, vmap
import jax
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import f1_score, classification_report, confusion_matrix

# PennyLane 0.44 may call jax.core.is_concrete, which is absent in newer JAX releases.
if not hasattr(jax.core, "is_concrete"):
    def _is_concrete(x):
        return not isinstance(x, jax.core.Tracer)
    jax.core.is_concrete = _is_concrete

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Phase 1 training configuration"""
    
    # Paths
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / "VQC" / "vqc_v7_phase1_output"
    OUTPUT_DIR = PROJECT_ROOT / "VQC" / "vqc_v7_phase1_trained"
    
    # Phase 1 input data (prepared by phase1_data_preparation.py)
    VAE_TRAIN = DATA_DIR / "vae_z_train_sampled.npy"
    Y_TRAIN = DATA_DIR / "y_train_sampled.npy"
    VAE_TEST = DATA_DIR / "vae_z_test.npy"
    Y_TEST = DATA_DIR / "y_test.npy"
    
    # Quantum circuit parameters (matching v6)
    NUM_QUBITS = 8
    RA_REPS = 2  # RealAmplitudes repetitions
    ZZ_REPS = 2  # ZZFeatureMap repetitions
    
    # Classification
    NUM_CLASSES = 5
    CLASS_NAMES = ['NORMAL', 'DoSD', 'PROBE', 'EXPLOIT', 'MALWARE']
    
    # Phase 1 ENHANCED PARAMETERS
    CLASS_WEIGHTS_MANUAL = [1.0, 2.0, 2.0, 5.0, 30.0]  # MALWARE doubled to 30x
    
    # Per-class focal gamma (MALWARE gets extreme focus)
    FOCAL_GAMMA = {
        0: 2.0,  # NORMAL
        1: 2.0,  # DoS
        2: 2.5,  # PROBE
        3: 3.0,  # EXPLOIT
        4: 4.0   # MALWARE (EXTREME: 4.0 vs 2.5 in v6)
    }
    
    # Training parameters
    BATCH_SIZE = 2000
    PHASE1_EPOCHS = 150
    PHASE2_EPOCHS = 100
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
    HEAD_DIMS = [128, 64]
    DROPOUT = 0.2
    
    # Validation
    VAL_FRAC = 0.10

    # Precision guardrails for threshold search
    PROBE_PRECISION_FLOOR = 0.80
    MALWARE_PRECISION_FLOOR = 0.70
    
    # Device
    DEVICE = "default.qubit"  # PennyLane device
    DIFF_METHOD = "backprop"   # Differentiation method for current PennyLane + JAX stack
    SEED = 42


# ============================================================================
# LOGGING
# ============================================================================

def setup_logging(output_dir):
    """Setup logging to file and console"""
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
# QUANTUM CIRCUIT (from v6)
# ============================================================================

def build_quantum_circuit():
    """
    Build VQC circuit identical to v6
    Features: 8 (VAE latent codes)
    Parameters: 24 (quantum variational)
    Output: 16 (8 PauliZ + 8 PauliX)
    """
    dev = qml.device(Config.DEVICE, wires=Config.NUM_QUBITS)
    
    @qml.qnode(dev, interface='jax', diff_method=Config.DIFF_METHOD)
    def vqc_circuit(x, weights):
        """
        Quantum circuit: angle encoding + ZZFeatureMap + RealAmplitudes
        Returns: 16 observables (PauliZ + PauliX for each qubit)
        """
        # Angle encoding: RY gates with input data
        for i in range(Config.NUM_QUBITS):
            qml.RY(x[i], wires=i)
        
        # ZZFeatureMap: pairwise entanglement
        for _ in range(Config.ZZ_REPS):
            for i in range(Config.NUM_QUBITS):
                for j in range(i + 1, Config.NUM_QUBITS):
                    qml.IsingZZ(x[i] * x[j], wires=[i, j])
        
        # RealAmplitudes: variational ansatz
        for rep in range(Config.RA_REPS):
            # RY rotations
            for i in range(Config.NUM_QUBITS):
                qml.RY(weights[rep * Config.NUM_QUBITS * 2 + i], wires=i)
            # CNOT ladder
            for i in range(Config.NUM_QUBITS - 1):
                qml.CNOT(wires=[i, i + 1])
            # RY rotations again
            for i in range(Config.NUM_QUBITS):
                qml.RY(weights[rep * Config.NUM_QUBITS * 2 + Config.NUM_QUBITS + i], wires=i)
        
        # Measure: PauliZ + PauliX for each qubit
        return (tuple(qml.expval(qml.PauliZ(i)) for i in range(Config.NUM_QUBITS)) +
                tuple(qml.expval(qml.PauliX(i)) for i in range(Config.NUM_QUBITS)))
    
    return vqc_circuit, dev


# ============================================================================
# NEURAL HEAD
# ============================================================================

def init_layer_norm(dim, seed):
    """Initialize LayerNorm"""
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
    """Initialize linear layer (He)"""
    np.random.seed(seed)
    std = np.sqrt(2.0 / in_dim)
    W = np.random.randn(in_dim, out_dim).astype(np.float32) * std
    b = np.zeros(out_dim, dtype=np.float32)
    return W, b


def init_params(seed=Config.SEED):
    """Initialize all parameters: quantum + neural head"""
    np.random.seed(seed)
    params = {}
    
    # Quantum parameters: 24 total (2 reps × 8 qubits × 2 layers + adjustments)
    q_params = np.random.randn(Config.RA_REPS * Config.NUM_QUBITS * 2).astype(np.float32) * 0.1
    params['q'] = q_params
    
    # Neural head: 16 → 128 → 64 → 5
    W0, b0 = init_linear(16, Config.HEAD_DIMS[0], seed)
    params['W0'], params['b0'] = W0, b0
    ln0 = init_layer_norm(Config.HEAD_DIMS[0], seed)
    params['ln0_gamma'], params['ln0_beta'] = ln0['gamma'], ln0['beta']
    
    W1, b1 = init_linear(Config.HEAD_DIMS[0], Config.HEAD_DIMS[1], seed + 1)
    params['W1'], params['b1'] = W1, b1
    ln1 = init_layer_norm(Config.HEAD_DIMS[1], seed + 1)
    params['ln1_gamma'], params['ln1_beta'] = ln1['gamma'], ln1['beta']
    
    W2, b2 = init_linear(Config.HEAD_DIMS[1], Config.NUM_CLASSES, seed + 2)
    params['W2'], params['b2'] = W2, b2
    
    return params


def full_forward(params, X_batch, vqc_circuit, training=False, dropout_key=None):
    """
    Full forward: quantum circuit + neural head
    X_batch: (batch_size, 8) VAE features
    Returns: (batch_size, 5) logits
    """
    # Quantum layer: vmap over batch
    q_features_fn = vmap(lambda x: jnp.array(vqc_circuit(x, params['q'])))
    qnn_out = q_features_fn(X_batch)  # (batch_size, 16)
    
    # Neural head layer 0
    h0 = jnp.dot(qnn_out, params['W0']) + params['b0']
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
    
    # Output
    logits = jnp.dot(h1, params['W2']) + params['b2']
    return logits


# ============================================================================
# PER-CLASS FOCAL LOSS (v7 ENHANCEMENT)
# ============================================================================

def per_class_focal_loss(params, X_batch, Y_oh, vqc_circuit, cw_jax, gamma_dict,
                         dropout_key=None, l2=Config.L2_WEIGHT, training=False):
    """
    Per-class focal loss with individual gamma values
    Key innovation: MALWARE gets gamma=4.0 (vs 2.5 in v6)
    """
    # Forward pass
    logits = full_forward(params, X_batch, vqc_circuit, training=training, dropout_key=dropout_key)
    proba = jax.nn.softmax(logits, axis=1)
    proba_c = jnp.clip(proba, 1e-7, 1.0)
    
    # pt: probability of true class
    pt = jnp.sum(Y_oh * proba_c, axis=1)
    
    # Per-sample gamma based on true class
    class_indices = jnp.argmax(Y_oh, axis=1)
    gamma_array = jnp.array([gamma_dict[i] for i in range(Config.NUM_CLASSES)])
    gamma_per_sample = gamma_array[class_indices]
    
    # Focal weight: (1 - pt)^gamma
    focal_wt = (1.0 - pt) ** gamma_per_sample
    
    # Class weights
    sample_cw = jnp.sum(Y_oh * cw_jax[None, :], axis=1)
    
    # Cross-entropy
    ce = -jnp.sum(Y_oh * jnp.log(proba_c), axis=1)
    
    # Weighted focal loss
    base_loss = jnp.mean(focal_wt * sample_cw * ce)
    
    # L2 on weights only
    l2_loss = l2 * (jnp.sum(params['W0'] ** 2) +
                    jnp.sum(params['W1'] ** 2) +
                    jnp.sum(params['W2'] ** 2))
    
    return base_loss + l2_loss


# ============================================================================
# OPTIMIZATION
# ============================================================================

def cosine_lr(epoch, n_epochs, base_lr, min_lr=Config.ADAM_LR_MIN, warmup=Config.ADAM_WARMUP):
    """Cosine LR with warmup"""
    if epoch < warmup:
        return float(base_lr * (epoch + 1) / warmup)
    progress = (epoch - warmup) / max(1, n_epochs - warmup)
    return float(min_lr + 0.5 * (base_lr - min_lr) * (1 + np.cos(np.pi * progress)))


def adam_init(params):
    """Initialize Adam"""
    return (jax.tree_util.tree_map(jnp.zeros_like, params),
            jax.tree_util.tree_map(jnp.zeros_like, params))


def adam_step(params, m, v, t, grads, lr):
    """Adam step"""
    lr = jnp.float32(lr)
    t_n = t + jnp.float32(1.0)
    m_n = jax.tree_util.tree_map(lambda mi, gi: Config.ADAM_B1 * mi + (1 - Config.ADAM_B1) * gi, m, grads)
    v_n = jax.tree_util.tree_map(lambda vi, gi: Config.ADAM_B2 * vi + (1 - Config.ADAM_B2) * gi ** 2, v, grads)
    mh = jax.tree_util.tree_map(lambda mi: mi / (1 - Config.ADAM_B1 ** t_n), m_n)
    vh = jax.tree_util.tree_map(lambda vi: vi / (1 - Config.ADAM_B2 ** t_n), v_n)
    p_n = jax.tree_util.tree_map(lambda pi, mi, vi: pi - lr * mi / (jnp.sqrt(vi) + Config.ADAM_EPS),
                                  params, mh, vh)
    return p_n, m_n, v_n, t_n


# ============================================================================
# INFERENCE
# ============================================================================

def predict_logits(params, X_np, vqc_circuit):
    """Predict logits for full dataset"""
    logits_list = []
    batch_size = Config.BATCH_SIZE
    n_samples = len(X_np)
    
    for i in range(0, n_samples, batch_size):
        X_batch = X_np[i:i + batch_size]
        X_jax = jnp.array(X_batch, dtype=jnp.float32)
        logits = full_forward(params, X_jax, vqc_circuit, training=False)
        logits_list.append(np.array(logits))
    
    return np.vstack(logits_list)


# ============================================================================
# DATA LOADING
# ============================================================================

def build_train_val(X_train_full, Y_train_full, val_frac=Config.VAL_FRAC, seed=Config.SEED):
    """Stratified train/val split"""
    np.random.seed(seed)
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
    """Load Phase 1 prepared data"""
    logging.info("="*70)
    logging.info("LOADING PHASE 1 DATA")
    logging.info("="*70)
    
    X_train_full = np.load(Config.VAE_TRAIN)
    Y_train_full = np.load(Config.Y_TRAIN)
    X_test = np.load(Config.VAE_TEST)
    Y_test = np.load(Config.Y_TEST)
    
    logging.info(f"Train: {X_train_full.shape}, Test: {X_test.shape}")
    logging.info(f"\nTraining class distribution:")
    for c in range(Config.NUM_CLASSES):
        count = (Y_train_full == c).sum()
        pct = 100 * count / len(Y_train_full)
        logging.info(f"  {Config.CLASS_NAMES[c]:<10}: {count:5d} ({pct:.1f}%)")
    
    X_train, Y_train, X_val, Y_val = build_train_val(X_train_full, Y_train_full)
    
    return X_train, Y_train, X_val, Y_val, X_test, Y_test


# ============================================================================
# TRAINING
# ============================================================================

def train_single_restart(tag, X_train_np, Y_train_np, X_val_np, Y_val_np, vqc_circuit, seed=Config.SEED):
    """
    Two-phase training with per-class focal loss
    Phase 1: Head only (150 epochs)
    Phase 2: Joint fine-tune (100 epochs)
    """
    np.random.seed(seed)
    dk_seq = jax.random.PRNGKey(seed)
    cw_jax = jnp.array(Config.CLASS_WEIGHTS_MANUAL, dtype=jnp.float32)
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
    logging.info(f"  Class weights: {Config.CLASS_WEIGHTS_MANUAL}")
    logging.info(f"  PHASE 1: head only ({Config.PHASE1_EPOCHS} epochs, lr={Config.PHASE1_LR})")
    logging.info(f"{'='*70}")
    
    # Phase 1: Head only
    q_fixed = jnp.array(params['q'])
    head_keys = [k for k in params if k != 'q']
    hp = {k: params[k] for k in head_keys}
    
    def p1_loss_fn(head_params, X, y_oh, dropout_key):
        all_p = dict(head_params)
        all_p['q'] = q_fixed
        return per_class_focal_loss(all_p, X, y_oh, vqc_circuit, cw_jax, gamma_dict,
                                     dropout_key=dropout_key, training=True)
    
    m1, v1 = adam_init(hp)
    t1 = jnp.float32(0.0)
    p1_grad_fn = jit(value_and_grad(p1_loss_fn))
    
    best_vf1_p1, best_hp, no_imp_p1 = -1.0, hp, 0
    t_start = time.time()
    
    for epoch in range(1, Config.PHASE1_EPOCHS + 1):
        lr_ep = cosine_lr(epoch - 1, Config.PHASE1_EPOCHS, Config.PHASE1_LR)
        idx = rng.permutation(n_tr)
        Xs, Ys = X_train_np[idx], Y_oh[idx]
        ep_loss = 0.0
        
        for b in range(n_bat):
            s, e = b * Config.BATCH_SIZE, min((b + 1) * Config.BATCH_SIZE, n_tr)
            Xb = jnp.array(Xs[s:e], dtype=jnp.float32)
            Yb = jnp.array(Ys[s:e], dtype=jnp.float32)
            dk_seq, dk = jax.random.split(dk_seq)
            lv, grads = p1_grad_fn(hp, Xb, Yb, dk)
            hp, m1, v1, t1 = adam_step(hp, m1, v1, t1, grads, lr_ep)
            ep_loss += float(lv)
        
        if epoch % Config.LOG_EVERY == 0 or epoch == Config.PHASE1_EPOCHS:
            all_p = dict(hp)
            all_p['q'] = q_fixed
            val_log = predict_logits(all_p, X_val_np, vqc_circuit)
            vf1 = f1_score(Y_val_np, val_log.argmax(1), average='macro', zero_division=0)
            elapsed = time.time() - t_start
            logging.info(f'  EP {epoch:4d}/{Config.PHASE1_EPOCHS}  loss={ep_loss/n_bat:.4f}  '
                        f'val_F1={vf1:.4f}  lr={lr_ep:.2e}  {elapsed:.0f}s')
            if vf1 > best_vf1_p1:
                best_vf1_p1, best_hp, no_imp_p1 = vf1, {k: v for k, v in hp.items()}, 0
            else:
                no_imp_p1 += Config.LOG_EVERY
                if no_imp_p1 >= Config.PHASE1_PATIENCE:
                    logging.info(f'  Early stop (P1) at epoch {epoch}')
                    break
    
    hp = best_hp
    logging.info(f'  Phase 1 best val F1: {best_vf1_p1:.4f}')
    
    # Phase 2: Joint fine-tune
    logging.info(f'\n  PHASE 2: all params ({Config.PHASE2_EPOCHS} epochs, lr={Config.PHASE2_LR})')
    all_p = dict(hp)
    all_p['q'] = q_fixed
    
    def p2_loss_fn(params, X, y_oh):
        return per_class_focal_loss(params, X, y_oh, vqc_circuit, cw_jax, gamma_dict, training=False)
    
    m2, v2 = adam_init(all_p)
    t2 = jnp.float32(0.0)
    p2_grad_fn = jit(value_and_grad(p2_loss_fn))
    
    best_vf1_p2, best_all_p, no_imp_p2 = best_vf1_p1, all_p, 0
    
    for epoch in range(1, Config.PHASE2_EPOCHS + 1):
        lr_ep = cosine_lr(epoch - 1, Config.PHASE2_EPOCHS, Config.PHASE2_LR)
        idx = rng.permutation(n_tr)
        Xs, Ys = X_train_np[idx], Y_oh[idx]
        ep_loss = 0.0
        
        for b in range(n_bat):
            s, e = b * Config.BATCH_SIZE, min((b + 1) * Config.BATCH_SIZE, n_tr)
            Xb = jnp.array(Xs[s:e], dtype=jnp.float32)
            Yb = jnp.array(Ys[s:e], dtype=jnp.float32)
            lv, grads = p2_grad_fn(all_p, Xb, Yb)
            all_p, m2, v2, t2 = adam_step(all_p, m2, v2, t2, grads, lr_ep)
            ep_loss += float(lv)
        
        if epoch % Config.LOG_EVERY == 0 or epoch == Config.PHASE2_EPOCHS:
            val_log = predict_logits(all_p, X_val_np, vqc_circuit)
            vf1 = f1_score(Y_val_np, val_log.argmax(1), average='macro', zero_division=0)
            elapsed = time.time() - t_start
            logging.info(f'  EP {epoch:4d}/{Config.PHASE2_EPOCHS}  loss={ep_loss/n_bat:.4f}  '
                        f'val_F1={vf1:.4f}  lr={lr_ep:.2e}  {elapsed:.0f}s')
            if vf1 > best_vf1_p2:
                best_vf1_p2 = vf1
                best_all_p = {k: np.array(v) for k, v in all_p.items()}
                no_imp_p2 = 0
            else:
                no_imp_p2 += Config.LOG_EVERY
                if no_imp_p2 >= Config.PHASE2_PATIENCE:
                    logging.info(f'  Early stop (P2) at epoch {epoch}')
                    break
    
    logging.info(f'  Phase 2 best val F1: {best_vf1_p2:.4f}')
    logging.info(f'  Total training time: {(time.time() - t_start) / 60:.1f} minutes')
    
    return best_all_p, best_vf1_p2


# ============================================================================
# EVALUATION
# ============================================================================

def find_best_temperature(logits, y, T_range=None):
    """Temperature scaling"""
    if T_range is None:
        T_range = np.concatenate([np.linspace(0.1, 1.0, 30), np.linspace(1.0, 5.0, 20)])
    best_T, best_f1 = 1.0, -1.0
    for T in T_range:
        preds = np.argmax(logits / T, axis=1)
        f1 = f1_score(y, preds, average='macro', zero_division=0)
        if f1 > best_f1:
            best_f1, best_T = f1, T
    return best_T, best_f1


def find_per_class_thresholds(proba, y):
    """Per-class threshold optimization"""
    thresh_range = np.linspace(0.05, 0.90, 35)
    thresholds = np.zeros(Config.NUM_CLASSES)
    
    for c in range(Config.NUM_CLASSES):
        best_t, best_f1c = 0.5, -1.0
        best_prec, best_rec = 0.0, 0.0

        uncon_t, uncon_f1 = 0.5, -1.0
        uncon_prec, uncon_rec = 0.0, 0.0

        precision_floor = None
        if c == 2:
            precision_floor = Config.PROBE_PRECISION_FLOOR
        elif c == 4:
            precision_floor = Config.MALWARE_PRECISION_FLOOR

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

            if f1c > uncon_f1:
                uncon_f1, uncon_t = f1c, t
                uncon_prec, uncon_rec = prec, rec

            if precision_floor is not None and prec < precision_floor:
                continue

            if f1c > best_f1c:
                best_f1c, best_t = f1c, t
                best_prec, best_rec = prec, rec

        if best_f1c < 0 and uncon_f1 >= 0:
            best_f1c, best_t = uncon_f1, uncon_t
            best_prec, best_rec = uncon_prec, uncon_rec
            if precision_floor is not None:
                logging.warning(
                    f'    {Config.CLASS_NAMES[c]:<10}: no threshold met precision floor '
                    f'{precision_floor:.2f}; using best unconstrained threshold={best_t:.2f}'
                )

        thresholds[c] = best_t
        logging.info(
            f'    {Config.CLASS_NAMES[c]:<10}: threshold={best_t:.2f}  '
            f'F1={best_f1c:.4f}  P={best_prec:.4f}  R={best_rec:.4f}'
        )
    
    return thresholds


def predict_with_thresholds(proba, thresholds):
    """Apply per-class thresholds"""
    margin = proba - thresholds[None, :]
    above = proba > thresholds[None, :]
    has_above = above.any(axis=1)
    masked_margin = np.where(above, margin, -np.inf)
    return np.where(has_above, masked_margin.argmax(axis=1), proba.argmax(axis=1))


def softmax_np(x):
    """Softmax"""
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def evaluate_model(params, X_val, Y_val, X_test, Y_test, vqc_circuit, tag):
    """Complete evaluation"""
    logging.info(f"\n{'='*70}")
    logging.info(f"EVALUATING {tag}")
    logging.info(f"{'='*70}")
    
    # Validation
    val_logits = predict_logits(params, X_val, vqc_circuit)
    best_T, _ = find_best_temperature(val_logits, Y_val)
    logging.info(f"\nBest temperature: {best_T:.3f}")
    
    val_logits_scaled = val_logits / best_T
    val_proba = softmax_np(val_logits_scaled)
    
    # Thresholds
    logging.info("\nPer-class threshold optimization:")
    thresholds = find_per_class_thresholds(val_proba, Y_val)
    
    # Test
    test_logits = predict_logits(params, X_test, vqc_circuit)
    test_logits_scaled = test_logits / best_T
    test_proba = softmax_np(test_logits_scaled)
    test_pred_thresh = predict_with_thresholds(test_proba, thresholds)
    
    f1_thresh = f1_score(Y_test, test_pred_thresh, average='macro', zero_division=0)
    logging.info(f"\nTest F1 (threshold): {f1_thresh:.4f}")
    
    # Report
    logging.info("\nClassification Report:")
    report = classification_report(Y_test, test_pred_thresh, 
                                   target_names=Config.CLASS_NAMES, zero_division=0)
    logging.info(f"\n{report}")
    
    return {
        'temperature': float(best_T),
        'thresholds': thresholds.tolist(),
        'f1_thresh': float(f1_thresh),
        'test_proba': test_proba,
        'confusion_matrix': confusion_matrix(Y_test, test_pred_thresh).tolist(),
        'classification_report': classification_report(Y_test, test_pred_thresh,
                                                       target_names=Config.CLASS_NAMES,
                                                       output_dict=True, zero_division=0)
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main training pipeline"""
    
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    log_file = setup_logging(Config.OUTPUT_DIR)
    
    logging.info("="*70)
    logging.info("VQC v7 PHASE 1 TRAINING")
    logging.info("="*70)
    logging.info(f"Output: {Config.OUTPUT_DIR}")
    logging.info(f"Device: {jax.devices()[0].platform.upper()}")
    logging.info(f"PennyLane: {qml.__version__}")
    logging.info(f"JAX: {jax.__version__}")
    
    # Build quantum circuit
    logging.info("\nBuilding quantum circuit...")
    vqc_circuit, dev = build_quantum_circuit()
    logging.info(f"  Device: {Config.DEVICE}")
    logging.info(f"  Qubits: {Config.NUM_QUBITS}")
    logging.info(f"  Observables: 16 (PauliZ + PauliX)")
    
    # Load data
    X_train, Y_train, X_val, Y_val, X_test, Y_test = load_data()
    
    # Train VQC-A
    logging.info("\n" + "="*70)
    logging.info("TRAINING VQC-A")
    logging.info("="*70)
    params_a, val_f1_a = train_single_restart('VQC-A', X_train, Y_train, X_val, Y_val, 
                                               vqc_circuit, seed=Config.SEED)
    results_a = evaluate_model(params_a, X_val, Y_val, X_test, Y_test, vqc_circuit, 'VQC-A')
    
    # Save VQC-A
    output_dir_a = Config.OUTPUT_DIR / "vqc_a"
    os.makedirs(output_dir_a, exist_ok=True)
    np.savez(output_dir_a / "model_params.npz", **params_a)
    np.save(output_dir_a / "test_proba.npy", results_a['test_proba'])
    with open(output_dir_a / "meta.json", 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'model': 'VQC-A',
            'phase': 'Phase 1',
            'seed': Config.SEED,
            'val_f1': float(val_f1_a),
            'test_f1': results_a['f1_thresh'],
            'class_weights': Config.CLASS_WEIGHTS_MANUAL,
            'focal_gamma': Config.FOCAL_GAMMA,
        }, f, indent=2)
    
    # Train VQC-B
    logging.info("\n" + "="*70)
    logging.info("TRAINING VQC-B")
    logging.info("="*70)
    params_b, val_f1_b = train_single_restart('VQC-B', X_train, Y_train, X_val, Y_val,
                                               vqc_circuit, seed=Config.SEED + 1)
    results_b = evaluate_model(params_b, X_val, Y_val, X_test, Y_test, vqc_circuit, 'VQC-B')
    
    # Save VQC-B
    output_dir_b = Config.OUTPUT_DIR / "vqc_b"
    os.makedirs(output_dir_b, exist_ok=True)
    np.savez(output_dir_b / "model_params.npz", **params_b)
    np.save(output_dir_b / "test_proba.npy", results_b['test_proba'])
    with open(output_dir_b / "meta.json", 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'model': 'VQC-B',
            'phase': 'Phase 1',
            'seed': Config.SEED + 1,
            'val_f1': float(val_f1_b),
            'test_f1': results_b['f1_thresh'],
            'class_weights': Config.CLASS_WEIGHTS_MANUAL,
            'focal_gamma': Config.FOCAL_GAMMA,
        }, f, indent=2)
    
    # Ensemble
    logging.info("\n" + "="*70)
    logging.info("ENSEMBLE (VQC-A + VQC-B)")
    logging.info("="*70)
    ensemble_proba = (results_a['test_proba'] + results_b['test_proba']) / 2
    ensemble_pred = ensemble_proba.argmax(axis=1)
    ensemble_f1 = f1_score(Y_test, ensemble_pred, average='macro', zero_division=0)
    logging.info(f"Ensemble F1: {ensemble_f1:.4f}")
    logging.info(f"VQC-A F1:    {results_a['f1_thresh']:.4f}")
    logging.info(f"VQC-B F1:    {results_b['f1_thresh']:.4f}")
    
    # Final summary
    logging.info("\n" + "="*70)
    logging.info("PHASE 1 TRAINING COMPLETE!")
    logging.info("="*70)
    logging.info(f"Target: 0.77-0.78 F1 macro")
    if ensemble_f1 >= 0.77:
        logging.info("[SUCCESS] TARGET ACHIEVED!")
    else:
        logging.info(f"Gap: {0.77 - ensemble_f1:.4f}")
    
    logging.info(f"\nResults saved to: {Config.OUTPUT_DIR}")


if __name__ == "__main__":
    main()
