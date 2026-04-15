#!/usr/bin/env python3
"""
VQC v7 PHASE 1 TRAINING - PyTorch Backend
Optimized for Phase 1 improvements with per-class focal loss

Architecture:
- Neural network with Phase 1 data (ADASYN-enhanced)
- Per-class focal loss with gamma tuning for MALWARE
- Enhanced class weights (MALWARE 15x → 30x)

Expected Results:
- MALWARE F1:  0.20 → 0.36-0.40 (+82%)
- EXPLOIT F1:  0.72 → 0.76-0.78 (+6%)
- Macro F1:    0.73 → 0.77-0.78 (+5.7%)

Training Time: ~30 minutes (CPU) to 2-3 hours (quantum simulation)
"""

import os
import sys
import time
import json
import logging
from pathlib import Path
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, classification_report, confusion_matrix, accuracy_score
from tqdm import tqdm

# Set random seeds
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

print("[OK] All imports successful")

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Phase 1 training configuration"""
    
    PROJECT_ROOT = Path(__file__).parent  # VQC directory
    DATA_DIR = PROJECT_ROOT / "vqc_v7_phase1_output"
    OUTPUT_DIR = PROJECT_ROOT / "vqc_v7_phase1_trained"
    
    # Data
    N_FEATURES = 8  # VAE latent dim
    N_CLASSES = 5
    CLASS_NAMES = ['NORMAL', 'DoS', 'PROBE', 'EXPLOIT', 'MALWARE']
    
    # Neural head architecture (mimics v6 quantum head)
    HIDDEN_SIZES = [128, 64]  # 8 -> 128 -> 64 -> 5
    DROPOUT = 0.3
    USE_LAYERNORM = True  # v6 uses LayerNorm, not BatchNorm
    
    # Training
    BATCH_SIZE = 64
    MAX_EPOCHS_P1 = 150
    MAX_EPOCHS_P2 = 100
    LR_P1 = 0.001
    LR_P2 = 0.0001
    EARLY_STOPPING_P1 = 30
    EARLY_STOPPING_P2 = 20
    WEIGHT_DECAY = 1e-5
    
    # Class weights & loss
    CLASS_WEIGHTS = [1.0, 2.0, 2.0, 5.0, 30.0]  # MALWARE doubled
    FOCAL_GAMMA = {0: 2.0, 1: 2.0, 2: 2.5, 3: 3.0, 4: 4.0}  # Per-class focal loss
    
    # Validation
    VAL_FRAC = 0.1
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Output
    SAVE_BEST_MODEL = True
    SAVE_HISTORY = True


# ============================================================================
# NEURAL NETWORK MODEL
# ============================================================================

class QuantumInspiredNet(nn.Module):
    """Neural network mimicking quantum circuit + head from v6"""
    
    def __init__(self, input_size, hidden_sizes, n_classes, dropout=0.3, use_layernorm=True):
        super().__init__()
        
        layers = []
        prev_size = input_size
        
        # Build hidden layers
        for h_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, h_size))
            if use_layernorm:
                layers.append(nn.LayerNorm(h_size))
            else:
                layers.append(nn.BatchNorm1d(h_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_size = h_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, n_classes))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


# ============================================================================
# PER-CLASS FOCAL LOSS
# ============================================================================

class PerClassFocalLoss(nn.Module):
    """Focal loss with per-class gamma and class weights"""
    
    def __init__(self, class_weights, focal_gamma, alpha=1.0):
        super().__init__()
        self.class_weights = torch.tensor(class_weights, dtype=torch.float32)
        self.focal_gamma = focal_gamma  # dict: class_id -> gamma
        self.alpha = alpha
        self.ce = nn.CrossEntropyLoss(reduction='none')
    
    def forward(self, logits, targets):
        """
        Args:
            logits: (batch_size, n_classes)
            targets: (batch_size,) class indices
        Returns:
            loss: scalar
        """
        # Move weights to same device
        self.class_weights = self.class_weights.to(logits.device)
        
        # Cross entropy
        ce_loss = self.ce(logits, targets)
        
        # Softmax probabilities
        p = torch.softmax(logits, dim=1)
        
        # Get probability of true class
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # Per-class focal gamma
        gamma_t = torch.tensor(
            [self.focal_gamma.get(int(t.item()), 2.0) for t in targets],
            dtype=torch.float32,
            device=logits.device
        )
        
        # Focal weight
        focal_weight = (1.0 - p_t) ** gamma_t
        
        # Class weight
        class_weight = self.class_weights[targets]
        
        # Combined loss
        loss = ce_loss * focal_weight * class_weight * self.alpha
        
        return loss.mean()


# ============================================================================
# DATA LOADING
# ============================================================================

def load_data():
    """Load Phase 1 prepared data"""
    print("\n[INFO] Loading Phase 1 data...")
    
    data_dir = Config.DATA_DIR
    
    X_train = np.load(data_dir / "vae_z_train_sampled.npy")
    y_train = np.load(data_dir / "y_train_sampled.npy")
    X_test = np.load(data_dir / "vae_z_test.npy")
    y_test = np.load(data_dir / "y_test.npy")
    
    print(f"  X_train: {X_train.shape}")
    print(f"  y_train: {y_train.shape}")
    print(f"  X_test:  {X_test.shape}")
    print(f"  y_test:  {y_test.shape}")
    
    # Split train into train/val
    n_train = int(len(X_train) * (1 - Config.VAL_FRAC))
    indices = np.random.permutation(len(X_train))
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]
    
    X_train_split = X_train[train_idx]
    y_train_split = y_train[train_idx]
    X_val = X_train[val_idx]
    y_val = y_train[val_idx]
    
    # Normalize
    scaler = StandardScaler()
    X_train_split = scaler.fit_transform(X_train_split)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    # Convert to torch
    X_train_t = torch.from_numpy(X_train_split).float()
    y_train_t = torch.from_numpy(y_train_split).long()
    X_val_t = torch.from_numpy(X_val).float()
    y_val_t = torch.from_numpy(y_val).long()
    X_test_t = torch.from_numpy(X_test).float()
    y_test_t = torch.from_numpy(y_test).long()
    
    return {
        'X_train': X_train_t,
        'y_train': y_train_t,
        'X_val': X_val_t,
        'y_val': y_val_t,
        'X_test': X_test_t,
        'y_test': y_test_t,
        'scaler': scaler
    }


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def evaluate(model, X, y, device):
    """Evaluate model"""
    model.eval()
    with torch.no_grad():
        X = X.to(device)
        y = y.to(device)
        logits = model(X)
        preds = torch.argmax(logits, dim=1)
        
        accuracy = (preds == y).float().mean().item()
        f1_macro = f1_score(y.cpu().numpy(), preds.cpu().numpy(), average='macro')
        f1_weighted = f1_score(y.cpu().numpy(), preds.cpu().numpy(), average='weighted')
    
    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'preds': preds.cpu().numpy(),
        'y_true': y.cpu().numpy()
    }


def train_epoch(model, train_loader, loss_fn, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        
        # Forward
        logits = model(X_batch)
        loss = loss_fn(logits, y_batch)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)


# ============================================================================
# MAIN TRAINING
# ============================================================================

def main():
    print("\n" + "=" * 70)
    print("VQC v7 PHASE 1 TRAINING - PyTorch Backend")
    print("=" * 70)
    print(f"Device: {Config.DEVICE}")
    
    # Setup
    Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load data
    data = load_data()
    
    # Build model
    print("\n[INFO] Building neural network model...")
    model = QuantumInspiredNet(
        input_size=Config.N_FEATURES,
        hidden_sizes=Config.HIDDEN_SIZES,
        n_classes=Config.N_CLASSES,
        dropout=Config.DROPOUT,
        use_layernorm=Config.USE_LAYERNORM
    ).to(Config.DEVICE)
    
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss function
    loss_fn = PerClassFocalLoss(
        class_weights=Config.CLASS_WEIGHTS,
        focal_gamma=Config.FOCAL_GAMMA
    ).to(Config.DEVICE)
    
    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=Config.LR_P1,
        weight_decay=Config.WEIGHT_DECAY
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,
        T_mult=1,
        eta_min=1e-6
    )
    
    # Data loaders
    train_dataset = TensorDataset(data['X_train'], data['y_train'])
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )
    
    # Training loop
    print("\n[INFO] Phase 1: Head-only training")
    print(f"  Epochs: {Config.MAX_EPOCHS_P1}")
    print(f"  Batch size: {Config.BATCH_SIZE}")
    print(f"  Learning rate: {Config.LR_P1}")
    print(f"  Class weights: {Config.CLASS_WEIGHTS}")
    print()
    
    best_val_f1 = 0.0
    best_epoch = 0
    patience_counter = 0
    history = {'epoch': [], 'train_loss': [], 'val_f1': [], 'val_acc': []}
    
    for epoch in range(Config.MAX_EPOCHS_P1):
        # Train
        train_loss = train_epoch(model, train_loader, loss_fn, optimizer, Config.DEVICE)
        history['train_loss'].append(train_loss)
        
        # Validate every 5 epochs
        if epoch % 5 == 0:
            val_metrics = evaluate(model, data['X_val'], data['y_val'], Config.DEVICE)
            val_f1 = val_metrics['f1_macro']
            val_acc = val_metrics['accuracy']
            
            history['epoch'].append(epoch)
            history['val_f1'].append(val_f1)
            history['val_acc'].append(val_acc)
            
            status = "[BEST]" if val_f1 > best_val_f1 else ""
            print(f"Epoch {epoch+1:3d} | Loss: {train_loss:.4f} | F1: {val_f1:.4f} | Acc: {val_acc:.4f} {status}")
            
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_epoch = epoch
                patience_counter = 0
                # Save best model
                if Config.SAVE_BEST_MODEL:
                    torch.save(model.state_dict(), Config.OUTPUT_DIR / "best_model_p1.pt")
            else:
                patience_counter += 1
            
            if patience_counter >= Config.EARLY_STOPPING_P1:
                print(f"\n[INFO] Early stopping at epoch {epoch+1}")
                break
        
        scheduler.step(epoch)
    
    # Load best model
    model.load_state_dict(torch.load(Config.OUTPUT_DIR / "best_model_p1.pt"))
    
    # Evaluate on all sets
    print("\n" + "=" * 70)
    print("PHASE 1 RESULTS")
    print("=" * 70)
    
    train_metrics = evaluate(model, data['X_train'], data['y_train'], Config.DEVICE)
    val_metrics = evaluate(model, data['X_val'], data['y_val'], Config.DEVICE)
    test_metrics = evaluate(model, data['X_test'], data['y_test'], Config.DEVICE)
    
    print(f"\nTrain F1 Macro: {train_metrics['f1_macro']:.4f}")
    print(f"Val F1 Macro:   {val_metrics['f1_macro']:.4f}")
    print(f"Test F1 Macro:  {test_metrics['f1_macro']:.4f}")
    
    print(f"\nTrain Accuracy: {train_metrics['accuracy']:.4f}")
    print(f"Val Accuracy:   {val_metrics['accuracy']:.4f}")
    print(f"Test Accuracy:  {test_metrics['accuracy']:.4f}")
    
    # Classification report
    print("\n" + "-" * 70)
    print("Test Classification Report:")
    print("-" * 70)
    
    report = classification_report(
        test_metrics['y_true'],
        test_metrics['preds'],
        target_names=Config.CLASS_NAMES,
        digits=4
    )
    print(report)
    
    # Per-class F1 scores
    print("\nPer-Class F1 Scores:")
    print("-" * 40)
    f1_scores_per_class = f1_score(
        test_metrics['y_true'],
        test_metrics['preds'],
        average=None
    )
    for cls_name, f1 in zip(Config.CLASS_NAMES, f1_scores_per_class):
        print(f"  {cls_name:10s}: {f1:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(test_metrics['y_true'], test_metrics['preds'])
    print("\nConfusion Matrix:")
    print("-" * 40)
    for i, row in enumerate(cm):
        print(f"  {Config.CLASS_NAMES[i]:10s}: {row}")
    
    # Save results
    results = {
        'model': 'VQC_v7_Phase1_PyTorch',
        'config': {
            'architecture': 'QuantumInspiredNet',
            'hidden_sizes': Config.HIDDEN_SIZES,
            'dropout': Config.DROPOUT,
            'class_weights': Config.CLASS_WEIGHTS,
            'focal_gamma': Config.FOCAL_GAMMA,
        },
        'training': {
            'epochs_trained': best_epoch + 1,
            'batch_size': Config.BATCH_SIZE,
            'learning_rate': Config.LR_P1,
            'optimizer': 'Adam',
        },
        'metrics': {
            'train': {
                'f1_macro': float(train_metrics['f1_macro']),
                'accuracy': float(train_metrics['accuracy']),
                'f1_weighted': float(train_metrics['f1_weighted']),
            },
            'val': {
                'f1_macro': float(val_metrics['f1_macro']),
                'accuracy': float(val_metrics['accuracy']),
                'f1_weighted': float(val_metrics['f1_weighted']),
            },
            'test': {
                'f1_macro': float(test_metrics['f1_macro']),
                'accuracy': float(test_metrics['accuracy']),
                'f1_weighted': float(test_metrics['f1_weighted']),
                'f1_per_class': [float(f) for f in f1_scores_per_class],
            }
        },
        'confusion_matrix': cm.tolist(),
        'timestamp': datetime.now().isoformat(),
    }
    
    with open(Config.OUTPUT_DIR / "phase1_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    if Config.SAVE_HISTORY:
        with open(Config.OUTPUT_DIR / "phase1_history.json", "w") as f:
            json.dump(history, f, indent=2)
    
    print(f"\n[OK] Results saved to {Config.OUTPUT_DIR / 'phase1_results.json'}")
    print(f"[OK] Model saved to {Config.OUTPUT_DIR / 'best_model_p1.pt'}")
    
    return results


if __name__ == "__main__":
    try:
        results = main()
        print("\n" + "=" * 70)
        print("PHASE 1 TRAINING COMPLETE - SUCCESS")
        print("=" * 70)
    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
