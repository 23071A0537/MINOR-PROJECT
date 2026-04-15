#!/usr/bin/env python3
"""
VQC v7 PHASE 1 TRAINING - CORRECTED (No ADASYN)
Using only class weights + per-class focal loss (no resampling)

The previous attempt failed due to ADASYN creating severe distribution mismatch.
This version:
- Uses ORIGINAL training distribution (no resampling)
- Applies HIGHER class weights: MALWARE 50x, EXPLOIT 10x
- Uses aggressive focal loss (gamma 4.0-5.0 for MALWARE)
- Tests on original distribution (should work!)
"""

import json
import time
from pathlib import Path
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from tqdm import tqdm

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

print("[OK] Imports successful")

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Phase 1 corrected training configuration"""
    
    PROJECT_ROOT = Path(__file__).parent
    DATA_DIR = PROJECT_ROOT / "vqc_v7_phase1_output"
    OUTPUT_DIR = PROJECT_ROOT / "vqc_v7_phase1_trained_v2"
    
    # Data
    N_FEATURES = 8
    N_CLASSES = 5
    CLASS_NAMES = ['NORMAL', 'DoS', 'PROBE', 'EXPLOIT', 'MALWARE']
    
    # Architecture
    HIDDEN_SIZES = [128, 64]
    DROPOUT = 0.3
    USE_LAYERNORM = True
    
    # Training - WITHOUT ADASYN resampling
    BATCH_SIZE = 256  # Larger batch for imbalanced data
    MAX_EPOCHS_P1 = 200  # More epochs to learn
    LR_P1 = 0.0005  # Lower LR for stability
    EARLY_STOPPING_P1 = 50  # More patience
    WEIGHT_DECAY = 1e-4
    
    # Class weights - MUCH HIGHER for imbalance
    # Using original distribution ratios
    CLASS_WEIGHTS = [1.0, 6.0, 15.0, 60.0, 100.0]  # MALWARE 100x!
    FOCAL_GAMMA = {0: 2.0, 1: 2.0, 2: 2.5, 3: 4.0, 4: 5.0}  # MALWARE gamma 5.0!
    
    # Validation
    VAL_FRAC = 0.1
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============================================================================
# NEURAL NETWORK
# ============================================================================

class QuantumInspiredNet(nn.Module):
    """Neural network mimicking v6 quantum head"""
    
    def __init__(self, input_size, hidden_sizes, n_classes, dropout=0.3, use_layernorm=True):
        super().__init__()
        
        layers = []
        prev_size = input_size
        
        for h_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, h_size))
            if use_layernorm:
                layers.append(nn.LayerNorm(h_size))
            else:
                layers.append(nn.BatchNorm1d(h_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_size = h_size
        
        layers.append(nn.Linear(prev_size, n_classes))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


# ============================================================================
# LOSS FUNCTION
# ============================================================================

class PerClassFocalLoss(nn.Module):
    """Focal loss with per-class gamma and class weights"""
    
    def __init__(self, class_weights, focal_gamma, alpha=1.0):
        super().__init__()
        self.class_weights = torch.tensor(class_weights, dtype=torch.float32)
        self.focal_gamma = focal_gamma
        self.alpha = alpha
        self.ce = nn.CrossEntropyLoss(reduction='none')
    
    def forward(self, logits, targets):
        self.class_weights = self.class_weights.to(logits.device)
        
        ce_loss = self.ce(logits, targets)
        p = torch.softmax(logits, dim=1)
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        gamma_t = torch.tensor(
            [self.focal_gamma.get(int(t.item()), 2.0) for t in targets],
            dtype=torch.float32,
            device=logits.device
        )
        
        focal_weight = (1.0 - p_t) ** gamma_t
        class_weight = self.class_weights[targets]
        
        loss = ce_loss * focal_weight * class_weight * self.alpha
        return loss.mean()


# ============================================================================
# DATA LOADING (Original distribution, no resampling)
# ============================================================================

def load_data_original_distribution():
    """Load data WITHOUT ADASYN resampling - use original distribution"""
    print("\n[INFO] Loading original test data (no ADASYN resampling)...")
    
    data_dir = Config.DATA_DIR
    
    # Load original test data (not ADASYN resampled)
    X_test = np.load(data_dir / "vae_z_test.npy")
    y_test = np.load(data_dir / "y_test.npy")
    
    print(f"  X_test: {X_test.shape}")
    print(f"  y_test: {y_test.shape}")
    
    # For training, ALSO load original (no ADASYN)
    # We'll use class weights instead of resampling
    X_train_orig = np.load(data_dir / "vae_z_train_sampled.npy")  # This has original distribution
    y_train_orig = np.load(data_dir / "y_train_sampled.npy")
    
    # Actually, let's use the original v6 data
    # Load from VQC v6 if available
    vqc_v6_dir = Path("c:\\Users\\G.Monish Reddy\\Desktop\\MINOR PROJECT\\vqc_ensemble_v6")
    
    # For now, use stratified split of original + ADASYN mixed
    # Split into train/val from test set
    n_total_train = len(X_train_orig)
    n_train = int(n_total_train * (1 - Config.VAL_FRAC))
    
    indices = np.random.permutation(n_total_train)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]
    
    X_train = X_train_orig[train_idx]
    y_train = y_train_orig[train_idx]
    X_val = X_train_orig[val_idx]
    y_val = y_train_orig[val_idx]
    
    print(f"\n  Training distribution:")
    unique, counts = np.unique(y_train, return_counts=True)
    for cls_id, count in zip(unique, counts):
        pct = 100 * count / len(y_train)
        print(f"    {Config.CLASS_NAMES[cls_id]}: {count:5d} ({pct:5.1f}%)")
    
    print(f"\n  Test distribution:")
    unique, counts = np.unique(y_test, return_counts=True)
    for cls_id, count in zip(unique, counts):
        pct = 100 * count / len(y_test)
        print(f"    {Config.CLASS_NAMES[cls_id]}: {count:5d} ({pct:5.1f}%)")
    
    # Normalize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    # Convert to torch
    X_train_t = torch.from_numpy(X_train).float()
    y_train_t = torch.from_numpy(y_train).long()
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
# TRAINING
# ============================================================================

def evaluate(model, X, y, device):
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
    model.train()
    total_loss = 0.0
    
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        
        logits = model(X_batch)
        loss = loss_fn(logits, y_batch)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "=" * 70)
    print("VQC v7 PHASE 1 CORRECTED - No ADASYN Resampling")
    print("=" * 70)
    print(f"Device: {Config.DEVICE}")
    
    Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load data
    data = load_data_original_distribution()
    
    # Build model
    print("\n[INFO] Building model...")
    model = QuantumInspiredNet(
        input_size=Config.N_FEATURES,
        hidden_sizes=Config.HIDDEN_SIZES,
        n_classes=Config.N_CLASSES,
        dropout=Config.DROPOUT,
        use_layernorm=Config.USE_LAYERNORM
    ).to(Config.DEVICE)
    
    loss_fn = PerClassFocalLoss(
        class_weights=Config.CLASS_WEIGHTS,
        focal_gamma=Config.FOCAL_GAMMA
    ).to(Config.DEVICE)
    
    optimizer = optim.Adam(
        model.parameters(),
        lr=Config.LR_P1,
        weight_decay=Config.WEIGHT_DECAY
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=1, eta_min=1e-6
    )
    
    train_dataset = TensorDataset(data['X_train'], data['y_train'])
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )
    
    # Training
    print("\n[INFO] Starting training...")
    print(f"  Epochs: {Config.MAX_EPOCHS_P1}")
    print(f"  Class weights: {Config.CLASS_WEIGHTS}")
    print(f"  Focal gamma: {Config.FOCAL_GAMMA}")
    print()
    
    best_val_f1 = 0.0
    best_epoch = 0
    patience_counter = 0
    history = {'epoch': [], 'train_loss': [], 'val_f1': []}
    
    for epoch in range(Config.MAX_EPOCHS_P1):
        train_loss = train_epoch(model, train_loader, loss_fn, optimizer, Config.DEVICE)
        history['train_loss'].append(train_loss)
        
        if epoch % 10 == 0:
            val_metrics = evaluate(model, data['X_val'], data['y_val'], Config.DEVICE)
            val_f1 = val_metrics['f1_macro']
            
            history['epoch'].append(epoch)
            history['val_f1'].append(val_f1)
            
            status = "[BEST]" if val_f1 > best_val_f1 else ""
            print(f"Epoch {epoch:3d} | Loss: {train_loss:.4f} | F1: {val_f1:.4f} {status}")
            
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_epoch = epoch
                patience_counter = 0
                torch.save(model.state_dict(), Config.OUTPUT_DIR / "best_model_p1_v2.pt")
            else:
                patience_counter += 1
            
            if patience_counter >= Config.EARLY_STOPPING_P1:
                print(f"\n[INFO] Early stopping at epoch {epoch}")
                break
        
        scheduler.step(epoch)
    
    # Evaluate
    print("\n" + "=" * 70)
    print("PHASE 1 CORRECTED RESULTS")
    print("=" * 70)
    
    model.load_state_dict(torch.load(Config.OUTPUT_DIR / "best_model_p1_v2.pt"))
    
    train_metrics = evaluate(model, data['X_train'], data['y_train'], Config.DEVICE)
    val_metrics = evaluate(model, data['X_val'], data['y_val'], Config.DEVICE)
    test_metrics = evaluate(model, data['X_test'], data['y_test'], Config.DEVICE)
    
    print(f"\nTrain F1 Macro: {train_metrics['f1_macro']:.4f}")
    print(f"Val F1 Macro:   {val_metrics['f1_macro']:.4f}")
    print(f"TEST F1 MACRO:  {test_metrics['f1_macro']:.4f}  ← ANSWER")
    
    print(f"\nTest Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"Test F1 Weighted: {test_metrics['f1_weighted']:.4f}")
    
    print("\nPer-Class F1:")
    f1_scores_per_class = f1_score(
        test_metrics['y_true'],
        test_metrics['preds'],
        average=None
    )
    for cls_name, f1 in zip(Config.CLASS_NAMES, f1_scores_per_class):
        print(f"  {cls_name:10s}: {f1:.4f}")
    
    cm = confusion_matrix(test_metrics['y_true'], test_metrics['preds'])
    
    results = {
        'model': 'VQC_v7_Phase1_Corrected_NoADASYN',
        'epochs_trained': best_epoch + 1,
        'metrics': {
            'train': {
                'f1_macro': float(train_metrics['f1_macro']),
                'accuracy': float(train_metrics['accuracy']),
            },
            'val': {
                'f1_macro': float(val_metrics['f1_macro']),
                'accuracy': float(val_metrics['accuracy']),
            },
            'test': {
                'f1_macro': float(test_metrics['f1_macro']),
                'accuracy': float(test_metrics['accuracy']),
                'f1_weighted': float(test_metrics['f1_weighted']),
                'f1_per_class': [float(f) for f in f1_scores_per_class],
            }
        },
        'confusion_matrix': cm.tolist(),
        'config': {
            'class_weights': Config.CLASS_WEIGHTS,
            'focal_gamma': Config.FOCAL_GAMMA,
            'lr': Config.LR_P1,
            'batch_size': Config.BATCH_SIZE,
        },
        'timestamp': datetime.now().isoformat(),
    }
    
    with open(Config.OUTPUT_DIR / "phase1_results_v2.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n[OK] Results saved to {Config.OUTPUT_DIR / 'phase1_results_v2.json'}")
    print(f"[OK] Model saved to {Config.OUTPUT_DIR / 'best_model_p1_v2.pt'}")
    
    return results


if __name__ == "__main__":
    try:
        results = main()
        print("\n" + "=" * 70)
        print("TRAINING COMPLETE")
        print("=" * 70)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
