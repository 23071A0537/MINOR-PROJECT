#!/usr/bin/env python3
"""
VQC v7 PHASE 1 TRAINING - TensorFlow Backend
Quantum Machine Learning with PennyLane + TensorFlow

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
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf

# Suppress TensorFlow logging
tf.get_logger().setLevel('ERROR')

import pennylane as qml
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, classification_report, confusion_matrix

print("[OK] All imports successful")

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Phase 1 training configuration"""
    
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / "vqc_v7_phase1_output"
    OUTPUT_DIR = PROJECT_ROOT / "vqc_v7_phase1_trained"
    
    # Data
    N_FEATURES = 8  # VAE latent dim
    N_CLASSES = 5
    CLASS_NAMES = ['NORMAL', 'DoS', 'PROBE', 'EXPLOIT', 'MALWARE']
    
    # Quantum circuit
    N_QUBITS = 8
    RA_REPS = 2  # RealAmplitudes repetitions
    N_OBSERVABLES = 16  # 8 PauliZ + 8 PauliX
    
    # Neural head
    HEAD_LAYERS = [16, 128, 64, 5]
    
    # Training
    BATCH_SIZE = 64
    MAX_EPOCHS_P1 = 150  # Head-only
    MAX_EPOCHS_P2 = 100  # Fine-tune
    LR_P1 = 0.001
    LR_P2 = 0.0001
    EARLY_STOPPING_P1 = 30
    EARLY_STOPPING_P2 = 20
    
    # Class weights & loss
    CLASS_WEIGHTS = [1, 2, 2, 5, 30]  # MALWARE weight doubled
    FOCAL_GAMMA = {0: 2.0, 1: 2.0, 2: 2.5, 3: 3.0, 4: 4.0}
    
    # Validation
    VAL_FRAC = 0.1
    RANDOM_SEED = 42


# ============================================================================
# QUANTUM CIRCUIT
# ============================================================================

def build_quantum_circuit():
    """Build PennyLane quantum circuit matching v6"""
    dev = qml.device("default.qubit", wires=Config.N_QUBITS)
    
    @qml.qnode(dev, interface="tensorflow", diff_method="backprop")
    def circuit(x, weights):
        # Angle encoding with feature map
        for i, xi in enumerate(x):
            qml.RY(np.pi * xi, wires=i % Config.N_QUBITS)
        
        # ZZFeatureMap: pairwise entanglement
        for i in range(Config.N_QUBITS - 1):
            qml.IsingZZ(2 * np.pi * x[i] * x[i+1], wires=[i, i+1])
        
        # RealAmplitudes with trainable params
        template = qml.templates.RealAmplitudes(
            weights=weights,
            wires=list(range(Config.N_QUBITS)),
            entangling_gate=qml.CNOT,
            n_layers=Config.RA_REPS
        )
        template
        
        # Measure observables
        obs = []
        # 8 PauliZ
        for i in range(Config.N_QUBITS):
            obs.append(qml.expval(qml.PauliZ(i)))
        # 8 PauliX
        for i in range(Config.N_QUBITS):
            obs.append(qml.expval(qml.PauliX(i)))
        
        return tuple(obs)
    
    return circuit


# ============================================================================
# NEURAL NETWORK COMPONENTS
# ============================================================================

class QuantumNeuralNetwork(tf.keras.Model):
    """VQC: Quantum circuit + Neural head"""
    
    def __init__(self, circuit):
        super().__init__()
        self.circuit = circuit
        
        # Initialize quantum weights
        # RealAmplitudes: (n_layers, n_wires, 3)
        n_params = Config.RA_REPS * Config.N_QUBITS * 3
        self.quantum_weights = tf.Variable(
            tf.random.normal([n_params], stddev=0.01),
            trainable=True,
            name="quantum_weights"
        )
        
        # Neural head
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.bn1 = tf.keras.layers.LayerNormalization()
        self.drop1 = tf.keras.layers.Dropout(0.3)
        
        self.dense2 = tf.keras.layers.Dense(64, activation='relu')
        self.bn2 = tf.keras.layers.LayerNormalization()
        self.drop2 = tf.keras.layers.Dropout(0.3)
        
        self.output_layer = tf.keras.layers.Dense(Config.N_CLASSES)
    
    def call(self, x, training=False, freeze_quantum=False):
        """Forward pass"""
        # Quantum features
        batch_size = tf.shape(x)[0]
        
        # Process each sample through circuit
        quantum_features = []
        for i in range(batch_size):
            # Reshape weights for circuit
            w = tf.reshape(self.quantum_weights, [Config.RA_REPS, Config.N_QUBITS, 3])
            features = self.circuit(x[i], w)
            quantum_features.append(features)
        
        quantum_features = tf.stack(quantum_features)
        
        # Neural head
        h = self.dense1(quantum_features)
        h = self.bn1(h, training=training)
        h = self.drop1(h, training=training)
        
        h = self.dense2(h)
        h = self.bn2(h, training=training)
        h = self.drop2(h, training=training)
        
        logits = self.output_layer(h)
        return logits


# ============================================================================
# LOSS FUNCTION
# ============================================================================

def per_class_focal_loss(y_true, y_pred, class_weights, focal_gamma):
    """Per-class focal loss with weighted classes"""
    
    # Convert to probabilities
    probs = tf.nn.softmax(y_pred, axis=-1)
    
    # Cross entropy
    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=tf.argmax(y_true, axis=1),
        logits=y_pred
    )
    
    # Per-class focal weight
    y_class = tf.argmax(y_true, axis=1)
    batch_focal_gamma = tf.gather(
        tf.constant([focal_gamma[i] for i in range(Config.N_CLASSES)], dtype=tf.float32),
        y_class
    )
    
    # Get probability of true class
    pt = tf.reduce_max(y_true * probs, axis=1)
    
    # Focal weight
    focal_weight = tf.pow(1.0 - pt, batch_focal_gamma)
    
    # Class weight
    batch_class_weights = tf.gather(
        tf.constant(class_weights, dtype=tf.float32),
        y_class
    )
    
    # Combined loss
    loss = ce * focal_weight * batch_class_weights
    
    return tf.reduce_mean(loss)


# ============================================================================
# TRAINING UTILITIES
# ============================================================================

def load_data():
    """Load Phase 1 prepared data"""
    print("\n[INFO] Loading Phase 1 data...")
    
    data = {
        'X_train': np.load(Config.DATA_DIR / "vae_z_train_sampled.npy"),
        'y_train': np.load(Config.DATA_DIR / "y_train_sampled.npy"),
        'X_test': np.load(Config.DATA_DIR / "vae_z_test.npy"),
        'y_test': np.load(Config.DATA_DIR / "y_test.npy"),
    }
    
    print(f"  X_train: {data['X_train'].shape}")
    print(f"  y_train: {data['y_train'].shape}")
    print(f"  X_test:  {data['X_test'].shape}")
    print(f"  y_test:  {data['y_test'].shape}")
    
    # Split into train/val
    n_train = int(len(data['X_train']) * (1 - Config.VAL_FRAC))
    
    X_train, y_train = data['X_train'][:n_train], data['y_train'][:n_train]
    X_val, y_val = data['X_train'][n_train:], data['y_train'][n_train:]
    X_test, y_test = data['X_test'], data['y_test']
    
    # Normalize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    # Convert to one-hot
    y_train_onehot = tf.one_hot(y_train, Config.N_CLASSES)
    y_val_onehot = tf.one_hot(y_val, Config.N_CLASSES)
    y_test_onehot = tf.one_hot(y_test, Config.N_CLASSES)
    
    return {
        'X_train': tf.constant(X_train, dtype=tf.float32),
        'y_train': y_train_onehot,
        'X_val': tf.constant(X_val, dtype=tf.float32),
        'y_val': y_val_onehot,
        'X_test': tf.constant(X_test, dtype=tf.float32),
        'y_test': y_test_onehot,
        'scaler': scaler
    }


def evaluate(model, X, y, stage=""):
    """Evaluate model"""
    logits = model(X, training=False)
    preds = tf.argmax(logits, axis=1).numpy()
    y_true = tf.argmax(y, axis=1).numpy()
    
    f1_macro = f1_score(y_true, preds, average='macro')
    f1_weighted = f1_score(y_true, preds, average='weighted')
    accuracy = tf.reduce_mean(
        tf.cast(tf.equal(preds, y_true), tf.float32)
    ).numpy()
    
    return {
        'accuracy': float(accuracy),
        'f1_macro': float(f1_macro),
        'f1_weighted': float(f1_weighted),
        'preds': preds,
        'y_true': y_true
    }


# ============================================================================
# MAIN TRAINING
# ============================================================================

def main():
    print("\n" + "=" * 70)
    print("VQC v7 PHASE 1 TRAINING - TensorFlow Backend")
    print("=" * 70)
    
    # Setup
    Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.random.seed(Config.RANDOM_SEED)
    tf.random.set_seed(Config.RANDOM_SEED)
    
    # Load data
    data = load_data()
    
    # Build model
    print("\n[INFO] Building quantum circuit...")
    circuit = build_quantum_circuit()
    
    print("[INFO] Building neural network...")
    model = QuantumNeuralNetwork(circuit)
    
    # Compile
    optimizer = tf.keras.optimizers.Adam(learning_rate=Config.LR_P1)
    
    # Training loop
    print("\n[INFO] Phase 1: Head-only training (quantum frozen)")
    print(f"  Epochs: {Config.MAX_EPOCHS_P1}")
    print(f"  Batch size: {Config.BATCH_SIZE}")
    print(f"  Learning rate: {Config.LR_P1}")
    print(f"  Class weights: {Config.CLASS_WEIGHTS}")
    print()
    
    best_val_f1 = 0.0
    best_epoch = 0
    patience_counter = 0
    
    history = {'train_loss': [], 'val_f1': []}
    
    # Create datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((data['X_train'], data['y_train']))
    train_dataset = train_dataset.shuffle(len(data['X_train'])).batch(Config.BATCH_SIZE)
    
    for epoch in range(Config.MAX_EPOCHS_P1):
        # Training
        epoch_loss = 0.0
        n_batches = 0
        
        for X_batch, y_batch in train_dataset:
            with tf.GradientTape() as tape:
                logits = model(X_batch, training=True)
                loss = per_class_focal_loss(
                    y_batch, logits,
                    Config.CLASS_WEIGHTS,
                    Config.FOCAL_GAMMA
                )
            
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            
            epoch_loss += loss.numpy()
            n_batches += 1
        
        epoch_loss /= n_batches
        history['train_loss'].append(float(epoch_loss))
        
        # Validation
        if epoch % 5 == 0:
            val_metrics = evaluate(model, data['X_val'], data['y_val'])
            val_f1 = val_metrics['f1_macro']
            history['val_f1'].append(val_f1)
            
            status = "[BEST]" if val_f1 > best_val_f1 else ""
            print(f"Epoch {epoch+1:3d} | Loss: {epoch_loss:.4f} | Val F1: {val_f1:.4f} {status}")
            
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_epoch = epoch
                patience_counter = 0
                # Save best model
                model.save_weights(str(Config.OUTPUT_DIR / "best_model_p1.h5"))
            else:
                patience_counter += 1
            
            if patience_counter >= Config.EARLY_STOPPING_P1:
                print(f"\n[INFO] Early stopping at epoch {epoch+1} (no improvement for {patience_counter} evals)")
                break
    
    # Load best model
    model.load_weights(str(Config.OUTPUT_DIR / "best_model_p1.h5"))
    
    # Evaluate
    print("\n" + "=" * 70)
    print("PHASE 1 RESULTS")
    print("=" * 70)
    
    train_metrics = evaluate(model, data['X_train'], data['y_train'])
    val_metrics = evaluate(model, data['X_val'], data['y_val'])
    test_metrics = evaluate(model, data['X_test'], data['y_test'])
    
    print(f"\nTrain F1 Macro: {train_metrics['f1_macro']:.4f}")
    print(f"Val F1 Macro:   {val_metrics['f1_macro']:.4f}")
    print(f"Test F1 Macro:  {test_metrics['f1_macro']:.4f}")
    
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
    
    # Save results
    results = {
        'model': 'VQC_v7_Phase1_TensorFlow',
        'epoch_best': int(best_epoch),
        'metrics': {
            'train': {
                'f1_macro': train_metrics['f1_macro'],
                'accuracy': train_metrics['accuracy'],
            },
            'val': {
                'f1_macro': val_metrics['f1_macro'],
                'accuracy': val_metrics['accuracy'],
            },
            'test': {
                'f1_macro': test_metrics['f1_macro'],
                'accuracy': test_metrics['accuracy'],
                'f1_weighted': test_metrics['f1_weighted'],
            }
        },
        'confusion_matrix': confusion_matrix(test_metrics['y_true'], test_metrics['preds']).tolist(),
        'timestamp': datetime.now().isoformat(),
    }
    
    with open(Config.OUTPUT_DIR / "phase1_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n[OK] Results saved to {Config.OUTPUT_DIR / 'phase1_results.json'}")
    print(f"[OK] Model saved to {Config.OUTPUT_DIR / 'best_model_p1.h5'}")
    
    return results


if __name__ == "__main__":
    try:
        results = main()
        print("\n" + "=" * 70)
        print("PHASE 1 TRAINING COMPLETE")
        print("=" * 70)
    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
