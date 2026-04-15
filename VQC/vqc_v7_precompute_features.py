#!/usr/bin/env python3
"""
VQC v7 QUANTUM FEATURE PRECOMPUTATION
Precompute 16-dim quantum features from VAE latent codes

This script runs quantum circuits ONCE and caches the results,
making subsequent training 100× faster.

Expected runtime: 30-60 minutes (one-time cost)
Output: 16-dim quantum features for 30K training + 573K test samples

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
from tqdm import tqdm

# Qiskit imports
from qiskit import QuantumCircuit
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit_aer import AerSimulator
from qiskit.quantum_info import SparsePauliOp, Statevector

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Quantum feature computation configuration"""
    
    # Paths
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / "VQC" / "vqc_v7_phase1_output"
    OUTPUT_DIR = PROJECT_ROOT / "VQC" / "vqc_v7_quantum_features"
    
    # Input files
    TRAIN_X = DATA_DIR / "vae_z_train_sampled.npy"
    TRAIN_Y = DATA_DIR / "y_train_sampled.npy"
    TEST_X = DATA_DIR / "vae_z_test.npy"
    TEST_Y = DATA_DIR / "y_test.npy"
    
    # Quantum circuit parameters (matching v6)
    N_QUBITS = 8
    RA_REPS = 2  # RealAmplitudes repetitions
    ZZ_REPS = 2  # ZZFeatureMap repetitions
    
    # Random seed for reproducibility
    SEED = 42
    
    # Processing
    CHUNK_SIZE = 500  # Process in chunks to show progress


# ============================================================================
# LOGGING
# ============================================================================

def setup_logging(output_dir):
    """Setup logging"""
    os.makedirs(output_dir / "logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = output_dir / "logs" / f"precompute_{timestamp}.log"
    
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
# QUANTUM CIRCUIT
# ============================================================================

def build_circuit(n_qubits=Config.N_QUBITS, ra_reps=Config.RA_REPS, zz_reps=Config.ZZ_REPS):
    """
    Build VQC circuit identical to v6
    
    Returns:
        qc: QuantumCircuit with parameters
        n_params: Number of variational parameters
    """
    # Feature map: ZZFeatureMap (entangling input features)
    fm = ZZFeatureMap(feature_dimension=n_qubits, reps=zz_reps, entanglement='linear')
    
    # Variational ansatz: RealAmplitudes (trainable parameters)
    va = RealAmplitudes(n_qubits, reps=ra_reps, entanglement='linear')
    
    # Compose circuit
    qc = QuantumCircuit(n_qubits)
    qc.compose(fm, inplace=True)
    qc.compose(va, inplace=True)
    
    return qc, va.num_parameters


def create_pauli_observables(n_qubits):
    """
    Create PauliZ + PauliX observables for all qubits
    Returns 16 observables total (8 Z + 8 X)
    """
    observables = []
    
    # PauliZ for each qubit
    for i in range(n_qubits):
        pauli_string = 'I' * i + 'Z' + 'I' * (n_qubits - i - 1)
        observables.append(SparsePauliOp(pauli_string))
    
    # PauliX for each qubit
    for i in range(n_qubits):
        pauli_string = 'I' * i + 'X' + 'I' * (n_qubits - i - 1)
        observables.append(SparsePauliOp(pauli_string))
    
    return observables


def compute_quantum_features_batch(qc, vae_features, q_params, observables):
    """
    Compute quantum features for a batch of VAE latent codes
    
    Args:
        qc: QuantumCircuit template
        vae_features: (batch_size, 8) VAE latent features
        q_params: Variational parameters (n_params,)
        observables: List of 16 Pauli observables
    
    Returns:
        features: (batch_size, 16) quantum feature matrix
    """
    batch_size = len(vae_features)
    n_features = len(observables)
    features = np.zeros((batch_size, n_features), dtype=np.float32)
    
    for idx, vae_sample in enumerate(vae_features):
        # Bind circuit parameters: input features + variational params
        param_values = np.concatenate([vae_sample, q_params])
        param_dict = dict(zip(qc.parameters, param_values))
        bound_qc = qc.assign_parameters(param_dict)
        
        # Get statevector
        sv = Statevector.from_instruction(bound_qc)
        
        # Compute expectation values for all observables
        for obs_idx, obs in enumerate(observables):
            exp_val = sv.expectation_value(obs).real
            features[idx, obs_idx] = exp_val
    
    return features


# ============================================================================
# MAIN PRECOMPUTATION
# ============================================================================

def precompute_features(X, Y, tag, qc, q_params, observables):
    """
    Precompute quantum features for entire dataset
    
    Args:
        X: (n_samples, 8) VAE latent features
        Y: (n_samples,) labels
        tag: 'train' or 'test'
        qc: QuantumCircuit
        q_params: Variational parameters
        observables: List of Pauli observables
    
    Returns:
        features: (n_samples, 16) quantum features
    """
    n_samples = len(X)
    n_features = len(observables)
    features = np.zeros((n_samples, n_features), dtype=np.float32)
    
    logging.info(f"\nComputing {tag} quantum features...")
    logging.info(f"  Samples: {n_samples}")
    logging.info(f"  Chunk size: {Config.CHUNK_SIZE}")
    
    start_time = time.time()
    
    # Process in chunks with progress bar
    n_chunks = (n_samples + Config.CHUNK_SIZE - 1) // Config.CHUNK_SIZE
    
    with tqdm(total=n_samples, desc=f"Processing {tag}", unit="samples") as pbar:
        for chunk_idx in range(n_chunks):
            chunk_start = chunk_idx * Config.CHUNK_SIZE
            chunk_end = min((chunk_idx + 1) * Config.CHUNK_SIZE, n_samples)
            
            X_chunk = X[chunk_start:chunk_end]
            features_chunk = compute_quantum_features_batch(qc, X_chunk, q_params, observables)
            features[chunk_start:chunk_end] = features_chunk
            
            pbar.update(len(X_chunk))
            
            # Log progress every 10 chunks
            if (chunk_idx + 1) % 10 == 0 or chunk_idx == n_chunks - 1:
                elapsed = time.time() - start_time
                samples_per_sec = (chunk_end) / elapsed
                eta = (n_samples - chunk_end) / samples_per_sec if samples_per_sec > 0 else 0
                logging.info(f"    Progress: {chunk_end}/{n_samples} "
                           f"({100*chunk_end/n_samples:.1f}%) | "
                           f"Speed: {samples_per_sec:.1f} samples/s | "
                           f"ETA: {eta/60:.1f} min")
    
    elapsed = time.time() - start_time
    logging.info(f"  Completed in {elapsed/60:.1f} minutes")
    logging.info(f"  Feature shape: {features.shape}")
    
    return features


def main():
    """Main precomputation pipeline"""
    
    # Setup
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    log_file = setup_logging(Config.OUTPUT_DIR)
    
    logging.info("="*70)
    logging.info("VQC v7 QUANTUM FEATURE PRECOMPUTATION")
    logging.info("="*70)
    logging.info(f"Output directory: {Config.OUTPUT_DIR}")
    logging.info(f"Log file: {log_file}")
    logging.info(f"\nConfiguration:")
    logging.info(f"  Qubits: {Config.N_QUBITS}")
    logging.info(f"  RealAmplitudes reps: {Config.RA_REPS}")
    logging.info(f"  ZZFeatureMap reps: {Config.ZZ_REPS}")
    logging.info(f"  Chunk size: {Config.CHUNK_SIZE}")
    logging.info(f"  Random seed: {Config.SEED}")
    
    # Load data
    logging.info("\n" + "="*70)
    logging.info("LOADING DATA")
    logging.info("="*70)
    
    X_train = np.load(Config.TRAIN_X)
    Y_train = np.load(Config.TRAIN_Y)
    X_test = np.load(Config.TEST_X)
    Y_test = np.load(Config.TEST_Y)
    
    logging.info(f"Train: {X_train.shape}, Labels: {Y_train.shape}")
    logging.info(f"Test:  {X_test.shape}, Labels: {Y_test.shape}")
    
    # Build quantum circuit
    logging.info("\n" + "="*70)
    logging.info("BUILDING QUANTUM CIRCUIT")
    logging.info("="*70)
    
    qc, n_params = build_circuit()
    logging.info(f"  Circuit qubits: {qc.num_qubits}")
    logging.info(f"  Total parameters: {qc.num_parameters}")
    logging.info(f"  Variational parameters: {n_params}")
    logging.info(f"  Feature parameters: {qc.num_parameters - n_params}")
    
    # Initialize random variational parameters
    np.random.seed(Config.SEED)
    q_params = np.random.randn(n_params).astype(np.float32) * 0.1
    logging.info(f"  Initialized variational params with seed {Config.SEED}")
    
    # Create observables
    observables = create_pauli_observables(Config.N_QUBITS)
    logging.info(f"  Observables: {len(observables)} (8 PauliZ + 8 PauliX)")
    
    # Precompute training features
    logging.info("\n" + "="*70)
    logging.info("PRECOMPUTING TRAINING FEATURES")
    logging.info("="*70)
    
    train_features = precompute_features(X_train, Y_train, 'train', qc, q_params, observables)
    
    # Save training features
    np.save(Config.OUTPUT_DIR / "quantum_features_train.npy", train_features)
    np.save(Config.OUTPUT_DIR / "y_train.npy", Y_train)
    logging.info(f"\nSaved: quantum_features_train.npy ({train_features.shape})")
    
    # Precompute test features
    logging.info("\n" + "="*70)
    logging.info("PRECOMPUTING TEST FEATURES")
    logging.info("="*70)
    
    test_features = precompute_features(X_test, Y_test, 'test', qc, q_params, observables)
    
    # Save test features
    np.save(Config.OUTPUT_DIR / "quantum_features_test.npy", test_features)
    np.save(Config.OUTPUT_DIR / "y_test.npy", Y_test)
    logging.info(f"\nSaved: quantum_features_test.npy ({test_features.shape})")
    
    # Save initial quantum parameters (for reference)
    np.save(Config.OUTPUT_DIR / "initial_q_params.npy", q_params)
    
    # Save metadata
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'n_qubits': Config.N_QUBITS,
            'ra_reps': Config.RA_REPS,
            'zz_reps': Config.ZZ_REPS,
            'n_variational_params': int(n_params),
            'n_features': len(observables),
            'seed': Config.SEED
        },
        'data': {
            'train_samples': int(len(X_train)),
            'test_samples': int(len(X_test)),
            'input_dim': int(X_train.shape[1]),
            'output_dim': int(train_features.shape[1])
        },
        'observables': {
            'pauli_z_count': Config.N_QUBITS,
            'pauli_x_count': Config.N_QUBITS,
            'total': len(observables)
        }
    }
    
    with open(Config.OUTPUT_DIR / "precompute_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Sanity checks
    logging.info("\n" + "="*70)
    logging.info("SANITY CHECKS")
    logging.info("="*70)
    
    # Check feature statistics
    logging.info(f"\nTraining features statistics:")
    logging.info(f"  Mean: {train_features.mean():.4f}")
    logging.info(f"  Std:  {train_features.std():.4f}")
    logging.info(f"  Min:  {train_features.min():.4f}")
    logging.info(f"  Max:  {train_features.max():.4f}")
    
    logging.info(f"\nTest features statistics:")
    logging.info(f"  Mean: {test_features.mean():.4f}")
    logging.info(f"  Std:  {test_features.std():.4f}")
    logging.info(f"  Min:  {test_features.min():.4f}")
    logging.info(f"  Max:  {test_features.max():.4f}")
    
    # Check for NaNs or Infs
    train_valid = np.isfinite(train_features).all()
    test_valid = np.isfinite(test_features).all()
    
    if train_valid and test_valid:
        logging.info(f"\n✓ All features are valid (no NaN/Inf)")
    else:
        logging.warning(f"\n⚠ WARNING: Found NaN/Inf in features!")
        logging.warning(f"  Train valid: {train_valid}")
        logging.warning(f"  Test valid: {test_valid}")
    
    # Final summary
    logging.info("\n" + "="*70)
    logging.info("PRECOMPUTATION COMPLETE!")
    logging.info("="*70)
    
    total_samples = len(train_features) + len(test_features)
    logging.info(f"\nProcessed {total_samples:,} samples total")
    logging.info(f"  Training: {len(train_features):,} samples")
    logging.info(f"  Test:     {len(test_features):,} samples")
    
    logging.info(f"\nOutput files:")
    logging.info(f"  {Config.OUTPUT_DIR / 'quantum_features_train.npy'}")
    logging.info(f"  {Config.OUTPUT_DIR / 'quantum_features_test.npy'}")
    logging.info(f"  {Config.OUTPUT_DIR / 'y_train.npy'}")
    logging.info(f"  {Config.OUTPUT_DIR / 'y_test.npy'}")
    logging.info(f"  {Config.OUTPUT_DIR / 'initial_q_params.npy'}")
    logging.info(f"  {Config.OUTPUT_DIR / 'precompute_metadata.json'}")
    
    logging.info(f"\nNext step: Run streamlined training script")
    logging.info("="*70)


if __name__ == "__main__":
    main()
