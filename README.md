# Quantum-Classical Hybrid Machine Learning Pipeline

This repository contains a comprehensive machine learning pipeline designed to leverage both classical and quantum computing techniques. The pipeline integrates classical preprocessing, dimensionality reduction using Variational Autoencoders (VAE), and an ensemble of advanced classical models (XGBoost, Random Forest) with a Variational Quantum Classifier (VQC).

## Project Architecture

The pipeline is divided into five distinct stages:

1. **Stage 1 & 2: PreProcessing**
   - Handles raw data ingestion, missing value imputation, scaling, and feature engineering.
   - Saves processed datasets as memory-mapped files and parquets for efficient loading.

2. **Stage 3: Variational Autoencoder (VAE)**
   - A deep learning approach to reduce the dimensionality of the preprocessed data.
   - Extracts a latent representation (`z_train`, `z_test`) that captures the most critical features while filtering out noise.

3. **Stage 4: Variational Quantum Classifier (VQC)**
   - Utilizes quantum circuits (via PennyLane/Qiskit) parameterized by classical weights.
   - Trains on the VAE latent space to classify data, exploiting quantum feature maps.

4. **Stage 5: Hybrid Ensemble Layer**
   - Combines the predictive power of the quantum model (VQC) with classical state-of-the-art models (XGBoost and Random Forest).
   - Generates weighted probabilities to form a robust, final prediction.

5. **Live Inference & Explainability**
   - Features real-time inference scripts (`live_inference.py`, `diagnose_malware.py`).
   - Integrates **SHAP** and **LIME** for local and global model interpretability, ensuring the predictions of the ensemble are transparent and explainable.

## Key Results and Performance

- **Enhanced Accuracy**: By ensembling VQC with XGBoost and Random Forest, the model achieves highly robust F1 scores across multiple classes.
- **Dimensionality Reduction**: The VAE successfully compresses high-dimensional input into a dense latent space (e.g., 16 dimensions), significantly reducing the computational overhead for the quantum simulator.
- **Explainability**: SHAP and LIME integrations provide deep insights into feature importance, allowing for "glass-box" model interpretations that are crucial for diagnostic use cases.

## Getting Started

### Prerequisites
- Python 3.8+
- Required packages can be installed via the setup scripts:
  ```bash
  python setup_deps.py
  # or
  install_deps.bat
  ```

### Running the Pipeline
The entire execution is controlled via `configs/pipeline/config.json`. You can selectively turn on/off stages.

To execute the pipeline:
```bash
python scripts/run_pipeline.py --config configs/pipeline/config.json
```

### Live Inference
To run a live inference batch (with random sampling or specific inputs):
```bash
python scripts/live_inference.py --config configs/pipeline/config.json
```

## Security Note
This repository has been scrubbed of any personal paths and API keys to ensure safe public viewership. If running the VQC on real quantum hardware (e.g., IBM Quantum), you will need to securely provide your own API token.
