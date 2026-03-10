# Class Diagram Design — Explainable Hybrid Quantum-Classical NIDS

## Design Philosophy

The class diagram follows **layered architecture** aligned with your system's data flow pipeline, applying SOLID principles and key design patterns for extensibility and testability.

---

## Architecture Layers → Class Mapping

### Layer 1: Data Management

| Class              | Responsibility                                      |
| ------------------ | --------------------------------------------------- |
| `DatasetManager`   | Load/split NSL-KDD, UNSW-NB15, CICIDS2017 datasets  |
| `DataPreprocessor` | Min-Max normalization, One-Hot encoding, imputation |
| `ClassBalancer`    | SMOTE oversampling, stratified sampling             |

### Layer 2: Feature Engineering (Strategy Pattern)

| Class                        | Responsibility                                                                        |
| ---------------------------- | ------------------------------------------------------------------------------------- |
| `«interface» FeatureReducer` | Common interface: `fit()`, `transform()`, `fitTransform()`                            |
| `PCAReducer`                 | PCA-based dimensionality reduction                                                    |
| `AutoencoderReducer`         | Standard autoencoder (deterministic)                                                  |
| `VariationalAutoencoder`     | **VAE** — probabilistic encoding (49→8 features), KL divergence + reconstruction loss |

> **Why Strategy Pattern?** Your project compares PCA vs Autoencoder vs VAE. A common interface lets you swap reducers without changing pipeline code.

### Layer 3: Classical Models (Template Method Pattern)

| Class                       | Responsibility                                                        |
| --------------------------- | --------------------------------------------------------------------- |
| `«abstract» BaseClassifier` | Abstract base: `train()`, `predict()`, `predictProba()`, `evaluate()` |
| `SVMClassifier`             | SVM with RBF kernel, C parameter                                      |
| `RandomForestModel`         | 100 trees, Gini impurity, feature importance                          |
| `XGBoostModel`              | Gradient boosting, learning rate tuning                               |

> **Key:** Both quantum and classical classifiers extend `BaseClassifier`, enabling the `HybridEnsemble` to treat them uniformly (Liskov Substitution Principle).

### Layer 4: Quantum Processing

| Class                   | Responsibility                                                    |
| ----------------------- | ----------------------------------------------------------------- |
| `QuantumCircuitBuilder` | Factory for creating feature maps and ansatz circuits             |
| `QuantumFeatureMap`     | ZZ/Z feature maps with configurable reps and entanglement         |
| `QuantumAnsatz`         | RealAmplitudes / EfficientSU2, trainable parameters               |
| `QuantumBackend`        | Simulator (Aer) or real hardware (IBM Brisbane), noise models     |
| `VQCClassifier`         | Variational Quantum Classifier — COBYLA optimizer, 100 iterations |
| `QSVMClassifier`        | Quantum kernel SVM — static kernel estimation                     |

> **Why separate `QuantumCircuitBuilder`?** It acts as a **Builder Pattern** — isolates circuit construction complexity from the classifier logic. You can experiment with different feature map depths (reps=2,3,4) and entanglement patterns (linear, circular, full) independently.

### Layer 5: Hybrid Ensemble (Composite Pattern)

| Class                | Responsibility                                                                    |
| -------------------- | --------------------------------------------------------------------------------- |
| `HybridEnsemble`     | Weighted voting: `0.5*quantum + 0.3*xgb + 0.2*rf`                                 |
| `MultiClassStrategy` | One-vs-Rest quantum circuits for multi-class (5 classes NSL-KDD, 10 classes NB15) |

> **Weight optimization** is done via validation set — `optimizeWeights()` tunes the 3 weights.

### Layer 6: Explainability (Strategy Pattern)

| Class                   | Responsibility                                                               |
| ----------------------- | ---------------------------------------------------------------------------- |
| `«interface» Explainer` | Common XAI interface                                                         |
| `SHAPExplainer`         | KernelExplainer — global feature importance, summary/dependence/force plots  |
| `LIMEExplainer`         | LimeTabularExplainer — local instance-level explanations                     |
| `ExplanationResult`     | Value object: feature importance, local/global explanations, attack insights |
| `ExplanationVisualizer` | Publication-quality plots, dashboard generation                              |

### Layer 7: Evaluation & Output

| Class               | Responsibility                                                               |
| ------------------- | ---------------------------------------------------------------------------- |
| `EvaluationMetrics` | Accuracy, Precision, Recall, F1, confusion matrix, cross-validation, t-tests |
| `ThreatAssessment`  | Output object: classification + confidence + explanation                     |
| `NIDSPipeline`      | **Orchestrator (Facade Pattern)** — wires everything together                |

---

## Key Design Patterns Used

| Pattern             | Where                         | Why                                                               |
| ------------------- | ----------------------------- | ----------------------------------------------------------------- |
| **Strategy**        | `FeatureReducer`, `Explainer` | Swap PCA/AE/VAE and SHAP/LIME without changing pipeline           |
| **Template Method** | `BaseClassifier`              | Uniform interface for classical + quantum classifiers             |
| **Builder**         | `QuantumCircuitBuilder`       | Complex circuit construction with configurable depth/entanglement |
| **Facade**          | `NIDSPipeline`                | Single entry point hiding system complexity                       |
| **Composite**       | `HybridEnsemble`              | Treats multiple classifiers as one                                |

---

## Key Relationships

```
NIDSPipeline ◆── DatasetManager         (composition — owns lifecycle)
NIDSPipeline ◆── DataPreprocessor       (composition)
NIDSPipeline ◆── HybridEnsemble         (composition)
NIDSPipeline ◆── EvaluationMetrics      (composition)
NIDSPipeline ──▷ FeatureReducer          (dependency — injected)
NIDSPipeline ──▷ Explainer               (dependency — injected)
NIDSPipeline ──▷ ThreatAssessment        (produces)

HybridEnsemble ◇── VQCClassifier        (aggregation — shared)
HybridEnsemble ◇── BaseClassifier[0..*]  (aggregation — classical models)

VQCClassifier ◇── QuantumFeatureMap      (aggregation)
VQCClassifier ◇── QuantumAnsatz          (aggregation)
VQCClassifier ◇── QuantumBackend         (aggregation)

FeatureReducer ◁·· PCAReducer            (implements)
FeatureReducer ◁·· VariationalAutoencoder (implements)
BaseClassifier ◁── SVMClassifier          (extends)
BaseClassifier ◁── VQCClassifier          (extends — quantum IS-A classifier)
```

---

## Multi-Class Extension Points

For the multi-class requirement (NSL-KDD: Normal/DoS/Probe/R2L/U2R; NB15: 10 attack families):

- `MultiClassStrategy` handles One-vs-Rest quantum circuits
- `BaseClassifier.predictProba()` returns probability vectors for all classes
- `HybridEnsemble` aggregates multi-class probabilities across models

---

## How This Maps to Your Team (4 Members)

| Member                     | Classes to Implement                                                                                                                                  |
| -------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| Member 1 (Classical + VAE) | `DatasetManager`, `DataPreprocessor`, `ClassBalancer`, `VariationalAutoencoder`, `PCAReducer`, `AutoencoderReducer`                                   |
| Member 2 (Quantum)         | `QuantumCircuitBuilder`, `QuantumFeatureMap`, `QuantumAnsatz`, `QuantumBackend`, `VQCClassifier`, `QSVMClassifier`                                    |
| Member 3 (XAI)             | `SHAPExplainer`, `LIMEExplainer`, `ExplanationResult`, `ExplanationVisualizer`                                                                        |
| Member 4 (Integration)     | `HybridEnsemble`, `MultiClassStrategy`, `NIDSPipeline`, `EvaluationMetrics`, `ThreatAssessment`, `SVMClassifier`, `RandomForestModel`, `XGBoostModel` |

---

## Total: 22 Classes, 2 Interfaces, 1 Abstract Class

---

---

# Use Case Diagram — Explainable Hybrid Quantum-Classical NIDS

## Actors

| Actor                    | Type            | Description                                                                                          |
| ------------------------ | --------------- | ---------------------------------------------------------------------------------------------------- |
| **Security Analyst**     | Primary         | End user who monitors threats, views classifications, reviews explanations, analyzes false positives |
| **Data Scientist**       | Primary         | Builds/trains models, engineers features, evaluates performance, configures quantum circuits         |
| **System Administrator** | Secondary       | Manages quantum backends, loads datasets, configures infrastructure                                  |
| **IBM Quantum Cloud**    | External System | Provides real quantum hardware execution (127-qubit IBM Brisbane)                                    |

---

## Use Case Catalog (32 Use Cases in 8 Subsystems)

### Subsystem 1: Data Management

| UC ID | Use Case                        | Actor(s)                 | Description                                                   |
| ----- | ------------------------------- | ------------------------ | ------------------------------------------------------------- |
| UC1   | Load Benchmark Dataset          | Data Scientist, SysAdmin | Load NSL-KDD, UNSW-NB15, or CICIDS2017 dataset                |
| UC2   | Preprocess Network Traffic Data | Data Scientist           | Run full preprocessing pipeline                               |
| UC2a  | Normalize Features              | Data Scientist           | Min-Max scaling to [0,1] — **≪include≫** from UC2             |
| UC2b  | Encode Categorical Features     | Data Scientist           | One-Hot encoding for protocol types — **≪include≫** from UC2  |
| UC2c  | Impute Missing Values           | Data Scientist           | Handle missing data — **≪include≫** from UC2                  |
| UC3   | Balance Class Distribution      | Data Scientist           | Apply SMOTE oversampling — **≪extend≫** UC2 (when imbalanced) |

### Subsystem 2: Feature Engineering

| UC ID | Use Case                      | Actor(s)       | Description                                              |
| ----- | ----------------------------- | -------------- | -------------------------------------------------------- |
| UC4   | Reduce Feature Dimensionality | Data Scientist | Reduce from 49 to 8 features                             |
| UC4a  | Apply PCA                     | Data Scientist | PCA reduction — **≪extend≫** UC4                         |
| UC4b  | Apply Autoencoder             | Data Scientist | Deterministic AE — **≪extend≫** UC4                      |
| UC4c  | Apply VAE Encoding            | Data Scientist | Probabilistic VAE (primary method) — **≪extend≫** UC4    |
| UC5   | Visualize Latent Space        | Data Scientist | Plot latent space distributions — **≪extend≫** UC4       |
| UC6   | Compare Reduction Methods     | Data Scientist | PCA vs AE vs VAE comparative analysis — **≪extend≫** UC4 |

### Subsystem 3: Classical Model Training

| UC ID | Use Case                        | Actor(s)       | Description                                       |
| ----- | ------------------------------- | -------------- | ------------------------------------------------- |
| UC7   | Train Classical Baseline Models | Data Scientist | Train all 3 classical baselines                   |
| UC7a  | Train SVM Classifier            | Data Scientist | SVM with RBF kernel — **≪include≫** from UC7      |
| UC7b  | Train Random Forest             | Data Scientist | 100 trees, Gini impurity — **≪include≫** from UC7 |
| UC7c  | Train XGBoost                   | Data Scientist | Gradient boosting — **≪include≫** from UC7        |
| UC8   | Tune Hyperparameters            | Data Scientist | GridSearchCV, 5-fold CV — **≪extend≫** UC7        |

### Subsystem 4: Quantum Processing

| UC ID | Use Case                 | Actor(s)              | Description                                                         |
| ----- | ------------------------ | --------------------- | ------------------------------------------------------------------- |
| UC9   | Design Quantum Circuit   | Data Scientist        | Build feature map + ansatz circuit                                  |
| UC10  | Train VQC Model          | Data Scientist        | Variational Quantum Classifier, COBYLA 100 iter — **≪include≫** UC9 |
| UC11  | Train QSVM Model         | Data Scientist        | Quantum kernel SVM — **≪include≫** UC9                              |
| UC12  | Select Quantum Backend   | SysAdmin              | Choose Aer simulator or real IBM hardware                           |
| UC13  | Configure Feature Map    | Data Scientist        | Set ZZ type, reps (2-4), entanglement pattern — **≪extend≫** UC9    |
| UC14  | Execute on Real Hardware | SysAdmin, IBM Quantum | Submit circuits to IBM Brisbane (127-qubit)                         |

### Subsystem 5: Hybrid Classification

| UC ID | Use Case                  | Actor(s)         | Description                                                    |
| ----- | ------------------------- | ---------------- | -------------------------------------------------------------- |
| UC15  | Configure Hybrid Ensemble | Data Scientist   | Set up quantum + classical weighted voting                     |
| UC16  | Optimize Ensemble Weights | Data Scientist   | Tune weights (0.5 quantum, 0.3 XGB, 0.2 RF) via validation set |
| UC17  | Classify Network Traffic  | Security Analyst | Binary detection: Normal vs Attack — **≪include≫** UC15        |
| UC18  | Identify Attack Type      | Security Analyst | Multi-class: DoS, Probe, R2L, U2R — **≪include≫** UC15         |

### Subsystem 6: Explainability (XAI)

| UC ID | Use Case                    | Actor(s)         | Description                                          |
| ----- | --------------------------- | ---------------- | ---------------------------------------------------- |
| UC19  | Generate SHAP Explanations  | Data Scientist   | Global feature importance via KernelExplainer        |
| UC20  | Generate LIME Explanations  | Data Scientist   | Local instance-level explanations                    |
| UC21  | Visualize Attack Signatures | Data Scientist   | Per-attack-type feature patterns — **≪extend≫** UC19 |
| UC22  | View Feature Contribution   | Security Analyst | See top features driving a detection decision        |

### Subsystem 7: Evaluation & Reporting

| UC ID | Use Case                        | Actor(s)       | Description                                                     |
| ----- | ------------------------------- | -------------- | --------------------------------------------------------------- |
| UC23  | Evaluate Model Performance      | Data Scientist | Accuracy, Precision, Recall, F1-Score, confusion matrix         |
| UC24  | Benchmark Quantum Advantage     | Data Scientist | Per-attack-type quantum vs classical comparison                 |
| UC25  | Run Cross-Validation            | Data Scientist | 10-fold CV for statistical robustness — **≪extend≫** UC23       |
| UC26  | Run Statistical Tests           | Data Scientist | t-tests, p<0.05 significance — **≪extend≫** UC23                |
| UC27  | Generate Performance Report     | Data Scientist | Tables, figures, publication-quality output — **≪extend≫** UC23 |
| UC28  | Export Results & Visualizations | Data Scientist | Save to files for paper/presentation                            |

### Subsystem 8: Threat Assessment (Output)

| UC ID | Use Case                          | Actor(s)         | Description                                    |
| ----- | --------------------------------- | ---------------- | ---------------------------------------------- |
| UC29  | View Threat Classification        | Security Analyst | See Normal/Attack + attack type label          |
| UC30  | View Confidence Score             | Security Analyst | See probabilistic prediction confidence        |
| UC31  | View Explanation for Alert        | Security Analyst | See SHAP/LIME explanation for a specific alert |
| UC32  | Analyze False Positives/Negatives | Security Analyst | Review misclassifications with explanations    |

---

## Relationship Summary

### ≪include≫ (mandatory sub-steps)

- UC2 → UC2a, UC2b, UC2c (preprocessing always runs all 3)
- UC7 → UC7a, UC7b, UC7c (baseline trains all 3 classifiers)
- UC10, UC11 → UC9 (quantum models always need circuit design)
- UC17, UC18 → UC15 (classification requires ensemble)

### ≪extend≫ (optional/conditional)

- UC3 → UC2 (only when class imbalance detected)
- UC4a, UC4b, UC4c → UC4 (user selects one reduction method)
- UC5, UC6 → UC4 (optional visualization/comparison)
- UC8 → UC7 (optional hyperparameter tuning)
- UC13 → UC9 (optional advanced circuit config)
- UC21 → UC19 (optional deep-dive into attack patterns)
- UC25, UC26, UC27 → UC23 (optional extended evaluation)

---

## Actor–Use Case Matrix

| Use Case Subsystem     | Security Analyst | Data Scientist | SysAdmin | IBM Quantum |
| ---------------------- | :--------------: | :------------: | :------: | :---------: |
| Data Management        |        -         |       ✓        |    ✓     |      -      |
| Feature Engineering    |        -         |       ✓        |    -     |      -      |
| Classical Training     |        -         |       ✓        |    -     |      -      |
| Quantum Processing     |        -         |       ✓        |    ✓     |      ✓      |
| Hybrid Classification  |        ✓         |       ✓        |    -     |      -      |
| Explainability (XAI)   |        ✓         |       ✓        |    -     |      -      |
| Evaluation & Reporting |        -         |       ✓        |    -     |      -      |
| Threat Assessment      |        ✓         |       -        |    -     |      -      |
