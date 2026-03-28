# Actor–Use Case Interaction Explanation

## Explainable Hybrid Quantum-Classical Network Intrusion Detection System

This document explains **how** each actor interacts with every use case, **why** they need to, and the **workflow context** behind each interaction.

---

## 1. Security Analyst

**Role:** The operational end user working in a Security Operations Center (SOC). They do not build or train models — they consume the system's output to protect the network.

**Why this actor exists:** The entire purpose of the NIDS is to serve security analysts. Without transparent, explainable output, analysts cannot trust automated quantum-enhanced decisions — especially when false positives waste resources and false negatives leave the network exposed.

---

### UC17 — Classify Network Traffic (Binary: Normal vs Attack)

**How:** The analyst inputs or streams network traffic data into the trained hybrid ensemble system. The system returns a binary label (Normal or Attack) for each traffic sample.

**Why:** This is the analyst's primary task — continuous monitoring. They need to know immediately if incoming network traffic is malicious. The binary mode provides a fast first-pass filter to separate suspicious traffic from benign traffic.

**Workflow:**

1. Analyst receives incoming network traffic data (or selects a batch)
2. System runs data through the preprocessing → VAE encoding → hybrid ensemble pipeline
3. Analyst sees a "Normal" or "Attack" label for each sample
4. If "Attack" is flagged, analyst proceeds to UC18 and UC31 for deeper investigation

---

### UC18 — Identify Attack Type (Multi-class)

**How:** When traffic is flagged as an attack, the analyst invokes multi-class classification. The system outputs a specific attack category (DoS, Probe, R2L, U2R for NSL-KDD; or one of 9 attack families for NB15).

**Why:** Knowing _what kind_ of attack is happening determines the response. A DoS attack requires traffic throttling or IP blocking; a Probe attack means someone is scanning for vulnerabilities (triggering a security audit); R2L/U2R attacks demand immediate access revocation. Without attack type identification, the analyst cannot respond appropriately.

**Workflow:**

1. Analyst selects flagged attack samples from UC17
2. System classifies into specific attack categories using One-vs-Rest strategy
3. Analyst reads the attack type label + confidence score
4. Analyst initiates the correct incident response procedure based on attack type

---

### UC22 — View Feature Contribution for Detection Decision

**How:** For any specific detection (from UC17 or UC18), the analyst clicks on the prediction to see which network traffic features contributed most to the decision. This is presented as a ranked list of features with their contribution weights (e.g., "packet_rate: 42% contribution, login_failures: 28%").

**Why:** This is the core **trust mechanism**. Security analysts will not act on a "black box" quantum model's output. If the system says "this is a DoS attack" but the analyst can see that the top contributing features are packet_rate, byte_count, and connection_duration — all consistent with known DoS patterns — they trust the alert and act on it. If the features don't make sense, they might flag it as a potential false positive.

**Workflow:**

1. Analyst sees an alert from UC17/UC18
2. Analyst opens the explanation view for that specific alert
3. System displays a feature contribution chart (from SHAP/LIME)
4. Analyst validates whether the top features align with known attack signatures
5. Analyst either escalates the alert or marks it as a false positive

---

### UC29 — View Threat Classification

**How:** The analyst accesses the threat assessment dashboard showing all recent classifications with their labels.

**Why:** Provides a summary view of the security posture. The analyst can quickly scan for patterns — e.g., "there have been 15 DoS attack detections in the last hour from the same subnet" — which might indicate a coordinated attack.

**Workflow:**

1. Analyst opens the threat dashboard
2. Sees a table/list of recent traffic samples with classification labels
3. Filters by attack type, time range, or source IP
4. Identifies patterns or clusters of attacks needing investigation

---

### UC30 — View Confidence Score

**How:** Each prediction comes with a probabilistic confidence score (e.g., "Attack — 94.3% confidence" or "Normal — 67.2% confidence").

**Why:** Not all predictions are equally reliable. A high-confidence attack alert (>90%) demands immediate action. A low-confidence alert (60-70%) might be a borderline case worth manual investigation before escalating. This prevents the analyst from over-reacting to uncertain predictions or ignoring genuinely dangerous low-confidence alerts that might be novel attack patterns.

**Workflow:**

1. Analyst reviews classification output from UC29
2. Checks the confidence score for each prediction
3. High confidence (>90%): acts on the alert immediately
4. Medium confidence (70-90%): reviews the explanation (UC31) before deciding
5. Low confidence (<70%): flags for manual review, checks similar historical samples

---

### UC31 — View Explanation for Alert

**How:** The analyst selects a specific alert and views the full SHAP/LIME explanation — including a feature importance bar chart, individual feature values, and how they compare to the training distribution.

**Why:** This is the detailed investigation mode. When an alert seems suspicious (either a potential false positive or a novel attack), the analyst needs to understand _exactly_ why the system made this decision. For example, if the system flags traffic as R2L but the explanation shows the primary feature is "dst_bytes" (unusual for R2L), the analyst might suspect a misclassification.

**Workflow:**

1. Analyst selects a specific alert from UC29 or UC30
2. System shows a LIME local explanation for that individual instance
3. Analyst reads: "Top 3 reasons for this classification: (1) src_bytes=0 [strongly indicates attack], (2) protocol=TCP [neutral], (3) login_failed=5 [strongly indicates R2L]"
4. Analyst correlates with their domain knowledge
5. Analyst decides: escalate, dismiss, or investigate further

---

### UC32 — Analyze False Positives/Negatives

**How:** The analyst reviews cases where the system's prediction was later confirmed to be wrong — either normal traffic flagged as an attack (false positive) or actual attacks missed (false negative). For each misclassification, the explanation from UC31 is available.

**Why:** False positive analysis helps refine the system over time. If the analyst notices the system consistently produces false positives for a certain type of legitimate traffic (e.g., high-volume video streaming flagged as DoS), they can report this to the data scientist to retrain the model. False negative analysis is even more critical — missed attacks reveal blind spots in the detection system that need urgent addressing.

**Workflow:**

1. Analyst opens the misclassification review panel
2. Filters by false positive or false negative
3. For each case, reviews the explanation (via UC31)
4. Identifies common patterns in misclassifications
5. Submits feedback to the data scientist for model improvement
6. Documents findings for security incident reports

---

---

## 2. Data Scientist

**Role:** The researcher/developer who builds, trains, and evaluates the entire hybrid quantum-classical pipeline. They work with datasets, engineer features, design quantum circuits, and measure model performance.

**Why this actor exists:** The system requires expert ML/quantum engineering to build. The data scientist makes all technical decisions — which feature reduction method to use, how to configure quantum circuits, how to weight the ensemble, and how to validate results for publication.

---

### UC1 — Load Benchmark Dataset

**How:** The data scientist uses the `DatasetManager` class to load one of the three benchmark datasets (NSL-KDD ~150MB, UNSW-NB15 ~2GB, CICIDS2017 ~8GB) from local storage or downloads.

**Why:** Each dataset serves a specific research purpose:

- **NSL-KDD**: Legacy benchmark for comparison with prior work (baseline 92% QSVM accuracy to replicate)
- **UNSW-NB15**: The most challenging dataset (prior quantum work achieved only 64% — the main target for improvement to ≥80%)
- **CICIDS2017**: Modern attack dataset to prove generalization beyond legacy benchmarks (novel contribution)

**Workflow:**

1. Data scientist selects a dataset by name
2. `DatasetManager.loadDataset("UNSW-NB15")` loads the raw CSV/ARFF files
3. Splits into train/test (80/20 with stratification)
4. Verifies class distributions and sample counts

---

### UC2 — Preprocess Network Traffic Data

**How:** The data scientist runs the full preprocessing pipeline, which automatically triggers UC2a (normalization), UC2b (encoding), and UC2c (imputation) as ≪include≫ steps.

**Why:** Raw network traffic data cannot be fed to quantum circuits. Quantum gates operate on normalized values in [0, 2π]. Categorical protocol fields (TCP, UDP, ICMP) must be converted to numerical form. Missing values would crash the quantum circuit execution.

**Workflow:**

1. Data scientist calls `DataPreprocessor.buildPipeline()`
2. **UC2a** runs: Min-Max normalizes all numerical features to [0, 1]
3. **UC2b** runs: One-Hot encodes protocol_type, service, flag columns
4. **UC2c** runs: Imputes missing values (median for numerical, mode for categorical)
5. Output: clean, normalized DataFrame ready for feature reduction

---

### UC3 — Balance Class Distribution

**How:** The data scientist checks class balance ratios and applies SMOTE (Synthetic Minority Over-sampling Technique) when the minority class (e.g., R2L, U2R attacks) is underrepresented.

**Why:** NSL-KDD has severe class imbalance — U2R attacks are only ~0.1% of the dataset. Without balancing, the model would achieve high accuracy by simply predicting "Normal" for everything and completely missing rare-but-dangerous attacks. SMOTE generates synthetic minority samples to create a more balanced training set.

**Workflow:**

1. Data scientist checks: if minority class < 10% of dataset → trigger UC3
2. `ClassBalancer.applySMOTE(X_train, y_train)` generates synthetic samples
3. Verifies new class distribution is more balanced
4. Uses balanced data for training only (test set remains unbalanced for realistic evaluation)

---

### UC4, UC4a, UC4b, UC4c — Reduce Feature Dimensionality

**How:** The data scientist selects a dimensionality reduction method (PCA, Autoencoder, or VAE) to compress the original 49 features into 8 features suitable for the 8-qubit quantum circuit.

**Why:** Quantum computers have limited qubits. The VQC uses 8 qubits, so the input must be exactly 8 features. Additionally, the base paper's poor NB15 performance (64%) was caused by naive PCA creating distribution mismatches. The VAE (UC4c) solves this with probabilistic encoding that preserves feature distributions — this is the project's key innovation.

**Workflow for VAE (UC4c — primary method):**

1. Data scientist initializes `VariationalAutoencoder(latent_dim=8)`
2. Encoder: 49 → 128 → 64 → 8 (latent mean + logvar)
3. Trains with KL divergence + reconstruction loss
4. Validates reconstruction loss < 0.05
5. Transforms all data: `vae.transform(X)` → 8-dimensional latent representation

**Why all three options (UC4a, UC4b, UC4c)?** The paper must show comparative results — "VAE outperforms PCA by X% and Autoencoder by Y% on NB15" to justify the VAE choice.

---

### UC5 — Visualize Latent Space

**How:** The data scientist creates 2D/3D scatter plots of the 8-dimensional encodings (using t-SNE or UMAP) colored by attack class.

**Why:** Visual evidence that the VAE creates separable clusters for different attack types — this goes into the paper as a figure proving the encoding quality. If Normal and Attack samples overlap heavily in latent space, the quantum classifier will struggle.

---

### UC6 — Compare Reduction Methods

**How:** The data scientist trains the VQC with encodings from PCA, Autoencoder, and VAE separately, then compares accuracy on each dataset.

**Why:** To produce the comparison table in the paper (e.g., "VQC+PCA: 72% → VQC+AE: 76% → VQC+VAE: 82% on NB15") proving VAE is the best choice.

---

### UC7, UC7a, UC7b, UC7c — Train Classical Baseline Models

**How:** The data scientist trains all three classical models (SVM, Random Forest, XGBoost) on the same preprocessed data used for quantum models.

**Why:** Classical baselines are essential for two reasons:

1. **Scientific comparison**: The paper must show quantum vs classical performance side-by-side
2. **Ensemble components**: RF and XGBoost become part of the hybrid ensemble (UC15), combined with the quantum VQC via weighted voting

**Workflow:**

1. Train SVM (RBF kernel, C=1.0, gamma='scale')
2. Train RF (n_estimators=100, criterion='gini')
3. Train XGBoost (learning_rate=0.1, max_depth=6)
4. Record accuracy, F1, training time for each
5. Results go into the baseline comparison table

---

### UC8 — Tune Hyperparameters

**How:** The data scientist runs GridSearchCV with 5-fold cross-validation to find optimal hyperparameters for classical models.

**Why:** Ensures the classical baselines are at their best performance — if you compare quantum against poorly-tuned classical models, reviewers will reject the paper. Fair comparison requires optimized baselines.

---

### UC9 — Design Quantum Circuit

**How:** The data scientist uses `QuantumCircuitBuilder` to create the complete quantum circuit: feature map (data encoding) + ansatz (trainable parameters).

**Why:** The circuit design directly determines model capacity. A shallow circuit (reps=1) might underfit; a deep circuit (reps=4) might overfit or be too slow. The data scientist must find the right balance through experimentation.

---

### UC10 — Train VQC Model

**How:** The data scientist creates a VQC with ZZFeatureMap (reps=2) + RealAmplitudes ansatz (reps=3), trains it using COBYLA optimizer for 100 iterations.

**Why:** The VQC is the **core innovation** of the project. Unlike the base paper's static QSVM kernel, the VQC has trainable parameters that adapt to the specific data distribution — this is what enables the 16% accuracy improvement on NB15.

**Workflow:**

1. Build feature map: `ZZFeatureMap(feature_dimension=8, reps=2)`
2. Build ansatz: `RealAmplitudes(num_qubits=8, reps=3)`
3. Create VQC: `VQC(feature_map, ansatz, optimizer=COBYLA(maxiter=100))`
4. Train: `vqc.train(X_train_encoded, y_train)`
5. Monitor optimization loss curve
6. Evaluate on test set

---

### UC11 — Train QSVM Model

**How:** The data scientist trains a Quantum SVM using quantum kernel estimation — the method from the base paper (Gouveia & Correia, 2020).

**Why:** This is the **replication** step. The data scientist must reproduce the base paper's results (92% on NSL-KDD, 64% on NB15) before showing improvement. Without replication, the paper's claims of improvement are unsubstantiated.

---

### UC13 — Configure Feature Map

**How:** The data scientist experiments with different feature map configurations: ZZ vs Z feature maps, repetition depths (2, 3, 4), and entanglement patterns (linear, circular, full).

**Why:** Different configurations affect which feature interactions the circuit can capture:

- **Linear entanglement**: Each qubit entangled only with neighbor — good for sequential features
- **Circular entanglement**: Ring topology — captures wrap-around correlations
- **Full entanglement**: All-to-all — maximum expressiveness but deepest circuit (slower)
- **Reps**: More repetitions = deeper circuit = more parameters = more expressive but harder to train

---

### UC15 — Configure Hybrid Ensemble

**How:** The data scientist creates the `HybridEnsemble` object, registers the trained VQC + RF + XGBoost models, and sets initial weights (`0.5*quantum + 0.3*xgb + 0.2*rf`).

**Why:** No single model is best for all attack types. The quantum VQC may excel at detecting complex non-linear patterns (e.g., R2L, U2R) while classical XGBoost handles straightforward DoS attacks better. The ensemble combines their strengths.

---

### UC16 — Optimize Ensemble Weights

**How:** The data scientist uses a validation set to find the optimal weight combination that maximizes overall F1-score.

**Why:** The initial weights (0.5/0.3/0.2) are educated guesses. Optimization might reveal that for NB15, the best weights are (0.4/0.4/0.2) — giving more weight to XGBoost because NB15's attack patterns are more amenable to gradient boosting.

---

### UC19 — Generate SHAP Explanations

**How:** The data scientist creates a `KernelExplainer` with 100 background samples, then computes SHAP values for the entire test set.

**Why:** SHAP provides **global** feature importance — which features matter most across all predictions. This addresses the "black box" criticism of quantum models. The paper's explainability contribution requires SHAP analysis showing insights like "DoS attacks are 85% driven by packet_rate features."

**Workflow:**

1. `explainer = KernelExplainer(ensemble.predict, X_background)`
2. `shap_values = explainer.shap_values(X_test)`
3. Generate summary plot: global feature ranking
4. Generate dependence plots: how individual features affect predictions
5. Analyze per-attack-type feature patterns

---

### UC20 — Generate LIME Explanations

**How:** The data scientist creates a `LimeTabularExplainer` and generates individual instance explanations.

**Why:** LIME complements SHAP by providing **local** explanations — why _this specific_ traffic sample was classified as an attack. This is what security analysts see in UC31. While SHAP shows "generally, feature X matters most," LIME shows "for this particular alert, feature X=value is the reason."

---

### UC21 — Visualize Attack Signatures

**How:** The data scientist aggregates SHAP values by attack type and creates per-class feature importance plots.

**Why:** This produces the paper's key insights — "DoS signature: high packet_rate + high src_bytes" or "R2L signature: multiple login failures + low connection duration." These attack signature visualizations are a core contribution.

---

### UC23 — Evaluate Model Performance

**How:** The data scientist computes accuracy, precision, recall, F1-score, and confusion matrix for every model on every dataset.

**Why:** These metrics are the paper's primary results. The key target: UNSW-NB15 accuracy ≥80% (vs 64% baseline).

---

### UC24 — Benchmark Quantum Advantage

**How:** For each attack type, the data scientist compares quantum VQC performance vs classical models to identify where quantum processing provides measurable gains.

**Why:** This is a secondary objective — identifying _specific_ attack types where quantum helps. The paper might find "quantum VQC improves U2R detection by 12% over XGBoost due to better handling of non-linear boundary patterns."

**Workflow:**

1. For each attack type (DoS, Probe, R2L, U2R):
2. Compute classical_accuracy = XGBoost on that attack type
3. Compute quantum_accuracy = VQC on that attack type
4. Quantum advantage = quantum_accuracy - classical_accuracy
5. Report which types benefit from quantum processing

---

### UC25 — Run Cross-Validation

**How:** The data scientist runs 10-fold cross-validation on the hybrid ensemble.

**Why:** Single train/test splits can be misleading. 10-fold CV provides mean ± standard deviation, proving results are statistically robust. Reviewers will reject papers without CV.

---

### UC26 — Run Statistical Tests

**How:** The data scientist performs paired t-tests comparing quantum vs classical results across folds.

**Why:** "Our method gets 82% vs baseline 75%" is not convincing without statistical significance. The t-test proves the improvement is not due to random chance (p < 0.05).

---

### UC27 — Generate Performance Report

**How:** The data scientist auto-generates publication-quality tables and figures (300 DPI) from all collected metrics.

**Why:** The paper needs standardized comparison tables across all datasets and methods. Manual formatting is error-prone.

---

### UC28 — Export Results & Visualizations

**How:** The data scientist saves all results (CSVs, figures, model checkpoints) to the project directory.

**Why:** For reproducibility (GitHub repo), paper compilation (LaTeX), and presentation slides.

---

---

## 3. System Administrator

**Role:** The infrastructure manager who handles computational resources, dataset storage, and quantum backend configuration.

**Why this actor exists:** Quantum computing requires specialized infrastructure management — API keys, backend selection, queue management, and noise characterization — that goes beyond typical ML operations.

---

### UC1 — Load Benchmark Dataset

**How:** The SysAdmin downloads and stores the benchmark datasets on the server/workstation. UNSW-NB15 (~2GB) and CICIDS2017 (~8GB) require significant storage. The SysAdmin ensures proper file permissions and directory structure.

**Why:** Datasets must be available locally — downloading 8GB during experiments wastes time. The SysAdmin also verifies academic license compliance for CICIDS2017.

---

### UC12 — Select Quantum Backend

**How:** The SysAdmin configures whether the system uses:

- **Qiskit Aer simulator** (local, no queue, deterministic) — for development and primary experiments
- **IBM Quantum real hardware** (cloud, queue times, noisy) — for real hardware validation

**Why:** Simulator vs real hardware is a infrastructure decision. The simulator is used for all development and primary results. Real hardware is optional (Week 14-15 in the roadmap) and depends on queue times (often 2+ weeks). The SysAdmin manages IBM Quantum API credentials and monitors queue status.

**Workflow:**

1. SysAdmin loads IBM Quantum account: `IBMQ.load_account()`
2. Selects provider: `IBMQ.get_provider(hub='ibm-q')`
3. Checks available backends and their queue lengths
4. Configures: `backend = provider.get_backend('ibm_brisbane')` or `Aer.get_backend('aer_simulator')`
5. Communicates backend availability to the data scientist

---

### UC14 — Execute on Real Hardware

**How:** The SysAdmin submits quantum circuits to IBM Brisbane (127-qubit system), monitors job status, and retrieves results.

**Why:** Real hardware validation is important for the paper's credibility, but it introduces noise (expect 5-15% accuracy drop). The SysAdmin handles:

- Job submission and queue management
- Shot count configuration (1024 shots per circuit)
- Error mitigation strategy selection
- Result retrieval and noise characterization

**Workflow:**

1. SysAdmin verifies backend is online and calibrated
2. Submits circuit jobs with `backend.run(circuit, shots=1024)`
3. Monitors job queue position and estimated wait time
4. Downloads results when jobs complete
5. Provides raw hardware results to data scientist for analysis

---

---

## 4. IBM Quantum Cloud (External System)

**Role:** The external quantum computing infrastructure provided by IBM. This is not a human actor — it's a system actor representing the real quantum hardware.

**Why this actor exists:** The system interacts with external quantum hardware for optional real-device validation. The quantum cloud has its own constraints (queue times, noise, limited qubit connectivity) that affect system behavior.

---

### UC14 — Execute on Real Hardware

**How:** The IBM Quantum Cloud receives quantum circuit jobs, queues them, executes them on the 127-qubit IBM Brisbane processor, and returns measurement results.

**Why:** Real hardware execution proves the system works beyond simulation. However, NISQ (Noisy Intermediate-Scale Quantum) devices introduce decoherence and gate errors, so results will differ from simulator results. This validates the system's practical feasibility and provides data for discussing NISQ limitations in the paper.

**Interaction details:**

- **Input**: Compiled quantum circuits (transpiled for IBM Brisbane's native gate set)
- **Processing**: Execute circuits with 1024 measurement shots each
- **Output**: Bit-string measurement counts (e.g., {"00000001": 512, "00000000": 489, ...})
- **Constraints**: Queue times (hours to weeks), noise (~0.1% per gate), limited qubit connectivity (heavy-hex lattice)

---

---

## Actor Interaction Flow (End-to-End Scenario)

Here is a complete scenario showing how all actors interact for a full system run:

```
1. SysAdmin (UC1)  → Downloads UNSW-NB15 dataset, sets up storage
2. SysAdmin (UC12) → Configures Aer simulator as the quantum backend
3. Data Sci (UC1)  → Loads UNSW-NB15 into DatasetManager
4. Data Sci (UC2)  → Preprocesses: normalize (UC2a) + encode (UC2b) + impute (UC2c)
5. Data Sci (UC3)  → Applies SMOTE for class balancing
6. Data Sci (UC4c) → Applies VAE encoding (49 → 8 features)
7. Data Sci (UC5)  → Visualizes latent space to verify cluster separation
8. Data Sci (UC7)  → Trains classical baselines: SVM (UC7a) + RF (UC7b) + XGBoost (UC7c)
9. Data Sci (UC8)  → Tunes hyperparameters via GridSearchCV
10. Data Sci (UC9)  → Designs quantum circuit (ZZFeatureMap + RealAmplitudes)
11. Data Sci (UC10) → Trains VQC with COBYLA optimizer (100 iterations)
12. Data Sci (UC15) → Configures hybrid ensemble (VQC + RF + XGBoost)
13. Data Sci (UC16) → Optimizes ensemble weights on validation set
14. Data Sci (UC23) → Evaluates: accuracy, F1, precision, recall
15. Data Sci (UC25) → Runs 10-fold cross-validation
16. Data Sci (UC26) → Runs t-tests for statistical significance
17. Data Sci (UC19) → Generates SHAP global explanations
18. Data Sci (UC20) → Generates LIME local explanations
19. Data Sci (UC21) → Visualizes per-attack-type feature signatures
20. Data Sci (UC24) → Benchmarks quantum advantage per attack type
21. Data Sci (UC27) → Generates publication-quality report
22. Data Sci (UC28) → Exports all results and figures
--- System deployed for analyst use ---
23. Analyst (UC17)  → Classifies incoming network traffic (Normal/Attack)
24. Analyst (UC18)  → Identifies attack type for flagged traffic
25. Analyst (UC29)  → Views threat classification dashboard
26. Analyst (UC30)  → Checks confidence scores
27. Analyst (UC31)  → Reviews SHAP/LIME explanation for a specific alert
28. Analyst (UC22)  → Validates feature contributions against domain knowledge
29. Analyst (UC32)  → Reviews false positives/negatives for model feedback
--- Optional: Real hardware validation ---
30. SysAdmin (UC12) → Switches backend to IBM Brisbane
31. SysAdmin (UC14) → Submits circuits to IBM Quantum Cloud
32. IBM QC  (UC14)  → Executes circuits, returns noisy results
33. Data Sci (UC24) → Compares simulator vs real hardware performance
```

---

## Summary: Why Each Actor Needs Their Use Cases

| Actor                    | Core Need                         | Use Cases                            | Key Motivation                                                                                                 |
| ------------------------ | --------------------------------- | ------------------------------------ | -------------------------------------------------------------------------------------------------------------- |
| **Security Analyst**     | Trust automated quantum decisions | UC17, UC18, UC22, UC29-32            | Cannot act on opaque "black box" alerts — needs **explainability** to validate decisions and respond correctly |
| **Data Scientist**       | Build and validate the system     | UC1-UC11, UC13, UC15-UC16, UC19-UC28 | Must engineer the entire pipeline, prove quantum advantage ≥80% on NB15, and produce publication-ready results |
| **System Administrator** | Manage infrastructure             | UC1, UC12, UC14                      | Quantum backends require specialized management (API keys, queues, noise) beyond standard ML infrastructure    |
| **IBM Quantum Cloud**    | Provide real hardware             | UC14                                 | NISQ device execution validates practical feasibility — proves the system works beyond simulation              |
