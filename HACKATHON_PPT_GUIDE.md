# HACKATHON PPT GUIDE — 6 Slides

## Track: **QML – Quantum Machine Learning**

> Your project is a perfect fit for the QML track. It uses Variational Quantum Circuits (VQC) for machine learning-based Network Intrusion Detection.

---

## SCORING STRATEGY

| Criteria            | Weight | Targeted On Slides                      | Strategy                                                                    |
| ------------------- | ------ | --------------------------------------- | --------------------------------------------------------------------------- |
| **Problem Clarity** | 30%    | Slide 2 (primary), Slide 1 (hook)       | Lead with a real-world pain point + quantify the gap with hard numbers      |
| **Innovation**      | 30%    | Slide 3 (primary), Slide 4 (supporting) | Show 3 distinct novel contributions; use "First to…" language               |
| **Feasibility**     | 20%    | Slide 5 (primary), Slide 4 (supporting) | Show concrete expected results tables, open-source stack, realistic targets |
| **Technical Plan**  | 20%    | Slide 4 (primary), Slide 3 (supporting) | Architecture diagram + tech stack logos + clear pipeline flow               |

---

---

## SLIDE 1 — TITLE SLIDE

### Purpose: Hook the judges in 5 seconds

### Content Layout:

**Top Section:**

- Project Title: **"Explainable Hybrid Quantum-Classical Network Intrusion Detection System"**
- Subtitle: _"Using Variational Quantum Circuits for Transparent Cybersecurity"_

**Middle Section:**

- **Track Badge:** QML – Quantum Machine Learning
- **One-liner hook (large font, bold):**
  > "Quantum-powered threat detection that security analysts can actually trust."

**Bottom Section:**

- Team name & member names
- Institution name & logo

### Design Tips:

- Dark theme (navy/black background) with a quantum-circuit motif or cyber grid
- Use a subtle background image of a quantum circuit or network topology
- Keep text minimal — this slide is about impact, not detail

---

---

## SLIDE 2 — PROBLEM STATEMENT

### Purpose: Score maximum on **Problem Clarity (30%)**

### Visual Layout: Four themed sections — one per guiding question

---

#### SECTION 1 — "What Real-World Problem Are We Solving?" 🔴

**Headline:** _"Network Intrusion Detection Is Broken"_

- Cyberattacks are growing in **volume, speed, and sophistication** — traditional signature-based detection can't keep up
- Machine Learning-based Intrusion Detection Systems (ML-IDS) were supposed to be the answer, but they introduce **three critical failures:**
  1. **Black-Box Decisions** — Security analysts cannot see _why_ a threat was flagged, leading to a dangerous trust gap
  2. **High-Dimensional Bottleneck** — Modern network traffic has 49–78 features; real-time classification at scale is computationally infeasible
  3. **Unacceptable Accuracy on Modern Data** — The best existing quantum-enhanced IDS (Gouveia & Correia, 2020) achieves only **64% accuracy** on the UNSW-NB15 dataset — meaning **1 in 3 attacks goes undetected**

---

#### SECTION 2 — "Why Does This Problem Matter in the Quantum Era?" ⚛️

**Headline:** _"Quantum Computing Changes the Threat Landscape — and the Solution Space"_

- **Quantum threats are emerging:** Future quantum computers will break classical encryption (RSA, ECC), making proactive intrusion detection even more critical
- **Quantum advantage for ML:** Variational Quantum Circuits can explore exponentially large feature spaces — capturing non-linear attack correlations that classical kernels miss
- **But current quantum IDS efforts have failed:** Existing approaches use **static quantum kernels** with no trainable parameters and provide **zero explainability** — making them unusable in real security operations
- **The gap is clear:** We need quantum-enhanced detection that is both _accurate_ and _transparent_ before the quantum threat era fully arrives

---

#### SECTION 3 — "Who Is Affected?" 👥

**Headline:** _"The Impact Spans Every Sector"_

| Stakeholder                   | How They're Affected                                                                                                                                  |
| ----------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| **🏢 Industry / Enterprises** | SOC teams waste hours investigating false positives from black-box IDS; missed attacks lead to data breaches costing **$4.45M on average** (IBM 2023) |
| **🏛️ Government & Defense**   | Critical infrastructure (power grids, defense networks) faces nation-state cyber threats requiring next-gen detection beyond classical capabilities   |
| **🎓 Research & Academia**    | No reproducible, open-source quantum IDS benchmark exists — slowing progress in quantum cybersecurity research                                        |
| **👤 End Users & Society**    | Personal data, healthcare records, and financial information are exposed when IDS systems fail silently                                               |

---

#### SECTION 4 — "What Are the Limitations of Current Classical Solutions?" ⚠️

**Headline:** _"Why Classical ML Alone Is Not Enough"_

| Limitation                       | Detail                                                                                                                                 |
| -------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| **Opaque Decision-Making**       | Classical ML models (Deep Neural Nets, SVMs) act as black boxes — no feature-level explanation for alerts                              |
| **Accuracy Plateau**             | Classical models plateau at ~80% on complex datasets like UNSW-NB15; rare attack types (R2L, U2R) are consistently missed              |
| **Dimensionality Curse**         | 49–78 raw features require aggressive reduction (PCA), which **destroys distributional properties** needed for subtle attack detection |
| **No Quantum Readiness**         | Classical-only systems cannot leverage quantum feature spaces — missing non-linear correlations that quantum circuits can capture      |
| **Zero Explainability Standard** | No existing quantum or hybrid IDS integrates explainability (SHAP/LIME) — leaving a critical trust gap in security operations          |

---

#### RESEARCH GAP SUMMARY (Visual callout box — red/orange border)

> **Research Gap:** Existing quantum IDS implementations use **static quantum kernels** (no learning) and provide **zero explainability** — a critical failure where false positives cost money and false negatives cost data breaches.

| Limitation         | Current State  | Our Target   |
| ------------------ | -------------- | ------------ |
| NB15 Accuracy      | 64% ❌         | ≥80% ✅      |
| Explainability     | None ❌        | SHAP+LIME ✅ |
| Trainable Circuits | No (static) ❌ | Yes (VQC) ✅ |

---

### IMAGE TO USE: None needed (text-driven slide). Optional: icons next to each section header for visual variety.

### Pro Tips for This Slide:

- The **four-question structure** maps directly to what judges expect — use it as your verbal flow during the pitch
- The 64% → 80% gap is your **strongest hook**. Make the "64%" number LARGE and RED.
- Use the phrase _"1 in 3 attacks missed"_ — judges remember human-readable stats
- The "$4.45M average breach cost" stat adds real-world weight — cite IBM Cost of a Data Breach 2023
- This slide should take ~75 seconds to present. Spend ~20 sec per section.

---

---

## SLIDE 3 — PROPOSED SOLUTION (Innovation Slide)

### Purpose: Score maximum on **Innovation (30%)**

### Visual Layout: Four concise sections + workflow diagram at bottom. Keep text minimal — phrases only, no paragraphs.

---

#### 💡 What Is Our Solution?

**Headline:** _"Explainable Hybrid Quantum-Classical IDS"_

- **Trainable VQC** — Quantum circuit _learns_ attack patterns via gradient optimization (not static kernels)
- **VAE Encoding** — Compresses 49 features → 8 qubits, preserving distributions
- **Hybrid Ensemble** — 0.5×Quantum + 0.3×XGBoost + 0.2×Random Forest
- **XAI Layer** — SHAP (global) + LIME (local) for full transparency

---

#### ⚛️ Why Quantum?

- VQC maps data into $2^n$-dimensional Hilbert space — exponentially richer than classical feature maps
- Entanglement captures **inter-feature dependencies** classical models treat independently
- Trainable θ parameters **adapt** to each attack type — static kernels cannot

> Classical models plateau at ~80% on NB15. Quantum accesses the feature space they can't.

---

#### 📈 How Does It Improve Over Classical?

| Metric             | Before        | Ours          | Gain               |
| ------------------ | ------------- | ------------- | ------------------ |
| NB15 Accuracy      | 64% (quantum) | **≥80%**      | **+25% relative**  |
| Rare Attacks (R2L) | Missed        | **Detected**  | +10-12%            |
| Explainability     | None          | **SHAP+LIME** | **New capability** |
| Feature Encoding   | PCA (lossy)   | **VAE**       | Fixes 28% drop     |

**"First to…"** claims:

1. ✅ First quantum IDS with SHAP + LIME explainability
2. ✅ First to use VAE encoding for quantum distribution fix
3. ✅ First quantum IDS benchmarked on 3 datasets

---

#### 🏗️ Architecture Overview

```
Network Traffic → Preprocessing + VAE (49→8) → VQC (8-qubit) → Hybrid Ensemble → XAI Output
```

**Optional:** Place a small `Images/Arc_diag.png` preview here pointing to Slide 4.

---

### Pro Tips for This Slide:

- Each section = **3-4 lines max** on the actual PPT slide
- Say the "First to…" claims out loud — judges remember firsts
- The workflow arrow is the visual anchor — keep it prominent
- ~75 sec delivery: ~20 sec per section

---

---

## SLIDE 4 — IMPLEMENTATION PLAN & TECH STACK

### Purpose: Score maximum on **Technical Plan (20%)** + prove Feasibility. Judges must see this is practical and executable.

### Visual Layout: Four compact sections. Architecture diagram at top, rest as concise cards/tables.

---

#### 🛠️ How Will We Implement This?

**Architecture Diagram:** Place `Images/Arc_diag.png` prominently (50% of slide)

**Pipeline in one line:**

```
Raw Data → Preprocessing + VAE (49→8) → VQC (8-qubit) → Hybrid Ensemble → SHAP/LIME Output
```

**36-Hour Hackathon Roadmap:**

| Phase                     | Hours | Deliverable                                              |
| ------------------------- | ----- | -------------------------------------------------------- |
| **Data + Preprocessing**  | 0–8   | Load NSL-KDD/NB15, normalize, SMOTE, VAE encoder trained |
| **Quantum Pipeline**      | 8–20  | VQC circuit built & trained on Qiskit Aer simulator      |
| **Hybrid Ensemble + XAI** | 20–30 | XGBoost/RF integration, SHAP/LIME explanations working   |
| **Demo + Polish**         | 30–36 | End-to-end demo, results table, slides finalized         |

---

#### 🧰 Tools & Platforms

| Tool                          | Role                                                 |
| ----------------------------- | ---------------------------------------------------- |
| **Qiskit + Aer**              | VQC design & quantum simulation                      |
| **scikit-learn / XGBoost**    | Classical ML ensemble                                |
| **TensorFlow**                | VAE encoder training                                 |
| **SHAP + LIME**               | Explainability layer                                 |
| **IBM Quantum**               | Optional real-device validation (127-qubit Brisbane) |
| **NSL-KDD, NB15, CICIDS2017** | Benchmark datasets (free, public)                    |

> All tools are **free & open-source** — zero licensing barriers.

---

#### ⚠️ Challenges & Mitigation

| Challenge                   | Mitigation                                               |
| --------------------------- | -------------------------------------------------------- |
| Quantum simulation speed    | Use Aer simulator (fast); real hardware is optional demo |
| NB15 distribution mismatch  | VAE encoding preserves distributions (unlike PCA)        |
| Class imbalance in datasets | SMOTE oversampling + stratified splits                   |
| Limited 36-hour window      | Modular design — each member owns one pipeline stage     |

---

### Pro Tips for This Slide:

- The **36-hour roadmap** is the key differentiator — it proves you've planned execution, not just theory
- Architecture diagram should take ~50% of the slide; tables below it
- Say: _"Everything we use is free — Qiskit, public datasets, IBM Quantum free tier"_
- ~60 sec delivery: point at diagram, walk through roadmap phases

---

---

## SLIDE 5 — IMPACT & FUTURE VISION

### Purpose: Show why this idea matters **beyond the hackathon**. Judges must see real-world value + long-term potential.

### Visual Layout: Four compact sections — short phrases, no paragraphs.

---

#### 👥 Who Benefits?

| Beneficiary                 | Value Delivered                                                     |
| --------------------------- | ------------------------------------------------------------------- |
| **🔐 Cybersecurity Teams**  | Transparent alerts they can trust — SHAP explains _why_ each flag   |
| **🏢 Enterprises / SOCs**   | Fewer false positives → saves investigation hours & breach costs    |
| **🏛️ Government / Defense** | Next-gen quantum-ready IDS for critical infrastructure protection   |
| **🎓 Researchers**          | First open-source quantum IDS benchmark — reproducible & extensible |

---

#### 🏭 Industry Relevance

- **Cybersecurity** — Direct application: quantum-enhanced threat detection for enterprise networks
- **Finance** — Fraud & anomaly detection on high-dimensional transaction data using same VQC pipeline
- **Healthcare** — Protect patient records with explainable intrusion alerts (HIPAA compliance needs transparency)
- **AI / Quantum Computing** — Demonstrates practical QML beyond toy benchmarks — real datasets, real results

---

#### 💰 Research & Commercialization Potential

- **Publishable research** — Novel contributions (VQC + VAE + XAI) suitable for IEEE/ACM security conferences
- **Open-source framework** — GitHub repo becomes a community benchmark for quantum cybersecurity
- **Commercial path** — Explainable quantum IDS module can integrate into existing SIEM/SOC platforms (Splunk, QRadar)
- **Patent opportunity** — VAE-to-quantum encoding pipeline is a novel technique with IP potential

---

#### 🚀 Long-Term Scalability in the Quantum Ecosystem

| Horizon          | Scaling Opportunity                                                                         |
| ---------------- | ------------------------------------------------------------------------------------------- |
| **Near (1 yr)**  | Scale to 16–32 qubits as IBM hardware improves → richer feature encoding                    |
| **Mid (2–3 yr)** | Real-time quantum IDS on fault-tolerant hardware → production deployment                    |
| **Long (5+ yr)** | Quantum-native security platform — IDS, fraud detection, anomaly detection in one framework |

> _"This isn't a hackathon-only project — it's a foundation for quantum cybersecurity."_

---

### Pro Tips for This Slide:

- Lead with **"Who benefits?"** — judges want to see human impact, not just metrics
- The commercialization path shows you've thought beyond code → _business value_
- The scalability timeline proves quantum relevance grows with hardware progress
- ~60 sec delivery: ~15 sec per section, close with the quote

---

---

## SLIDE 6 — CONCLUSION & CLOSING

### Purpose: Mic-drop slide. Quick recap → uniqueness → strong close. Keep it CLEAN.

### Visual Layout: Three minimal sections — maximum white space, zero clutter.

---

#### 🔁 Quick Recap: Problem → Quantum Solution → Impact

| Step         | One-liner                                                             |
| ------------ | --------------------------------------------------------------------- |
| **Problem**  | IDS fails at 64% accuracy with zero explainability                    |
| **Solution** | Trainable VQC + VAE encoding + SHAP/LIME — all quantum-powered        |
| **Impact**   | +25% accuracy gain, transparent alerts, open-source for the community |

---

#### ⭐ What Makes Us Unique?

- ✅ **First** quantum IDS with explainability (SHAP + LIME)
- ✅ **First** to fix quantum encoding failures with VAE (not PCA)
- ✅ **First** quantum IDS benchmarked on 3 real-world datasets
- ✅ 100% free & open-source stack — reproducible by anyone

---

#### 🎤 Closing Statement (large font, bold, centered)

> **"We don't just detect threats — we explain them. This is our launchpad into quantum cybersecurity."**

---

**Optional additions (bottom bar):**

- Future roadmap: 16→32 qubits, real-time deployment, quantum security platform
- Collaboration: Open to research partnerships & industry integration
- Team contact email / GitHub link / QR code

---

### Pro Tips for This Slide:

- Memorize the closing line — deliver it looking at the judges, not the screen
- The recap table should be **glanceable in 3 seconds**
- This slide = 30 sec max. Say the recap, hit the "firsts", deliver the close.
- If judges ask follow-ups, pull up Slide 4 (architecture) or Slide 5 (impact) as backup

---

---

## IMAGE PLACEMENT SUMMARY

| Slide   | Image to Use                                     | Purpose                                   |
| ------- | ------------------------------------------------ | ----------------------------------------- |
| Slide 1 | None (or subtle quantum-circuit background)      | Title aesthetics                          |
| Slide 2 | None (icon-driven)                               | Problem clarity through text              |
| Slide 3 | `Images/Arc_diag.png` (small preview)            | Tease the architecture                    |
| Slide 4 | **`Images/Arc_diag.png`** (LARGE, centered)      | **Primary visual — architecture**         |
| Slide 5 | `Images/class_diag.png` (thumbnail)              | Feasibility proof — "system is designed"  |
| Slide 5 | `Images/use_case_diag.png` (thumbnail, optional) | Feasibility proof — "32 use cases mapped" |
| Slide 6 | None (clean, minimal)                            | Impact statement                          |

### Images NOT recommended for PPT slides (too detailed for a 6-slide deck):

- Activity diagrams (7 module-level + 1 overall) — too granular
- Sequence diagrams (7 module-level + 1 overall) — too granular
- These are excellent for a **technical appendix** or **Q&A backup** but will overwhelm a 6-slide hackathon pitch

---

---

## PRESENTATION DELIVERY TIPS

### Timing (assume ~5-7 minutes total):

| Slide   | Time   | Focus                                                         |
| ------- | ------ | ------------------------------------------------------------- |
| Slide 1 | 15 sec | Title + hook line                                             |
| Slide 2 | 75 sec | **Problem clarity** — spend the most time here (30% weight!)  |
| Slide 3 | 75 sec | **Innovation** — 25 sec per panel, punch up the "firsts"      |
| Slide 4 | 60 sec | **Architecture + tech stack** — walk through the pipeline L→R |
| Slide 5 | 60 sec | **Outcomes + feasibility** — let the numbers speak            |
| Slide 6 | 30 sec | **Summary + call to action** — confident close                |

### Key Phrases to Use During Presentation:

- _"The current best quantum IDS gets only 64% — that means 1 in 3 attacks is missed."_ (Problem Clarity)
- _"We are the first to combine trainable quantum circuits with explainable AI for cybersecurity."_ (Innovation)
- _"Everything we use is free and open-source — Qiskit, IBM Quantum, public datasets."_ (Feasibility)
- _"Our architecture has 22 classes across 7 modules, all designed with proven software patterns."_ (Technical Plan)
- _"We don't just detect threats — we explain them."_ (Closing)

### Anticipated Q&A Questions + Answers:

1. **"Why not just use classical ML? What's the quantum advantage?"**
   → "Classical models plateau at ~80% on NB15. Quantum feature spaces can capture non-linear correlations that classical kernels miss — especially for rare attack types like R2L and U2R where we expect 10-12% quantum advantage."

2. **"How do you handle the noise on real quantum hardware?"**
   → "Our primary results use the Aer simulator. We plan optional real hardware validation on IBM Brisbane with 1024 shots per circuit. We expect 5-15% accuracy drop due to NISQ noise, and we document noise mitigation strategies."

3. **"Why 8 qubits? Why not more?"**
   → "Current NISQ devices have limited coherence times. 8 qubits is the sweet spot — enough expressiveness for our 8-dimensional VAE encodings, while keeping circuit depth manageable for real hardware execution."

4. **"How is this different from the base paper (Gouveia & Correia, 2020)?"**
   → "Three key differences: (1) We use trainable VQC instead of static QSVM kernels, (2) We use VAE instead of PCA to solve the distribution mismatch on NB15, (3) We add SHAP+LIME explainability — the base paper has none."

5. **"What's the real-world impact?"**
   → "Security analysts get transparent threat assessments — they can see which network features triggered each alert, building trust in quantum-powered security decisions. This is critical for SOC operations where false positives waste resources."

---

---

## DESIGN RECOMMENDATIONS

### Color Scheme:

- **Primary:** Dark navy (#1a1a2e) or deep purple (#16213e)
- **Accent 1:** Quantum blue (#0f3460)
- **Accent 2:** Cyber green (#00ff41) or electric teal (#00d2ff)
- **Alert/Important:** Red (#e94560) — use for the "64%" stat and pain points
- **Background:** Dark gradient (professional, tech-forward feel)

### Font Recommendations:

- **Headlines:** Montserrat Bold or Poppins Bold
- **Body:** Inter, Open Sans, or Roboto
- **Code/Technical:** JetBrains Mono or Fira Code (for circuit notation or code snippets)

### General Slide Design Rules:

- **Max 6 bullet points per slide** — otherwise it's a wall of text
- **One key message per slide** — if you can't summarize the slide in one sentence, it's overloaded
- **60/40 rule** — 60% visual (diagrams, tables, icons) and 40% text
- **Consistent margins and alignment** — use slide master/templates
- **No full paragraphs** — phrases and fragments only
