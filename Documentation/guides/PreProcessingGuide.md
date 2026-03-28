# Preprocessing Guide — Colab

---

## Section 1 — Understanding the Three Datasets

### 1.1 Attack Category Coverage Per Dataset

| Attack Category          | NSL-KDD | UNSW-NB15 | CICIDS-2017 |
| ------------------------ | ------- | --------- | ----------- |
| DoS / DDoS               | ✅      | ✅        | ✅          |
| Port Scanning            | ✅      | ✅        | ✅          |
| Brute Force              | ✅      | ✅        | ✅          |
| Buffer Overflow          | ✅      | ✅        | ❌          |
| SQL Injection            | ❌      | ❌        | ✅          |
| XSS Web Attacks          | ❌      | ❌        | ✅          |
| Backdoors                | ❌      | ✅        | ❌          |
| Worms                    | ❌      | ✅        | ❌          |
| Botnet                   | ❌      | ❌        | ✅          |
| Fuzzing                  | ❌      | ✅        | ❌          |
| Shellcode Injection      | ❌      | ✅        | ❌          |
| Heartbleed (CVE)         | ❌      | ❌        | ✅          |
| Rootkits                 | ✅      | ❌        | ❌          |
| Reconnaissance           | ✅      | ✅        | ✅          |
| **Total Unique Classes** | **4**   | **9**     | **11**      |

---

### 1.2 Unified 5-Class Master Label System

All three datasets are mapped to the same 5 integer classes.
This is what `y_train` and `y_test` will contain after Step 7.

| Master Label | Name    | Meaning                                             |
| ------------ | ------- | --------------------------------------------------- |
| **0**        | NORMAL  | Legitimate traffic                                  |
| **1**        | DoS     | Denial of Service / DDoS attacks                    |
| **2**        | PROBE   | Port scanning / Reconnaissance / Fuzzing / Analysis |
| **3**        | EXPLOIT | R2L, U2R, Brute Force, Web attacks, Buffer Overflow |
| **4**        | MALWARE | Backdoors, Worms, Botnet, Shellcode Injection       |

**Per-dataset class availability after mapping:**

| Dataset     | Classes present after mapping |
| ----------- | ----------------------------- |
| NSL-KDD     | 0, 1, 2, 3 (no MALWARE)       |
| UNSW-NB15   | 0, 1, 2, 3, 4 (all 5)         |
| CICIDS-2017 | 0, 1, 2, 3, 4 (all 5)         |

NSL-KDD does not contain backdoors, worms, or shellcode so class 4
never appears in it — that is expected and fine.

---

### 1.3 Dimension Problem

After loading and one-hot encoding, each dataset has a different width.
The quantum circuit is fixed at **8 qubits = 8 input features**.
PCA bridges the gap.

| Dataset     | Raw Features | Categorical Cols             | After One-Hot (approx) | After PCA |
| ----------- | ------------ | ---------------------------- | ---------------------- | --------- |
| NSL-KDD     | 41           | protocol_type, service, flag | ~120                   | **8**     |
| UNSW-NB15   | 49           | proto, state, service        | ~80                    | **8**     |
| CICIDS-2017 | 78           | none (all numeric)           | ~74 (after drops)      | **8**     |

---

## Section 2 — Exact Attack-to-Master-Label Mapping

### NSL-KDD

| NSL-KDD Raw Label                                                                                                                           | Master Label |
| ------------------------------------------------------------------------------------------------------------------------------------------- | ------------ |
| normal                                                                                                                                      | 0 — NORMAL   |
| back, land, neptune, pod, smurf, teardrop, apache2, udpstorm, processtable, mailbomb                                                        | 1 — DoS      |
| ipsweep, nmap, portsweep, satan, mscan, saint                                                                                               | 2 — PROBE    |
| ftp_write, guess_passwd, imap, multihop, phf, spy, warezclient, warezmaster, sendmail, named, snmpgetattack, snmpguess, xlock, xsnoop, worm | 3 — EXPLOIT  |
| buffer_overflow, loadmodule, perl, rootkit, httptunnel, ps, sqlattack, xterm                                                                | 3 — EXPLOIT  |

> U2R and R2L both map to 3 (EXPLOIT). NSL-KDD has no class 4.

---

### UNSW-NB15

Uses the `attack_cat` column (not the binary `label` column).

| UNSW-NB15 attack_cat | Master Label |
| -------------------- | ------------ |
| Normal               | 0 — NORMAL   |
| DoS                  | 1 — DoS      |
| Generic              | 1 — DoS      |
| Reconnaissance       | 2 — PROBE    |
| Fuzzers              | 2 — PROBE    |
| Analysis             | 2 — PROBE    |
| Exploits             | 3 — EXPLOIT  |
| Backdoors            | 4 — MALWARE  |
| Shellcode            | 4 — MALWARE  |
| Worms                | 4 — MALWARE  |

---

### CICIDS-2017

| CICIDS-2017 Label                                                      | Master Label |
| ---------------------------------------------------------------------- | ------------ |
| BENIGN                                                                 | 0 — NORMAL   |
| DoS slowloris, DoS Slowhttptest, DoS Hulk, DoS GoldenEye, DDoS         | 1 — DoS      |
| PortScan                                                               | 2 — PROBE    |
| FTP-Patator, SSH-Patator                                               | 3 — EXPLOIT  |
| Web Attack – Brute Force, Web Attack – XSS, Web Attack – Sql Injection | 3 — EXPLOIT  |
| Heartbleed, Infiltration                                               | 3 — EXPLOIT  |
| Bot                                                                    | 4 — MALWARE  |

---

## Section 3 — Step-by-Step Colab Preprocessing

### Step 1 — Install dependencies

```python
!pip install scikit-learn pandas numpy imbalanced-learn -q
```

---

### Step 2 — Upload dataset CSV to Colab

```python
from google.colab import files
uploaded  = files.upload()
file_path = list(uploaded.keys())[0]
```

Or mount Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
file_path = '/content/drive/MyDrive/your_dataset.csv'
```

---

### Step 3 — Load the raw CSV

```python
import pandas as pd
import numpy as np

df = pd.read_csv(file_path, low_memory=False)
print(df.shape)
print(df.head(2))
```

For NSL-KDD without a header row:

```python
NSL_KDD_COLUMNS = [
    "duration","protocol_type","service","flag","src_bytes","dst_bytes",
    "land","wrong_fragment","urgent","hot","num_failed_logins","logged_in",
    "num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate","dst_host_srv_diff_host_rate",
    "dst_host_serror_rate","dst_host_srv_serror_rate","dst_host_rerror_rate",
    "dst_host_srv_rerror_rate","label","difficulty_level",
]

# Use this only if the CSV has no header
df = pd.read_csv(file_path, header=None, names=NSL_KDD_COLUMNS, low_memory=False)
```

---

### Step 4 — Set dataset and constants

Change only ONE line when switching datasets:

```python
DATASET     = "nsl_kdd"       # <-- change this only
# DATASET   = "unsw_nb15"
# DATASET   = "cicids2017"

QUANTUM_DIM = 8               # DO NOT change — fixed by the VQC circuit
```

---

### Step 5 — Drop irrelevant columns

```python
DROP_COLS = {
    "nsl_kdd":    ["difficulty_level"],
    "unsw_nb15":  ["srcip", "dstip", "stime", "ltime"],
    "cicids2017": ["Flow ID", "Source IP", "Destination IP", "Timestamp"],
}

df = df.drop(columns=[c for c in DROP_COLS[DATASET] if c in df.columns])
print("After drop:", df.shape)
```

---

### Step 6 — Separate features and label column

```python
LABEL_COL = {
    "nsl_kdd":    "label",
    "unsw_nb15":  "attack_cat",   # use attack_cat for multi-class
    "cicids2017": "Label",
}

target_col = LABEL_COL[DATASET]

# For UNSW-NB15, also drop the binary 'label' column from X
EXTRA_DROP = {
    "nsl_kdd":    [],
    "unsw_nb15":  ["label"],       # binary col, not needed
    "cicids2017": [],
}

X = df.drop(columns=[target_col] + EXTRA_DROP[DATASET], errors='ignore')
y = df[target_col].copy()

print("X shape:", X.shape)
print("y sample:", y.unique()[:10])
```

---

### Step 7 — Map all labels to the unified 5-class master label

Everything becomes 0 / 1 / 2 / 3 / 4.

**NSL-KDD**

```python
if DATASET == "nsl_kdd":
    DOS   = {"back","land","neptune","pod","smurf","teardrop",
             "apache2","udpstorm","processtable","mailbomb"}
    PROBE = {"ipsweep","nmap","portsweep","satan","mscan","saint"}
    EXPLOIT = {
        # R2L
        "ftp_write","guess_passwd","imap","multihop","phf","spy",
        "warezclient","warezmaster","sendmail","named",
        "snmpgetattack","snmpguess","xlock","xsnoop","worm",
        # U2R
        "buffer_overflow","loadmodule","perl","rootkit",
        "httptunnel","ps","sqlattack","xterm",
    }

    def map_nsl(label):
        l = str(label).lower().strip().rstrip(".")
        if l == "normal":   return 0   # NORMAL
        if l in DOS:        return 1   # DoS
        if l in PROBE:      return 2   # PROBE
        if l in EXPLOIT:    return 3   # EXPLOIT
        return 1                       # unknown → DoS

    y = y.apply(map_nsl)
    print("NSL-KDD classes:", sorted(y.unique()))
    # Expected: [0, 1, 2, 3]   — class 4 does not exist in NSL-KDD
```

**UNSW-NB15**

```python
if DATASET == "unsw_nb15":
    y = y.fillna("normal").str.strip().str.lower()

    UNSW_MAP = {
        "normal":         0,   # NORMAL
        "dos":            1,   # DoS
        "generic":        1,   # DoS
        "reconnaissance": 2,   # PROBE
        "fuzzers":        2,   # PROBE
        "analysis":       2,   # PROBE
        "exploits":       3,   # EXPLOIT
        "backdoors":      4,   # MALWARE
        "shellcode":      4,   # MALWARE
        "worms":          4,   # MALWARE
    }

    y = y.map(UNSW_MAP).fillna(1).astype(int)
    print("UNSW-NB15 classes:", sorted(y.unique()))
    # Expected: [0, 1, 2, 3, 4]
```

**CICIDS-2017**

```python
if DATASET == "cicids2017":
    y = y.str.strip()

    CICIDS_MAP = {
        "BENIGN":                      0,   # NORMAL
        "DoS slowloris":               1,   # DoS
        "DoS Slowhttptest":            1,
        "DoS Hulk":                    1,
        "DoS GoldenEye":               1,
        "DDoS":                        1,
        "PortScan":                    2,   # PROBE
        "FTP-Patator":                 3,   # EXPLOIT
        "SSH-Patator":                 3,
        "Web Attack \x96 Brute Force": 3,
        "Web Attack – Brute Force":    3,
        "Web Attack - Brute Force":    3,
        "Web Attack \x96 XSS":         3,
        "Web Attack – XSS":            3,
        "Web Attack - XSS":            3,
        "Web Attack \x96 Sql Injection":3,
        "Web Attack – Sql Injection":  3,
        "Web Attack - Sql Injection":  3,
        "Heartbleed":                  3,
        "Infiltration":                3,
        "Bot":                         4,   # MALWARE
    }

    y = y.map(CICIDS_MAP).fillna(0).astype(int)
    print("CICIDS-2017 classes:", sorted(y.unique()))
    # Expected: [0, 1, 2, 3, 4]
```

> **Note on CICIDS-2017 dashes:** The dataset files sometimes encode the dash
> in "Web Attack – Brute Force" as a Unicode en-dash (–), a Windows-1252
> byte (\x96), or a plain hyphen (-). All three variants are mapped above.

---

### Step 8 — One-hot encode categorical columns

```python
CAT_COLS = {
    "nsl_kdd":    ["protocol_type", "service", "flag"],
    "unsw_nb15":  ["proto", "state", "service"],
    "cicids2017": [],
}

cat_cols_present = [c for c in CAT_COLS[DATASET] if c in X.columns]
print("Encoding:", cat_cols_present)

if cat_cols_present:
    X = pd.get_dummies(X, columns=cat_cols_present, drop_first=False)

print("X shape after one-hot:", X.shape)
```

---

### Step 9 — Impute missing values

```python
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="median")
X_arr   = imputer.fit_transform(X)

print("Shape:", X_arr.shape)
print("NaNs remaining:", np.isnan(X_arr).any())
```

---

### Step 10 — Scale to [0, 1]

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))
X_arr  = scaler.fit_transform(X_arr)

print("Min:", round(X_arr.min(), 4), "  Max:", round(X_arr.max(), 4))
```

---

### Step 11 — Train / test split

```python
from sklearn.model_selection import train_test_split

y_arr = y.to_numpy() if hasattr(y, 'to_numpy') else np.array(y)

X_train, X_test, y_train, y_test = train_test_split(
    X_arr, y_arr,
    test_size    = 0.20,
    random_state = 42,
    stratify     = y_arr,
)

print("X_train:", X_train.shape)
print("X_test: ", X_test.shape)
print("Classes in train:", np.unique(y_train))
```

---

### Step 12 — SMOTE for class imbalance

NSL-KDD has extremely few EXPLOIT samples, so SMOTE fires for it almost always.
UNSW-NB15 and CICIDS-2017 should be checked.

```python
from imblearn.over_sampling import SMOTE

classes, counts = np.unique(y_train, return_counts=True)
for cls, cnt in zip(classes, counts):
    NAMES = {0:"NORMAL", 1:"DoS", 2:"PROBE", 3:"EXPLOIT", 4:"MALWARE"}
    print(f"  Class {cls} ({NAMES[cls]}): {cnt} samples")

min_ratio = counts.min() / counts.max()
print(f"\nMin class ratio: {min_ratio:.4f}")

if min_ratio < 0.10:
    k = max(1, min(5, counts.min() - 1))
    smote = SMOTE(k_neighbors=k, random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    print("After SMOTE — X_train:", X_train.shape)
else:
    print("Balance OK — SMOTE skipped")
```

---

### Step 13 — Reduce to 8 features with PCA

This is the dimension adjustment step.
PCA is fitted on X_train only, then reused on X_test.

```python
from sklearn.decomposition import PCA

print(f"Features before PCA : {X_train.shape[1]}")
print(f"Features after PCA  : {QUANTUM_DIM}")

pca        = PCA(n_components=QUANTUM_DIM, random_state=42)
X_train_q  = pca.fit_transform(X_train)   # fit + transform on train
X_test_q   = pca.transform(X_test)        # transform only on test

explained  = pca.explained_variance_ratio_.sum()
print(f"Variance retained   : {explained:.3f}")
print(f"X_train_q : {X_train_q.shape}")
print(f"X_test_q  : {X_test_q.shape}")
```

Expected results:

| Dataset     | Before PCA | After PCA | Typical Variance Retained |
| ----------- | ---------- | --------- | ------------------------- |
| NSL-KDD     | ~120       | 8         | ~0.75 – 0.85              |
| UNSW-NB15   | ~80        | 8         | ~0.70 – 0.80              |
| CICIDS-2017 | ~74        | 8         | ~0.80 – 0.90              |

---

### Step 14 — Verify and confirm

```python
assert X_train_q.shape[1] == QUANTUM_DIM, "Dimension mismatch!"
assert X_test_q.shape[1]  == QUANTUM_DIM, "Dimension mismatch!"

MASTER_NAMES = {0:"NORMAL", 1:"DoS", 2:"PROBE", 3:"EXPLOIT", 4:"MALWARE"}
classes_present = {c: MASTER_NAMES[c] for c in np.unique(y_train)}

print("\n========== PREPROCESSING COMPLETE ==========")
print(f"Dataset      : {DATASET}")
print(f"X_train_q    : {X_train_q.shape}  ← feed to VQC and classical models")
print(f"X_test_q     : {X_test_q.shape}")
print(f"y_train      : {y_train.shape}")
print(f"y_test       : {y_test.shape}")
print(f"Classes      : {classes_present}")
print(f"Variance kept: {explained:.3f}")
```

---

## Section 4 — Quick Reference

### What changes per dataset (Steps 4–7 only)

| Step               | NSL-KDD                      | UNSW-NB15                         | CICIDS-2017             |
| ------------------ | ---------------------------- | --------------------------------- | ----------------------- |
| DATASET var        | `"nsl_kdd"`                  | `"unsw_nb15"`                     | `"cicids2017"`          |
| Drop cols          | difficulty_level             | srcip, dstip, stime, ltime, label | Flow ID, IPs, Timestamp |
| Label column       | label (string)               | attack_cat (string)               | Label (string)          |
| Categorical encode | protocol_type, service, flag | proto, state, service             | none                    |
| Classes present    | 0, 1, 2, 3                   | 0, 1, 2, 3, 4                     | 0, 1, 2, 3, 4           |
| SMOTE needed       | yes (class 3 very rare)      | check ratio                       | check ratio             |

### What stays the same for all datasets

- QUANTUM_DIM = 8
- Imputer: median strategy
- Scaler: MinMaxScaler [0, 1]
- Split: 80/20 stratified
- PCA: n_components = 8, fit on train only
- Model inputs: `X_train_q`, `X_test_q` (always shape `(n, 8)`)
- Labels: unified 0–4 master label

---

## Section 5 — Passing Output to Models

```python
# Classical models (SVM, Random Forest, XGBoost)
model.fit(X_train_q, y_train)
preds = model.predict(X_test_q)

# Quantum VQC (Qiskit)
vqc.fit(X_train_q, y_train)
preds = vqc.predict(X_test_q)
```

`X_train_q` and `X_test_q` are the only arrays every model sees.
Shape is always `(n_samples, 8)` regardless of which dataset was loaded.

---

## Section 6 — Notes

- Never call `pca.fit()` or `scaler.fit()` on test data. Only `.transform()`.
- Save `pca`, `scaler`, `imputer` with `joblib.dump()` to reuse on live traffic.
- Switching datasets = change only `DATASET = "..."` in Step 4.
- Variance retained ~0.75 is acceptable for now. VAE will replace PCA later
  to increase variance retention without losing distributional information
  (this is the planned fix for the UNSW-NB15 accuracy gap from the base paper).
- CICIDS-2017 is 8 GB. Use `df = pd.read_csv(..., nrows=500000)` during
  development to load only the first 500 000 rows.
