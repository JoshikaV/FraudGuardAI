# 🛡️ FraudGuard AI

### On-Device Financial Fraud & Anomaly Copilot for Snapdragon PCs

FraudGuard AI is a privacy focused financial fraud detection prototype
designed to analyze transaction risk locally and provide an interpretable
risk assessment.

The system uses machine learning to convert transaction information into
a fraud risk score, LOW/MEDIUM/HIGH risk classification, recommended
action, and a human readable explanation.

---

## 🎯 Problem

Financial fraud detection systems often rely on centralized processing
of sensitive transaction information.

FraudGuard explores a privacy first alternative in which fraud risk
inference can be performed locally, with a deployment path targeting
Snapdragon powered PCs.

---

## 💡 Solution

FraudGuard processes transaction characteristics such as:

- Transaction amount
- Product information
- Card characteristics
- Email/account signals
- Device information
- Historical transaction features

and produces:

```text
Transaction
     ↓
Feature Processing
     ↓
Fraud Detection Model
     ↓
Risk Score
     ↓
LOW / MEDIUM / HIGH
     ↓
Recommended Action
     ↓
Human-Readable Explanation
```

---

## 📊 Dataset

The prototype is trained using the IEEE-CIS Fraud Detection dataset.

Training transactions:

**590,540**

Fraud transactions:

**20,663**

Fraud rate:

**3.50%**

The data is highly imbalanced, so evaluation focuses on metrics beyond
simple accuracy.

---

## 🧠 Machine Learning

Current prototype:

- CatBoost binary classifier
- Chronological 80/20 train validation split
- Class balancing
- Mixed numerical and categorical features
- Threshold optimization
- Local inference

### Validation Results

| Metric | Result |
|---|---:|
| ROC-AUC | 0.9062 |
| PR-AUC | 0.4825 |
| Tuned threshold | 0.85 |
| Precision @ threshold | 0.5583 |
| Recall @ threshold | 0.4230 |
| F1 @ threshold | 0.4813 |

The threshold was selected from the tested thresholds using validation
F1 rather than assuming the default 0.50 threshold.

---

## 🚦 Risk Engine

FraudGuard converts the model score into an application level risk
assessment:

```text
LOW       → lower risk
MEDIUM    → additional verification / monitoring
HIGH      → transaction review
```

The HIGH-risk boundary uses the tuned fraud decision threshold.

---

## 💬 Fraud Copilot

Instead of displaying only a binary prediction, FraudGuard provides:

- Model risk score
- Risk level
- Recommended action
- Plain English risk explanation

Example:

```text
Risk Score: 84.91%

Risk Level:
MEDIUM

Recommended Action:
Monitor or request additional verification

Risk Signals:
• High transaction amount
• Missing/unusual device information
• Missing purchaser email information
• Elevated learned fraud pattern
```

---

## 🖥️ Interactive Dashboard

The Streamlit interface allows a user to enter transaction information
and run the FraudGuard risk analysis pipeline interactively.

Run:

```bash
streamlit run app/dashboard.py
```

---

## ⚡ Snapdragon Target Architecture

The current prototype performs fraud inference locally using CatBoost.

The intended Snapdragon deployment path is:

```text
Transaction
       ↓
Local Feature Processing
       ↓
ONNX-Compatible Fraud Model
       ↓
ONNX Runtime
       ↓
Qualcomm QNN Execution Provider
       ↓
Snapdragon AI Hardware
       ↓
Fraud Risk + Explanation
```

Qualcomm AI Hub can be used in the optimization stage for model
compilation, profiling and validation on supported Snapdragon targets.

The current repository should therefore be considered a working local
prototype with a Snapdragon deployment target, rather than a claim that
the CatBoost model is already executing on the Snapdragon NPU.

---

## 🔐 Why On-Device?

Local inference can provide:

- Reduced dependence on cloud inference
- Lower network latency
- Greater privacy for transaction features
- Offline analysis capability
- A path toward hardware accelerated AI inference

---

## 📁 Project Structure

```text
Snapdragon-FraudGuard/
│
├── app/
│   └── dashboard.py
│
├── data/
│   └── fraudguard_training.csv        # local only
│
├── models/
│   ├── fraudguard_catboost.cbm
│   ├── fraudguard_threshold.txt
│   └── feature_columns.pkl
│
├── src/
│   ├── load_data.py
│   ├── prepare_data.py
│   ├── train_model.py
│   ├── tune_threshold.py
│   └── predict.py
│
├── .gitignore
├── requirements.txt
├── snapdragon_deployment.md
└── README.md
```

---

## 🚀 Running FraudGuard

### 1. Create environment

```bash
python -m venv .venv
```

### 2. Activate environment

Windows:

```bash
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add IEEE-CIS data

Place:

```text
train_transaction.csv
train_identity.csv
```

inside:

```text
data/
```

### 5. Prepare data

```bash
python src/prepare_data.py
```

### 6. Train

```bash
python src/train_model.py
```

### 7. Tune threshold

```bash
python src/tune_threshold.py
```

### 8. Launch application

```bash
streamlit run app/dashboard.py
```

---

## 🔮 Future Work

- ONNX compatible fraud classifier
- Qualcomm AI Hub compilation and profiling
- Snapdragon NPU execution through QNN
- Model calibration analysis
- Model attribution based explanations
- Behavioral or velocity features
- Production transaction API integration
- Drift monitoring

---

## ⚠️ Prototype Disclaimer

FraudGuard is an experimental hackathon prototype and is not intended
to make real-world financial authorization decisions without further
validation, calibration, security review and compliance testing.
