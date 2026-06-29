# 💳 Financial Fraud Detection

A machine learning project that explores and engineers features from a large-scale, highly imbalanced mobile-money transaction dataset to detect fraudulent transactions.

---

## 📌 Project Overview

Financial fraud costs banks and payment platforms billions every year. This project analyzes a synthetic mobile-money transaction dataset to understand **how and where fraud occurs**, engineer features that expose fraudulent behavior, and lay the groundwork for a binary classification model that flags fraudulent transactions in real time.

- **Task type:** Binary Classification (`isFraud`: 0 = legitimate, 1 = fraud)
- **Real-world analogy:** Same problem solved by fraud-scoring engines at PayPal, Visa, and mobile wallets like M-Pesa/UPI.
- **Core challenge:** Severe class imbalance — only **~0.13%** of transactions are fraudulent.

---

## 📂 Dataset

**File:** `AIML_Dataset.csv` (https://shorturl.at/PvLvt)

| Property       | Value                                          |
| -------------- | ---------------------------------------------- |
| Rows           | 6,362,620                                      |
| Columns        | 11 (original) → 12 (after feature engineering) |
| Missing values | None                                           |
| Fraud rate     | 8,213 / 6,362,620 (~0.13%)                     |

### Columns

| Column                              | Description                                                                                 |
| ----------------------------------- | ------------------------------------------------------------------------------------------- |
| `step`                              | Simulated time unit (1 step = 1 hour, 744 steps = 30 days)                                  |
| `type`                              | Transaction type — `PAYMENT`, `CASH_IN`, `CASH_OUT`, `TRANSFER`, `DEBIT`                    |
| `amount`                            | Transaction amount                                                                          |
| `nameOrig`                          | Sender/customer ID                                                                          |
| `oldbalanceOrg` / `newbalanceOrig`  | Sender's balance before / after the transaction                                             |
| `nameDest`                          | Receiver ID                                                                                 |
| `oldbalanceDest` / `newbalanceDest` | Receiver's balance before / after the transaction                                           |
| `isFraud`                           | **Target variable** — ground-truth fraud label                                              |
| `isFlaggedFraud`                    | A rule-based system's own flag (not ground truth — excluded from modeling to avoid leakage) |

---

## 🔍 Key EDA Findings

1. **Fraud only occurs in `TRANSFER` and `CASH_OUT` transactions.** `PAYMENT`, `CASH_IN`, and `DEBIT` have a 0% fraud rate.
2. **Extreme class imbalance** (~0.13% positive class) — accuracy is not a usable metric.
3. **`amount` is heavily right-skewed** (mean ≈ ₹179,861, max ≈ ₹92.4M) — log-transformed for visualization, revealing a bimodal distribution.
4. **Weak linear correlation** between any single numeric feature and `isFraud` (highest is `amount` at 0.077) — fraud likely depends on **feature interactions**, not single-variable thresholds.
5. **Strong multicollinearity** between `oldbalanceOrg` ↔ `newbalanceOrig` (0.998) and `oldbalanceDest` ↔ `newbalanceDest` (0.977).
6. **"Account drained to zero"** after a transfer/cash-out occurs in ~1.18M transactions — a common pattern, but far too broad to use alone (only 8,213 are real fraud).
7. Fraudulent `nameOrig` IDs appear **only once each** — ruling out "repeat offender" account-history features in this dataset.

---

## ⚙️ Feature Engineering

| New Feature       | Formula                           | Purpose                                                 |
| ----------------- | --------------------------------- | ------------------------------------------------------- |
| `balanceDiffOrig` | `oldbalanceOrg - newbalanceOrig`  | Captures actual balance change vs. stated `amount`      |
| `balanceDiffDest` | `oldbalanceDest - newbalanceDest` | Flags inconsistent balance updates on the receiving end |

`step` was dropped after engineering the balance-diff features (flagged as a possible improvement — see below).

---

## 🛠️ Tech Stack

- **Language:** Python 3
- **Libraries:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`

---

## 🚀 Planned Pipeline (Next Steps)

The notebook currently imports the following but does not yet execute them — this is the intended pipeline:

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
```

1. **Filter** to `TRANSFER` / `CASH_OUT` transactions (per EDA finding #1).
2. **Drop** high-cardinality ID columns (`nameOrig`, `nameDest`) or aggregate them into engineered features.
3. **Preprocess** using a `ColumnTransformer`:
   - `OneHotEncoder(drop="first")` on `type`
   - `StandardScaler()` on numeric columns
4. **Split** with `train_test_split(..., stratify=df["isFraud"])` to preserve the fraud ratio in train/test sets.
5. **Train** a baseline `LogisticRegression(class_weight="balanced")`, then compare against tree-based models (Random Forest / XGBoost) to capture non-linear interactions.
6. **Evaluate** using `classification_report`, `confusion_matrix`, **F1-score**, and **PR-AUC** (not accuracy, given the imbalance).

## 📁 Project Structure

```
.
├── AIML_Dataset.csv        # Raw dataset
├── analysis_model.ipynb    # EDA, feature engineering, and modeling notebook
├── app.py                  # Python file to run code locally using streamlit
└── README.md               # Project documentation
```

## 🙋 How to Run

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
jupyter notebook analysis_model.ipynb
streamlit run app.py
```

---

## 📜 License

This project uses a synthetic dataset for educational/portfolio purposes. No real customer or financial data is involved.
