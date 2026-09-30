from pathlib import Path
import pandas as pd
import numpy as np
import joblib

from catboost import CatBoostClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_ROOT / "data" / "fraudguard_training.csv"
MODEL_DIR = PROJECT_ROOT / "models"

MODEL_DIR.mkdir(exist_ok=True)

MODEL_FILE = MODEL_DIR / "fraudguard_catboost.cbm"
FEATURE_FILE = MODEL_DIR / "feature_columns.pkl"


print("=" * 60)
print("FRAUDGUARD — MODEL TRAINING")
print("=" * 60)


# ---------------------------------------------------------
# 1. Load prepared dataset
# ---------------------------------------------------------

print("\nLoading prepared dataset...")

df = pd.read_csv(DATA_FILE)

print("Rows:", len(df))
print("Columns:", len(df.columns))


# ---------------------------------------------------------
# 2. Remove columns not used as model inputs
# ---------------------------------------------------------

X = df.drop(
    columns=[
        "isFraud",
        "TransactionID"
    ]
)

y = df["isFraud"]


# ---------------------------------------------------------
# 3. Identify categorical columns
# ---------------------------------------------------------

categorical_columns = [
    "ProductCD",
    "card4",
    "card6",
    "P_emaildomain",
    "R_emaildomain",
    "DeviceType",
    "DeviceInfo"
]

categorical_columns = [
    column for column in categorical_columns
    if column in X.columns
]


# CatBoost requires categorical missing values to be handled
# explicitly as strings.

for column in categorical_columns:
    X[column] = X[column].fillna("Unknown").astype(str)


# ---------------------------------------------------------
# 4. Temporal train / validation split
# ---------------------------------------------------------

split_index = int(len(X) * 0.80)

X_train = X.iloc[:split_index].copy()
X_valid = X.iloc[split_index:].copy()

y_train = y.iloc[:split_index].copy()
y_valid = y.iloc[split_index:].copy()


print("\nTemporal split:")
print("Training rows:", len(X_train))
print("Validation rows:", len(X_valid))

print("\nTraining fraud rate:")
print(f"{y_train.mean() * 100:.2f}%")

print("Validation fraud rate:")
print(f"{y_valid.mean() * 100:.2f}%")


# ---------------------------------------------------------
# 5. Create CatBoost model
# ---------------------------------------------------------

print("\nTraining CatBoost model...")

model = CatBoostClassifier(
    iterations=500,
    depth=7,
    learning_rate=0.08,
    loss_function="Logloss",
    eval_metric="AUC",
    auto_class_weights="Balanced",
    random_seed=42,
    verbose=50,
    allow_writing_files=False
)


# ---------------------------------------------------------
# 6. Train
# ---------------------------------------------------------

model.fit(
    X_train,
    y_train,
    cat_features=categorical_columns,
    eval_set=(X_valid, y_valid),
    early_stopping_rounds=50
)


# ---------------------------------------------------------
# 7. Generate fraud probabilities
# ---------------------------------------------------------

probabilities = model.predict_proba(X_valid)[:, 1]


# ---------------------------------------------------------
# 8. Evaluate at default threshold
# ---------------------------------------------------------

threshold = 0.50

predictions = (
    probabilities >= threshold
).astype(int)


roc_auc = roc_auc_score(
    y_valid,
    probabilities
)

pr_auc = average_precision_score(
    y_valid,
    probabilities
)

precision = precision_score(
    y_valid,
    predictions,
    zero_division=0
)

recall = recall_score(
    y_valid,
    predictions,
    zero_division=0
)

f1 = f1_score(
    y_valid,
    predictions,
    zero_division=0
)


print("\n" + "=" * 60)
print("VALIDATION RESULTS")
print("=" * 60)

print(f"ROC-AUC:   {roc_auc:.4f}")
print(f"PR-AUC:    {pr_auc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_valid,
        predictions
    )
)


# ---------------------------------------------------------
# 9. Save model
# ---------------------------------------------------------

model.save_model(str(MODEL_FILE))

joblib.dump(
    list(X.columns),
    FEATURE_FILE
)


print("\nModel saved:")
print(MODEL_FILE)

print("\nFeature list saved:")
print(FEATURE_FILE)

print("\nFraudGuard baseline training complete.")