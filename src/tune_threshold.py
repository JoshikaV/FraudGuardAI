from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = PROJECT_ROOT / "data" / "fraudguard_training.csv"
MODEL_FILE = PROJECT_ROOT / "models" / "fraudguard_catboost.cbm"
THRESHOLD_FILE = PROJECT_ROOT / "models" / "fraudguard_threshold.txt"


print("=" * 60)
print("FRAUDGUARD — THRESHOLD TUNING")
print("=" * 60)


# ---------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------

print("\nLoading validation data...")

df = pd.read_csv(DATA_FILE)

X = df.drop(
    columns=[
        "isFraud",
        "TransactionID"
    ]
)

y = df["isFraud"]


# ---------------------------------------------------------
# 2. Handle categorical columns exactly as during training
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
    column
    for column in categorical_columns
    if column in X.columns
]

for column in categorical_columns:
    X[column] = X[column].fillna("Unknown").astype(str)


# ---------------------------------------------------------
# 3. Use same temporal validation split
# ---------------------------------------------------------

split_index = int(len(X) * 0.80)

X_valid = X.iloc[split_index:].copy()
y_valid = y.iloc[split_index:].copy()


# ---------------------------------------------------------
# 4. Load trained model
# ---------------------------------------------------------

print("Loading FraudGuard model...")

model = CatBoostClassifier()

model.load_model(
    str(MODEL_FILE)
)


# ---------------------------------------------------------
# 5. Fraud probabilities
# ---------------------------------------------------------

print("Generating fraud probabilities...")

probabilities = model.predict_proba(
    X_valid
)[:, 1]


# ---------------------------------------------------------
# 6. Test different thresholds
# ---------------------------------------------------------

thresholds = np.arange(
    0.30,
    0.91,
    0.05
)

results = []

print("\nThreshold comparison:\n")

print(
    f"{'Threshold':<12}"
    f"{'Precision':<12}"
    f"{'Recall':<12}"
    f"{'F1':<12}"
)

print("-" * 48)


for threshold in thresholds:

    predictions = (
        probabilities >= threshold
    ).astype(int)

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

    results.append({
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1
    })

    print(
        f"{threshold:<12.2f}"
        f"{precision:<12.4f}"
        f"{recall:<12.4f}"
        f"{f1:<12.4f}"
    )


# ---------------------------------------------------------
# 7. Select threshold with highest F1
# ---------------------------------------------------------

results_df = pd.DataFrame(results)

best_row = results_df.loc[
    results_df["f1"].idxmax()
]

best_threshold = float(
    best_row["threshold"]
)


print("\n" + "=" * 60)
print("BEST THRESHOLD")
print("=" * 60)

print(
    f"Threshold: {best_threshold:.2f}"
)

print(
    f"Precision: {best_row['precision']:.4f}"
)

print(
    f"Recall:    {best_row['recall']:.4f}"
)

print(
    f"F1 Score:  {best_row['f1']:.4f}"
)


# ---------------------------------------------------------
# 8. Confusion matrix
# ---------------------------------------------------------

best_predictions = (
    probabilities >= best_threshold
).astype(int)

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_valid,
        best_predictions
    )
)


# ---------------------------------------------------------
# 9. Save threshold
# ---------------------------------------------------------

with open(
    THRESHOLD_FILE,
    "w"
) as file:

    file.write(
        str(best_threshold)
    )


print("\nThreshold saved:")
print(THRESHOLD_FILE)

print("\nThreshold tuning complete.")