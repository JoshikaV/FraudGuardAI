from pathlib import Path

import joblib
import pandas as pd
from catboost import CatBoostClassifier


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_FILE = PROJECT_ROOT / "models" / "fraudguard_catboost.cbm"
FEATURE_FILE = PROJECT_ROOT / "models" / "feature_columns.pkl"
THRESHOLD_FILE = PROJECT_ROOT / "models" / "fraudguard_threshold.txt"


# ---------------------------------------------------------
# Load model
# ---------------------------------------------------------

model = CatBoostClassifier()
model.load_model(str(MODEL_FILE))

feature_columns = joblib.load(FEATURE_FILE)

with open(THRESHOLD_FILE, "r") as file:
    FRAUD_THRESHOLD = float(file.read().strip())


CATEGORICAL_COLUMNS = [
    "ProductCD",
    "card4",
    "card6",
    "P_emaildomain",
    "R_emaildomain",
    "DeviceType",
    "DeviceInfo",
]


# ---------------------------------------------------------
# Risk classification
# ---------------------------------------------------------

def get_risk_level(probability):

    if probability >= FRAUD_THRESHOLD:
        return "HIGH"

    elif probability >= 0.50:
        return "MEDIUM"

    else:
        return "LOW"


# ---------------------------------------------------------
# Recommended action
# ---------------------------------------------------------

def get_recommended_action(risk_level):

    if risk_level == "HIGH":
        return "Review transaction before approval"

    elif risk_level == "MEDIUM":
        return "Monitor or request additional verification"

    return "Proceed normally"


# ---------------------------------------------------------
# Simple human-readable explanations
# ---------------------------------------------------------

def generate_explanation(transaction, probability):

    reasons = []

    amount = transaction.get("TransactionAmt", 0)

    if pd.notna(amount):

        if amount >= 500:
            reasons.append(
                "High transaction amount increased the risk signal."
            )

        elif amount >= 200:
            reasons.append(
                "Moderately high transaction amount contributed to the risk."
            )

    device_type = transaction.get("DeviceType")

    if (
        pd.isna(device_type)
        or str(device_type).lower() == "unknown"
    ):
        reasons.append(
            "Device information is unavailable or unusual."
        )

    email = transaction.get("P_emaildomain")

    if (
        pd.isna(email)
        or str(email).lower() == "unknown"
    ):
        reasons.append(
            "Purchaser email information is unavailable."
        )

    if probability >= FRAUD_THRESHOLD:
        reasons.append(
            "The learned transaction pattern strongly resembles historical fraud."
        )

    elif probability >= 0.50:
        reasons.append(
            "The transaction contains patterns associated with elevated fraud risk."
        )

    else:
        reasons.append(
            "The overall transaction pattern is closer to legitimate historical activity."
        )

    return reasons[:4]


# ---------------------------------------------------------
# Prediction function
# ---------------------------------------------------------

def analyze_transaction(transaction):

    transaction = transaction.copy()

    # Engineered features must match training
    amount = transaction.get("TransactionAmt", 0)

    if pd.isna(amount):
        amount = 0

    import numpy as np

    transaction["TransactionAmt_log"] = np.log(
        max(float(amount), 0) + 1
    )

    transaction["TransactionAmt_decimal"] = (
        float(amount) % 1
    )

    # Build one-row DataFrame
    input_df = pd.DataFrame(
        [transaction]
    )

    # Add any missing model features
    for column in feature_columns:

        if column not in input_df.columns:

            if column in CATEGORICAL_COLUMNS:
                input_df[column] = "Unknown"

            else:
                input_df[column] = float("nan")

    # Correct feature order
    input_df = input_df[feature_columns]

    # Prepare categorical values exactly as training
    for column in CATEGORICAL_COLUMNS:

        if column in input_df.columns:

            input_df[column] = (
                input_df[column]
                .fillna("Unknown")
                .astype(str)
            )

    # Fraud probability
    probability = float(
        model.predict_proba(input_df)[0][1]
    )

    risk_level = get_risk_level(
        probability
    )

    action = get_recommended_action(
        risk_level
    )

    explanation = generate_explanation(
        transaction,
        probability
    )

    return {
        "fraud_probability": probability,
        "fraud_probability_percent": probability * 100,
        "risk_level": risk_level,
        "recommended_action": action,
        "explanation": explanation,
        "fraud_threshold": FRAUD_THRESHOLD,
    }


# ---------------------------------------------------------
# Local test
# ---------------------------------------------------------

if __name__ == "__main__":

    sample_transaction = {

        "TransactionDT": 10000000,

        "TransactionAmt": 650.00,

        "ProductCD": "W",

        "card1": 15000,
        "card2": 500,
        "card3": 150,
        "card4": "visa",
        "card5": 226,
        "card6": "credit",

        "addr1": 315,
        "addr2": 87,

        "dist1": 100,
        "dist2": None,

        "P_emaildomain": "Unknown",
        "R_emaildomain": "Unknown",

        "C1": 1,
        "C2": 1,
        "C4": 0,
        "C5": 0,
        "C6": 1,
        "C8": 0,
        "C9": 0,
        "C10": 0,
        "C11": 1,
        "C12": 0,
        "C13": 1,
        "C14": 1,

        "D1": 0,
        "D2": None,
        "D4": 0,
        "D10": 0,
        "D15": 0,

        "DeviceType": "Unknown",
        "DeviceInfo": "Unknown",
    }

    result = analyze_transaction(
        sample_transaction
    )

    print("=" * 60)
    print("FRAUDGUARD — TRANSACTION ANALYSIS")
    print("=" * 60)

    print(
        f"\nFraud Probability: "
        f"{result['fraud_probability_percent']:.2f}%"
    )

    print(
        "Risk Level:",
        result["risk_level"]
    )

    print(
        "Recommended Action:",
        result["recommended_action"]
    )

    print("\nWhy?")

    for reason in result["explanation"]:
        print("•", reason)

    print(
        f"\nFraud decision threshold: "
        f"{result['fraud_threshold']:.2f}"
    )