from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

TRANSACTION_FILE = DATA_DIR / "train_transaction.csv"
IDENTITY_FILE = DATA_DIR / "train_identity.csv"
OUTPUT_FILE = DATA_DIR / "fraudguard_training.csv"


def prepare_data():

    print("=" * 60)
    print("FRAUDGUARD — DATA PREPARATION")
    print("=" * 60)

    print("\nLoading datasets...")

    transactions = pd.read_csv(TRANSACTION_FILE)
    identities = pd.read_csv(IDENTITY_FILE)

    # ---------------------------------------------------------
    # 1. Merge transaction and identity information
    # ---------------------------------------------------------

    print("Merging transaction and identity data...")

    df = transactions.merge(
        identities,
        on="TransactionID",
        how="left"
    )

    print("Merged shape:", df.shape)

    # ---------------------------------------------------------
    # 2. Select useful and understandable baseline features
    # ---------------------------------------------------------

    features = [
        "TransactionID",
        "TransactionDT",
        "TransactionAmt",

        "ProductCD",

        "card1",
        "card2",
        "card3",
        "card4",
        "card5",
        "card6",

        "addr1",
        "addr2",

        "dist1",
        "dist2",

        "P_emaildomain",
        "R_emaildomain",

        "C1",
        "C2",
        "C4",
        "C5",
        "C6",
        "C8",
        "C9",
        "C10",
        "C11",
        "C12",
        "C13",
        "C14",

        "D1",
        "D2",
        "D4",
        "D10",
        "D15",

        "DeviceType",
        "DeviceInfo",

        "isFraud"
    ]

    # Keep only columns that actually exist
    available_features = [
        column for column in features
        if column in df.columns
    ]

    df = df[available_features].copy()

    # ---------------------------------------------------------
    # 3. Sort chronologically
    # ---------------------------------------------------------

    df = df.sort_values("TransactionDT").reset_index(drop=True)

    # ---------------------------------------------------------
    # 4. Basic useful engineered features
    # ---------------------------------------------------------

    df["TransactionAmt_log"] = (
        df["TransactionAmt"].clip(lower=0) + 1
    ).apply("log")

    df["TransactionAmt_decimal"] = (
        df["TransactionAmt"] % 1
    )

    # ---------------------------------------------------------
    # 5. Dataset summary
    # ---------------------------------------------------------

    print("\nPrepared dataset:")
    print("Rows:", len(df))
    print("Columns:", len(df.columns))

    print("\nFraud distribution:")
    print(df["isFraud"].value_counts())

    print("\nMissing values — top 10:")
    print(
        df.isnull()
        .mean()
        .sort_values(ascending=False)
        .head(10)
    )

    # ---------------------------------------------------------
    # 6. Save
    # ---------------------------------------------------------

    df.to_csv(OUTPUT_FILE, index=False)

    print("\nSaved:")
    print(OUTPUT_FILE)

    print("\nData preparation complete.")

    return df


if __name__ == "__main__":
    prepare_data()