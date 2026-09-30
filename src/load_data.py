from pathlib import Path
import pandas as pd


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

TRANSACTION_FILE = DATA_DIR / "train_transaction.csv"
IDENTITY_FILE = DATA_DIR / "train_identity.csv"


def load_data():
    print("=" * 60)
    print("FRAUDGUARD — DATA LOADING")
    print("=" * 60)

    print("\nLoading transaction data...")
    transactions = pd.read_csv(TRANSACTION_FILE)

    print("Loading identity data...")
    identities = pd.read_csv(IDENTITY_FILE)

    print("\nTransaction dataset:")
    print("Rows:", transactions.shape[0])
    print("Columns:", transactions.shape[1])

    print("\nIdentity dataset:")
    print("Rows:", identities.shape[0])
    print("Columns:", identities.shape[1])

    print("\nFraud distribution:")
    print(transactions["isFraud"].value_counts())

    print("\nFraud percentage:")
    fraud_rate = transactions["isFraud"].mean() * 100
    print(f"{fraud_rate:.2f}%")

    return transactions, identities


if __name__ == "__main__":
    load_data()