from pathlib import Path
import sys
import streamlit as st

# Allow dashboard to import src/predict.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))

from predict import analyze_transaction


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="FraudGuard AI",
    page_icon="🛡️",
    layout="wide"
)


# ---------------------------------------------------------
# Styling
# ---------------------------------------------------------

st.markdown(
    """
    <style>
    .main-title {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 0;
    }

    .subtitle {
        font-size: 17px;
        opacity: 0.75;
        margin-bottom: 25px;
    }

    .risk-card {
        padding: 25px;
        border-radius: 15px;
        border: 1px solid rgba(128,128,128,0.3);
        margin-top: 15px;
    }

    .risk-low {
        border-left: 8px solid #28a745;
    }

    .risk-medium {
        border-left: 8px solid #ffc107;
    }

    .risk-high {
        border-left: 8px solid #dc3545;
    }

    .risk-label {
        font-size: 32px;
        font-weight: 800;
    }

    .small-text {
        opacity: 0.7;
        font-size: 13px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ---------------------------------------------------------
# Header
# ---------------------------------------------------------

st.markdown(
    '<div class="main-title">🛡️ FraudGuard AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
    On-device Financial Fraud & Anomaly Copilot
    </div>
    """,
    unsafe_allow_html=True
)

st.caption(
    "Privacy-first fraud risk analysis designed for "
    "Snapdragon-powered PCs."
)

st.divider()


# ---------------------------------------------------------
# Main layout
# ---------------------------------------------------------

input_column, result_column = st.columns(
    [1, 1.15],
    gap="large"
)


# ---------------------------------------------------------
# Transaction inputs
# ---------------------------------------------------------

with input_column:

    st.subheader("Transaction Details")

    transaction_amount = st.number_input(
        "Transaction Amount ($)",
        min_value=0.0,
        value=650.0,
        step=10.0
    )

    product_cd = st.selectbox(
        "Product Type",
        ["W", "C", "R", "H", "S"]
    )

    card_network = st.selectbox(
        "Card Network",
        [
            "visa",
            "mastercard",
            "american express",
            "discover",
            "Unknown"
        ]
    )

    card_type = st.selectbox(
        "Card Type",
        [
            "credit",
            "debit",
            "Unknown"
        ]
    )

    device_type = st.selectbox(
        "Device Type",
        [
            "desktop",
            "mobile",
            "Unknown"
        ],
        index=2
    )

    purchaser_email = st.selectbox(
        "Purchaser Email Domain",
        [
            "gmail.com",
            "yahoo.com",
            "hotmail.com",
            "outlook.com",
            "anonymous.com",
            "Unknown"
        ],
        index=5
    )

    distance = st.number_input(
        "Transaction Distance",
        min_value=0.0,
        value=100.0
    )

    analyze_button = st.button(
        "🔍 Analyze Transaction",
        type="primary",
        use_container_width=True
    )


# ---------------------------------------------------------
# Results
# ---------------------------------------------------------

with result_column:

    st.subheader("FraudGuard Analysis")

    if not analyze_button:

        st.info(
            "Enter transaction information and select "
            "**Analyze Transaction**."
        )

        st.markdown("### What FraudGuard evaluates")

        st.write(
            "• Transaction characteristics\n\n"
            "• Payment/card patterns\n\n"
            "• Device information\n\n"
            "• Email/account signals\n\n"
            "• Historical fraud patterns"
        )


# ---------------------------------------------------------
# Analyze
# ---------------------------------------------------------

if analyze_button:

    transaction = {

        "TransactionDT": 10000000,

        "TransactionAmt": transaction_amount,

        "ProductCD": product_cd,

        "card1": 15000,
        "card2": 500,
        "card3": 150,
        "card4": card_network,
        "card5": 226,
        "card6": card_type,

        "addr1": 315,
        "addr2": 87,

        "dist1": distance,
        "dist2": None,

        "P_emaildomain": purchaser_email,
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

        "DeviceType": device_type,
        "DeviceInfo": "Unknown"
    }

    result = analyze_transaction(transaction)

    probability = result[
        "fraud_probability_percent"
    ]

    risk = result["risk_level"]

    action = result["recommended_action"]

    explanations = result["explanation"]


    with result_column:

        # ---------------------------------------------
        # Risk card
        # ---------------------------------------------

        css_class = {
            "LOW": "risk-low",
            "MEDIUM": "risk-medium",
            "HIGH": "risk-high"
        }[risk]

        st.markdown(
            f"""
            <div class="risk-card {css_class}">
                <div class="small-text">
                    FRAUD RISK
                </div>

                <div class="risk-label">
                    {risk} RISK
                </div>

                <h2>
                    {probability:.2f}%
                </h2>

                <div class="small-text">
                    Model risk score
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.progress(
            min(probability / 100, 1.0)
        )

        st.markdown("### Recommended Action")

        if risk == "HIGH":

            st.error(
                f"🚨 {action}"
            )

        elif risk == "MEDIUM":

            st.warning(
                f"⚠️ {action}"
            )

        else:

            st.success(
                f"✅ {action}"
            )


        # ---------------------------------------------
        # Explanation
        # ---------------------------------------------

        st.markdown(
            "### Why did FraudGuard assign this risk?"
        )

        for explanation in explanations:

            st.write(
                f"• {explanation}"
            )


        # ---------------------------------------------
        # Decision information
        # ---------------------------------------------

        with st.expander(
            "View decision details"
        ):

            st.write(
                f"**Risk score:** "
                f"{probability:.2f}%"
            )

            st.write(
                f"**High-risk threshold:** "
                f"{result['fraud_threshold'] * 100:.0f}%"
            )

            st.write(
                "**Model:** CatBoost fraud classifier"
            )

            st.write(
                "**Validation ROC-AUC:** 0.9062"
            )

            st.write(
                "**Validation PR-AUC:** 0.4825"
            )


# ---------------------------------------------------------
# Bottom section
# ---------------------------------------------------------

st.divider()

metric1, metric2, metric3, metric4 = st.columns(4)

metric1.metric(
    "ROC-AUC",
    "0.906"
)

metric2.metric(
    "PR-AUC",
    "0.483"
)

metric3.metric(
    "High Risk Threshold",
    "85%"
)

metric4.metric(
    "Training Transactions",
    "590K+"
)

st.divider()

st.markdown("### ⚡ Snapdragon On-Device Architecture")

st.write(
    """
    FraudGuard currently performs fraud-risk inference locally,
    keeping the prediction pipeline independent of a remote
    inference API.

    The Snapdragon deployment target uses an ONNX-compatible
    fraud model with ONNX Runtime and the Qualcomm QNN Execution
    Provider for hardware-accelerated inference on Snapdragon PCs.
    Qualcomm AI Hub provides the model compilation, profiling and
    device-validation path for this target architecture.
    """
)

st.caption(
    "Prototype trained using the IEEE-CIS Fraud Detection dataset."
)