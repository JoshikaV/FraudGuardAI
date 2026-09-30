# FraudGuard AI — Snapdragon Deployment

## Current Prototype

FraudGuard currently performs local fraud-risk inference using a
CatBoost classifier trained on the IEEE-CIS Fraud Detection dataset.

Pipeline:

Transaction
→ Feature preprocessing
→ CatBoost fraud model
→ Risk score
→ LOW / MEDIUM / HIGH classification
→ Human-readable explanation

The prototype does not require transaction data to be sent to a
remote inference API.

## Snapdragon Target Architecture

The production Snapdragon version is designed around:

Transaction
→ Local preprocessing
→ ONNX-compatible fraud classifier
→ ONNX Runtime
→ Qualcomm QNN Execution Provider
→ Snapdragon NPU
→ Fraud risk + explanation

## Qualcomm AI Hub Integration

Qualcomm AI Hub can be used to:

1. Compile an ONNX-compatible model for a Snapdragon target.
2. Profile model latency and memory usage.
3. Validate inference on Snapdragon hardware.
4. Produce optimized deployment artifacts.
5. Integrate the optimized model into the local FraudGuard application.

## Why On-Device?

Financial transaction information can be sensitive.

On-device inference provides:

- Reduced dependence on cloud inference
- Lower network latency
- Local processing of transaction features
- Offline inference capability
- Efficient execution on Snapdragon AI hardware

## Prototype Status

Implemented:
- IEEE-CIS fraud dataset pipeline
- Temporal validation
- CatBoost baseline classifier
- Threshold optimization
- LOW / MEDIUM / HIGH risk engine
- Human-readable risk explanations
- Interactive Streamlit dashboard
- Local inference

Target optimization:
- ONNX-compatible model
- Qualcomm AI Hub compilation
- ONNX Runtime integration
- QNN Execution Provider
- Snapdragon NPU profiling