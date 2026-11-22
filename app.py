from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn

# =====================================================
# CREDIT CARD FRAUD DETECTION DASHBOARD
# =====================================================

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_DIR / "models"
DATA_DIR = PROJECT_DIR / "processed"
OUTPUT_DIR = PROJECT_DIR / "outputs"

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

# -----------------------------------------------------
# 1. LOAD MODEL ARCHITECTURE
# -----------------------------------------------------

class FraudDetectionMLP(nn.Module):

    def __init__(self, input_size):

        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.30),

            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.20),

            nn.Linear(32, 16),
            nn.ReLU(),

            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.network(x)


@st.cache_resource
def load_model_and_assets():

    with open(DATA_DIR / "feature_names.json") as f:
        feature_names = json.load(f)

    scaler = joblib.load(DATA_DIR / "scaler.pkl")

    model = FraudDetectionMLP(len(feature_names))

    state_dict = torch.load(
        MODEL_DIR / "pytorch_mlp.pth",
        map_location="cpu",
        weights_only=True
    )

    model.load_state_dict(state_dict)
    model.eval()

    with open(OUTPUT_DIR / "pytorch_mlp_results.json") as f:
        results = json.load(f)

    return model, scaler, feature_names, results


# -----------------------------------------------------
# 2. DASHBOARD HEADER
# -----------------------------------------------------

st.title("💳 Credit Card Fraud Detection")
st.write(
    "A deep learning application using a PyTorch "
    "Multi-Layer Perceptron to identify potentially "
    "fraudulent credit card transactions."
)

st.divider()

# -----------------------------------------------------
# 3. LOAD SAVED MODEL
# -----------------------------------------------------

try:

    model, scaler, feature_names, results = load_model_and_assets()

except Exception as e:

    st.error("Unable to load the trained model or its files.")
    st.exception(e)
    st.stop()


threshold = float(results["best_validation_threshold"])

st.success("Trained fraud detection model loaded successfully.")

# -----------------------------------------------------
# 4. MODEL PERFORMANCE
# -----------------------------------------------------

st.header("1. Model Performance")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Model",
    "Deep MLP"
)

col2.metric(
    "ROC-AUC",
    f'{results["roc_auc"]:.4f}'
)

col3.metric(
    "PR-AUC",
    f'{results["pr_auc"]:.4f}'
)

col4.metric(
    "Decision Threshold",
    f"{threshold:.4f}"
)

st.caption(
    "Metrics shown above were calculated on validation data "
    "during model training."
)

st.divider()

# -----------------------------------------------------
# 5. TRAINING VISUALIZATIONS
# -----------------------------------------------------

st.header("2. Training and Evaluation Graphs")

graph_files = [
    (
        "Training Loss",
        OUTPUT_DIR / "pytorch_training_loss.png"
    ),
    (
        "Precision-Recall Curve",
        OUTPUT_DIR / "pytorch_precision_recall.png"
    ),
    (
        "Confusion Matrix",
        OUTPUT_DIR / "pytorch_confusion_matrix.png"
    )
]

for title, image_path in graph_files:

    if image_path.exists():

        st.subheader(title)
        st.image(str(image_path), use_container_width=True)

    else:

        st.warning(f"Graph not found: {image_path.name}")

st.divider()

# -----------------------------------------------------
# 6. SYNTHETIC FRAUD DATA
# -----------------------------------------------------

st.header("3. Synthetic Fraud Data Analysis")

synthetic_path = OUTPUT_DIR / "synthetic_fraud_pca.png"
comparison_path = OUTPUT_DIR / "synthetic_feature_comparison.png"

if synthetic_path.exists():

    st.subheader("PCA Visualization")
    st.image(str(synthetic_path), use_container_width=True)

if comparison_path.exists():

    st.subheader("Original vs Synthetic Feature Comparison")
    st.image(str(comparison_path), use_container_width=True)

st.write(
    "A Variational Autoencoder (VAE) was trained on "
    "fraudulent training examples to generate experimental "
    "synthetic transactions."
)

st.caption(
    "Synthetic transactions are experimental samples, "
    "not verified real-world fraud cases."
)

st.divider()

# -----------------------------------------------------
# 7. TRANSACTION PREDICTION
# -----------------------------------------------------

st.header("4. Test a Transaction")

st.write(
    "Upload a CSV containing transaction features. "
    "The CSV must contain the same 30 input features "
    "used during model training."
)

st.info(
    "Upload raw, unscaled feature values. "
    "Do not include the Class column. "
    "The application applies the saved scaler automatically."
)

uploaded_file = st.file_uploader(
    "Upload transaction CSV",
    type=["csv"]
)

if uploaded_file is not None:

    try:

        transaction_df = pd.read_csv(uploaded_file)

        # Remove target if accidentally included
        if "Class" in transaction_df.columns:
            transaction_df = transaction_df.drop(columns=["Class"])

        missing_features = [
            col for col in feature_names
            if col not in transaction_df.columns
        ]

        if missing_features:

            st.error(
                "Your CSV is missing these features: "
                + ", ".join(missing_features)
            )

        else:

            # Keep exactly the training feature order
            transaction_df = transaction_df[feature_names]

            if transaction_df.isnull().values.any():

                st.error(
                    "Your uploaded CSV contains missing values. "
                    "Please clean the data before uploading."
                )

            else:

                scaled_data = scaler.transform(transaction_df)

                input_tensor = torch.tensor(
                    scaled_data,
                    dtype=torch.float32
                )

                with torch.no_grad():

                    logits = model(input_tensor)

                    probabilities = torch.sigmoid(
                        logits
                    ).numpy().reshape(-1)

                predictions = (
                    probabilities >= threshold
                ).astype(int)

                results_df = pd.DataFrame({
                    "Fraud Probability": probabilities,
                    "Prediction": [
                        "Fraud" if p == 1 else "Normal"
                        for p in predictions
                    ]
                })

                st.subheader("Prediction Results")

                st.dataframe(
                    results_df,
                    use_container_width=True
                )

                fraud_count = int(np.sum(predictions == 1))
                normal_count = int(np.sum(predictions == 0))

                col1, col2 = st.columns(2)

                col1.metric("Potential Fraud", fraud_count)
                col2.metric("Normal Transactions", normal_count)

                csv_data = results_df.to_csv(index=False).encode("utf-8")

                st.download_button(
                    "Download Prediction Results",
                    data=csv_data,
                    file_name="fraud_predictions.csv",
                    mime="text/csv"
                )

    except Exception as e:

        st.error("Something went wrong while processing the CSV.")
        st.exception(e)

else:

    st.info(
        "Upload a transaction CSV to test the trained model."
    )

st.divider()

# -----------------------------------------------------
# 8. PROJECT INFORMATION
# -----------------------------------------------------

st.header("5. Project Information")

st.write("""
**Project:** Credit Card Fraud Detection and Synthetic Risk Lab

**Technology:** Python, PyTorch, NumPy, Pandas, Scikit-learn and Streamlit

**Deep Learning Model:** Multi-Layer Perceptron (MLP)

**Additional Experiments:** LSTM sequence model and Variational Autoencoder (VAE)

**Application Purpose:** Demonstrate how deep learning can help identify
potentially fraudulent credit card transactions.
""")

st.warning(
    "This is an academic demonstration. Predictions should not be "
    "used as the sole basis for real financial decisions."
)

st.caption("Credit Card Fraud Detection and Synthetic Risk Lab | Python Project")