from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn

# =====================================================
# CREDIT CARD FRAUD DETECTION APPLICATION
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

# =====================================================
# MODEL ARCHITECTURE
# =====================================================

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


# =====================================================
# LOAD MODEL AND ASSETS
# =====================================================

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


# =====================================================
# PREDICTION FUNCTION
# =====================================================

def predict_transactions(transaction_df, model, scaler, threshold):

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

    return probabilities, predictions


# =====================================================
# LOAD MODEL
# =====================================================

try:

    model, scaler, feature_names, results = load_model_and_assets()

    threshold = float(
        results["best_validation_threshold"]
    )

except Exception as e:

    st.error("Unable to load the trained model.")

    st.exception(e)

    st.stop()


# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("💳 Fraud Detection")

st.sidebar.write(
    "Deep Learning Based Credit Card Fraud Detection"
)

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "✍️ Manual Prediction",
        "📂 Batch Prediction",
        "📊 Model Analytics",
        "🧪 Synthetic Data",
        "ℹ️ Project Information"
    ]
)

st.sidebar.divider()

st.sidebar.success("Trained MLP Model Loaded")

st.sidebar.metric(
    "Decision Threshold",
    f"{threshold:.4f}"
)

st.sidebar.caption(
    "Academic demonstration using a PyTorch Deep Learning model."
)


# =====================================================
# PAGE 1: HOME
# =====================================================

if page == "🏠 Home":

    st.title("💳 Credit Card Fraud Detection")

    st.subheader(
        "Deep Learning Based Fraud Detection System"
    )

    st.write("""
    This application uses a Deep Learning Multi-Layer Perceptron
    to identify potentially fraudulent credit card transactions.

    Users can manually enter transaction details or upload
    transaction data in CSV format to receive predictions.
    """)

    st.success("Trained Deep Learning Model Loaded Successfully")

    st.divider()

    st.header("Model Overview")

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
        "Threshold",
        f"{threshold:.4f}"
    )

    st.divider()

    st.header("Application Features")

    col1, col2 = st.columns(2)

    with col1:

        st.info("""
        **Manual Fraud Prediction**

        Enter transaction information using an
        interactive form and get an instant prediction.
        """)

        st.info("""
        **Batch Prediction**

        Upload a CSV containing multiple transactions
        and analyse them simultaneously.
        """)

    with col2:

        st.info("""
        **Model Analytics**

        Explore model evaluation metrics, training loss,
        precision-recall curves and confusion matrix.
        """)

        st.info("""
        **Synthetic Data Analysis**

        Explore experimental synthetic fraud generation
        using a Variational Autoencoder.
        """)


# =====================================================
# PAGE 2: MANUAL PREDICTION
# =====================================================

elif page == "✍️ Manual Prediction":

    st.title("✍️ Manual Transaction Prediction")

    st.write("""
    Enter the transaction details below to analyse
    whether the transaction may be fraudulent.
    """)

    st.info("""
    The dataset uses anonymized features V1–V28.
    These features do not represent directly interpretable
    details such as merchant, location or card type.
    """)

    # Identify important features
    time_feature = next(
        (f for f in feature_names if f.lower() == "time"),
        None
    )

    amount_feature = next(
        (f for f in feature_names if f.lower() == "amount"),
        None
    )

    other_features = [
        f for f in feature_names
        if f not in [time_feature, amount_feature]
    ]

    # -----------------------------------------------
    # INPUT FORM
    # -----------------------------------------------

    with st.form("manual_prediction_form"):

        st.subheader("Transaction Details")

        col1, col2 = st.columns(2)

        with col1:

            if amount_feature:

                amount = st.number_input(
                    "Transaction Amount",
                    min_value=0.0,
                    value=1000.0,
                    step=100.0,
                    help="Enter the transaction amount."
                )

            else:

                amount = 0.0

        with col2:

            if time_feature:

                transaction_time = st.number_input(
                    "Transaction Time",
                    min_value=0.0,
                    value=0.0,
                    step=100.0,
                    help="Time elapsed since the first transaction in the dataset."
                )

            else:

                transaction_time = 0.0

        st.divider()

        st.subheader("Advanced Model Features")

        st.write("""
        The model also requires anonymized transaction features.
        For a simplified demonstration, these are initialized to zero.

        You can expand the section below to manually enter
        the feature values.
        """)

        with st.expander(
            "⚙️ Enter Advanced Features (V1–V28)",
            expanded=False
        ):

            st.warning("""
            These anonymized features cannot be derived from
            transaction amount or time alone.

            Zero values are demonstration defaults and may not
            represent a real transaction.
            """)

            feature_values = {}

            feature_columns = st.columns(3)

            for index, feature in enumerate(other_features):

                with feature_columns[index % 3]:

                    feature_values[feature] = st.number_input(
                        feature,
                        value=0.0,
                        step=0.01,
                        format="%.6f",
                        key=f"manual_{feature}"
                    )

        submitted = st.form_submit_button(
            "🔍 Detect Fraud",
            use_container_width=True
        )

    # -----------------------------------------------
    # PROCESS PREDICTION
    # -----------------------------------------------

    if submitted:

        transaction = {}

        if time_feature:
            transaction[time_feature] = transaction_time

        if amount_feature:
            transaction[amount_feature] = amount

        for feature in other_features:
            transaction[feature] = feature_values[feature]

        # Preserve training feature order
        transaction_df = pd.DataFrame(
            [transaction]
        )[feature_names]

        try:

            probabilities, predictions = predict_transactions(
                transaction_df,
                model,
                scaler,
                threshold
            )

            probability = float(
                probabilities[0]
            )

            prediction = int(
                predictions[0]
            )

            st.divider()

            st.header("Prediction Result")

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Transaction Amount",
                    f"{amount:,.2f}"
                )

            with col2:

                st.metric(
                    "Fraud Probability",
                    f"{probability * 100:.2f}%"
                )

            st.progress(
                min(max(probability, 0.0), 1.0)
            )

            if prediction == 1:

                st.error(
                    "🚨 Potentially Fraudulent Transaction"
                )

                st.write("""
                The model classified this transaction
                as potentially fraudulent based on
                the selected decision threshold.
                """)

            else:

                st.success(
                    "✅ Transaction Classified as Legitimate"
                )

                st.write("""
                The model classified this transaction
                as legitimate based on the selected
                decision threshold.
                """)

            # Risk classification
            if probability >= 0.80:

                risk_level = "High Risk"

            elif probability >= 0.40:

                risk_level = "Moderate Risk"

            else:

                risk_level = "Low Risk"

            st.subheader("Risk Analysis")

            st.metric(
                "Risk Category",
                risk_level
            )

            st.write(
                f"Model Decision Threshold: {threshold:.4f}"
            )

            st.write(
                f"Predicted Probability: {probability:.6f}"
            )

            st.divider()

            st.caption("""
            This is an academic demonstration. The anonymized
            features are initialized to zero unless manually changed.
            The result should not be considered a real-world
            financial risk assessment.
            """)

        except Exception as e:

            st.error(
                "Prediction failed."
            )

            st.exception(e)


# =====================================================
# PAGE 3: BATCH PREDICTION
# =====================================================

elif page == "📂 Batch Prediction":

    st.title("📂 Batch Fraud Detection")

    st.write("""
    Upload a CSV file containing transaction features.
    The trained Deep Learning model will analyse all
    transactions and generate predictions.
    """)

    st.info("""
    Upload raw, unscaled feature values.

    The saved scaler will automatically preprocess the data.

    Do not include the Class column.
    """)

    # CSV TEMPLATE
    template_df = pd.DataFrame(
        columns=feature_names
    )

    template_csv = template_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "📥 Download CSV Template",
        data=template_csv,
        file_name="transaction_template.csv",
        mime="text/csv"
    )

    st.divider()

    uploaded_file = st.file_uploader(
        "Upload Transaction CSV",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:

            transaction_df = pd.read_csv(
                uploaded_file
            )

            if "Class" in transaction_df.columns:

                transaction_df = transaction_df.drop(
                    columns=["Class"]
                )

            missing_features = [
                col for col in feature_names
                if col not in transaction_df.columns
            ]

            if missing_features:

                st.error(
                    "Missing required features:"
                )

                st.write(missing_features)

            elif transaction_df.empty:

                st.error(
                    "The uploaded CSV contains no transactions."
                )

            else:

                transaction_df = transaction_df[
                    feature_names
                ]

                transaction_df = transaction_df.apply(
                    pd.to_numeric,
                    errors="coerce"
                )

                if transaction_df.isnull().values.any():

                    st.error(
                        "The CSV contains missing or non-numeric values."
                    )

                else:

                    st.subheader("Uploaded Transactions")

                    st.dataframe(
                        transaction_df.head(10),
                        use_container_width=True
                    )

                    st.write(
                        f"Total transactions: **{len(transaction_df)}**"
                    )

                    probabilities, predictions = predict_transactions(
                        transaction_df,
                        model,
                        scaler,
                        threshold
                    )

                    results_df = pd.DataFrame({

                        "Fraud Probability (%)":
                            probabilities * 100,

                        "Prediction": [
                            "Fraudulent"
                            if p == 1
                            else "Legitimate"
                            for p in predictions
                        ]
                    })

                    st.header("Prediction Results")

                    fraud_count = int(
                        np.sum(predictions == 1)
                    )

                    normal_count = int(
                        np.sum(predictions == 0)
                    )

                    col1, col2, col3 = st.columns(3)

                    col1.metric(
                        "Total Transactions",
                        len(predictions)
                    )

                    col2.metric(
                        "Potential Fraud",
                        fraud_count
                    )

                    col3.metric(
                        "Legitimate Transactions",
                        normal_count
                    )

                    st.dataframe(
                        results_df,
                        use_container_width=True
                    )

                    if fraud_count > 0:

                        st.error(
                            f"⚠️ {fraud_count} potentially fraudulent "
                            "transaction(s) detected."
                        )

                    else:

                        st.success(
                            "No potentially fraudulent transactions detected."
                        )

                    complete_results = pd.concat(
                        [
                            transaction_df.reset_index(drop=True),
                            results_df.reset_index(drop=True)
                        ],
                        axis=1
                    )

                    csv_data = complete_results.to_csv(
                        index=False
                    ).encode("utf-8")

                    st.download_button(
                        "📥 Download Prediction Results",
                        data=csv_data,
                        file_name="fraud_predictions.csv",
                        mime="text/csv"
                    )

        except Exception as e:

            st.error(
                "An error occurred while processing the CSV."
            )

            st.exception(e)


# =====================================================
# PAGE 4: MODEL ANALYTICS
# =====================================================

elif page == "📊 Model Analytics":

    st.title("📊 Model Analytics")

    st.write(
        "Evaluation metrics and visualizations "
        "of the trained Deep Learning model."
    )

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

    st.divider()

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

            st.image(
                str(image_path),
                use_container_width=True
            )

        else:

            st.warning(
                f"Graph not found: {image_path.name}"
            )


# =====================================================
# PAGE 5: SYNTHETIC DATA
# =====================================================

elif page == "🧪 Synthetic Data":

    st.title("🧪 Synthetic Fraud Data Analysis")

    st.write("""
    Experimental synthetic fraud data generation
    using a Variational Autoencoder (VAE).
    """)

    synthetic_path = (
        OUTPUT_DIR / "synthetic_fraud_pca.png"
    )

    comparison_path = (
        OUTPUT_DIR / "synthetic_feature_comparison.png"
    )

    if synthetic_path.exists():

        st.subheader("PCA Visualization")

        st.image(
            str(synthetic_path),
            use_container_width=True
        )

    else:

        st.warning(
            "Synthetic PCA visualization not found."
        )

    if comparison_path.exists():

        st.subheader(
            "Original vs Synthetic Feature Comparison"
        )

        st.image(
            str(comparison_path),
            use_container_width=True
        )

    else:

        st.warning(
            "Feature comparison visualization not found."
        )

    st.divider()

    st.info("""
    A Variational Autoencoder was trained on fraudulent
    training examples to generate experimental synthetic data.

    Synthetic transactions are experimental samples
    and are not verified real-world fraud cases.
    """)


# =====================================================
# PAGE 6: PROJECT INFORMATION
# =====================================================

elif page == "ℹ️ Project Information":

    st.title("ℹ️ Project Information")

    st.header("Credit Card Fraud Detection")

    st.write("""
    **Project Objective**

    To develop a machine learning system that identifies
    potentially fraudulent credit card transactions
    using transaction data and deep learning techniques.
    """)

    st.header("Technology Stack")

    col1, col2 = st.columns(2)

    with col1:

        st.write("""
        - Python
        - PyTorch
        - NumPy
        - Pandas
        - Scikit-learn
        """)

    with col2:

        st.write("""
        - Streamlit
        - Matplotlib
        - Joblib
        - Machine Learning
        - Deep Learning
        """)

    st.header("Models and Experiments")

    st.write("""
    - Multi-Layer Perceptron (MLP)
    - LSTM
    - Variational Autoencoder (VAE)
    - Synthetic Fraud Generation
    - Model Evaluation and Comparison
    """)

    st.header("Application Workflow")

    st.write("""
    1. Transaction data is entered manually or uploaded.
    2. The transaction features are validated.
    3. The saved preprocessing scaler transforms the data.
    4. The trained Deep Learning model generates predictions.
    5. The fraud probability is compared against the
       selected decision threshold.
    6. The application displays and exports the results.
    """)

    st.warning("""
    This application is developed for academic demonstration.
    Predictions should not be used as the sole basis
    for real financial decisions.
    """)

    st.caption(
        "Credit Card Fraud Detection | Python Project"
    )