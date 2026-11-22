
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    f1_score
)

# ==========================================
# MLP MODEL EVALUATION
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "processed"
MODEL_DIR = PROJECT_DIR / "models"
OUTPUT_DIR = PROJECT_DIR / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)

print("=" * 60)
print("       FRAUD DETECTION - MODEL EVALUATION")
print("=" * 60)

# 1. Load validation data
X_val = np.load(DATA_DIR / "X_val.npy").astype(np.float32)
y_val = np.load(DATA_DIR / "y_val.npy").astype(int).reshape(-1)

# 2. Load trained MLP parameters
model = np.load(MODEL_DIR / "mlp_from_scratch.npz")

W1 = model["W1"]
b1 = model["b1"]
W2 = model["W2"]
b2 = model["b2"]

print("\n1. Model and validation data loaded")
print("Validation shape:", X_val.shape)

# 3. Forward propagation for inference
def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    x = np.clip(x, -50, 50)
    return 1 / (1 + np.exp(-x))

Z1 = X_val @ W1 + b1
A1 = relu(Z1)
Z2 = A1 @ W2 + b2
probabilities = sigmoid(Z2).reshape(-1)

# 4. Calculate ROC-AUC and PR-AUC
roc_auc = roc_auc_score(y_val, probabilities)
pr_auc = average_precision_score(y_val, probabilities)

print("\n2. Overall evaluation")
print(f"ROC-AUC: {roc_auc:.4f}")
print(f"PR-AUC:  {pr_auc:.4f}")

# 5. Compare different thresholds
thresholds_to_test = [
    0.01, 0.02, 0.05, 0.10,
    0.15, 0.20, 0.30, 0.40, 0.50
]

print("\n3. THRESHOLD COMPARISON")
print("-" * 75)
print(
    f"{'Threshold':<12}{'Precision':<14}"
    f"{'Recall':<14}{'F1 Score':<14}{'Alerts':<10}"
)
print("-" * 75)

best_f1 = -1
best_threshold = 0.5

for threshold in thresholds_to_test:
    predictions = (probabilities >= threshold).astype(int)

    precision = precision_score(
        y_val, predictions, zero_division=0
    )
    recall = recall_score(
        y_val, predictions, zero_division=0
    )
    f1 = f1_score(
        y_val, predictions, zero_division=0
    )

    alerts = int(predictions.sum())

    print(
        f"{threshold:<12.2f}{precision:<14.4f}"
        f"{recall:<14.4f}{f1:<14.4f}{alerts:<10}"
    )

    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

print("-" * 75)

# 6. Evaluate selected validation threshold
best_predictions = (
    probabilities >= best_threshold
).astype(int)

print("\n4. BEST VALIDATION F1 THRESHOLD")
print("Threshold:", best_threshold)
print("Precision:", precision_score(
    y_val, best_predictions, zero_division=0
))
print("Recall:", recall_score(
    y_val, best_predictions, zero_division=0
))
print("F1 Score:", f1_score(
    y_val, best_predictions, zero_division=0
))
from sklearn.metrics import precision_recall_curve

# Calculate precision and recall at different thresholds
precision_values, recall_values, thresholds = (
    precision_recall_curve(
        y_val.ravel(),
        probabilities.ravel()
    )
)

# Select a threshold with at least 50% precision
target_precision = 0.50

valid_indices = np.where(
    precision_values[:-1] >= target_precision
)[0]

print("\nTHRESHOLD OPTIMIZATION")

if len(valid_indices) > 0:

    # Among qualifying thresholds, select the one
    # with the highest recall
    best_index = valid_indices[
        np.argmax(recall_values[valid_indices])
    ]

    best_threshold = thresholds[best_index]

    print("Selected threshold:", round(float(best_threshold), 5))
    print("Precision:", round(float(precision_values[best_index]), 4))
    print("Recall:", round(float(recall_values[best_index]), 4))

    # Count predicted fraud alerts
    optimized_predictions = (
        probabilities.ravel() >= best_threshold
    ).astype(int)

    print(
        "Fraud alerts:",
        int(np.sum(optimized_predictions))
    )

    np.save(
        OUTPUT_DIR / "optimized_threshold.npy",
        np.array([best_threshold])
    )
else:
    print("50% precision was not achieved.")
    print("Try a lower target, such as 30%.")

# 7. Save precision-recall curve
precision_values, recall_values, curve_thresholds = precision_recall_curve(
    y_val, probabilities
)

plt.figure(figsize=(8, 6))
plt.plot(recall_values, precision_values)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve - MLP")
plt.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "precision_recall_curve.png")
plt.close()

# 8. Save confusion matrix at selected threshold
cm = confusion_matrix(y_val, best_predictions)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Normal", "Fraud"]
)

display.plot(cmap="Blues", values_format="d")
plt.title("MLP Confusion Matrix - Validation")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "confusion_matrix_mlp.png")
plt.close()

# 9. Save evaluation results
results = {
    "roc_auc": float(roc_auc),
    "pr_auc": float(pr_auc),
    "best_validation_f1_threshold": float(best_threshold),
    "validation_precision": float(
        precision_score(y_val, best_predictions, zero_division=0)
    ),
    "validation_recall": float(
        recall_score(y_val, best_predictions, zero_division=0)
    ),
    "validation_f1": float(best_f1)
}

with open(OUTPUT_DIR / "mlp_evaluation.json", "w") as file:
    json.dump(results, file, indent=4)

print("\n5. Evaluation files saved:")
print("outputs/precision_recall_curve.png")
print("outputs/confusion_matrix_mlp.png")
print("outputs/mlp_evaluation.json")

print("\n" + "=" * 60)
print("       EVALUATION COMPLETED!")
print("=" * 60)
