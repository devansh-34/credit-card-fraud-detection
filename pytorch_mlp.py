
from pathlib import Path
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    confusion_matrix,
    classification_report
)

import matplotlib.pyplot as plt

# ==========================================================
# CREDIT CARD FRAUD DETECTION
# DEEP MLP USING PYTORCH
# ==========================================================

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "processed"
MODEL_DIR = PROJECT_DIR / "models"
OUTPUT_DIR = PROJECT_DIR / "outputs"

MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

np.random.seed(42)
torch.manual_seed(42)

print("=" * 65)
print("       PYTORCH DEEP MLP - FRAUD DETECTION")
print("=" * 65)

# ----------------------------------------------------------
# 1. DEVICE CONFIGURATION
# ----------------------------------------------------------

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("\n1. Computing device:", device)

# ----------------------------------------------------------
# 2. LOAD PROCESSED DATA
# ----------------------------------------------------------

print("\n2. Loading processed dataset...")

X_train = np.load(DATA_DIR / "X_train.npy").astype(np.float32)
X_val = np.load(DATA_DIR / "X_val.npy").astype(np.float32)

y_train = np.load(DATA_DIR / "y_train.npy").astype(np.float32)
y_val = np.load(DATA_DIR / "y_val.npy").astype(np.float32)

y_train = y_train.reshape(-1, 1)
y_val = y_val.reshape(-1, 1)

print("Training shape:", X_train.shape)
print("Validation shape:", X_val.shape)

# ----------------------------------------------------------
# 3. CONVERT DATA INTO PYTORCH TENSORS
# ----------------------------------------------------------

X_train_tensor = torch.tensor(X_train)
y_train_tensor = torch.tensor(y_train)

X_val_tensor = torch.tensor(X_val)
y_val_tensor = torch.tensor(y_val)

train_dataset = TensorDataset(
    X_train_tensor,
    y_train_tensor
)

train_loader = DataLoader(
    train_dataset,
    batch_size=2048,
    shuffle=True
)

# ----------------------------------------------------------
# 4. DEFINE DEEP NEURAL NETWORK
# ----------------------------------------------------------

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


input_size = X_train.shape[1]

model = FraudDetectionMLP(input_size).to(device)

print("\n3. Model architecture")
print(model)

# ----------------------------------------------------------
# 5. HANDLE CLASS IMBALANCE
# ----------------------------------------------------------

positive_count = np.sum(y_train == 1)
negative_count = np.sum(y_train == 0)

pos_weight = negative_count / max(positive_count, 1)

print("\n4. Class imbalance")
print("Normal transactions:", int(negative_count))
print("Fraudulent transactions:", int(positive_count))
print("Positive class weight:", round(float(pos_weight), 2))

criterion = nn.BCEWithLogitsLoss(
    pos_weight=torch.tensor(
        [pos_weight],
        dtype=torch.float32,
        device=device
    )
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=0.0001
)

# ----------------------------------------------------------
# 6. TRAINING
# ----------------------------------------------------------

epochs = 30

train_losses = []
val_losses = []

# Early stopping settings
best_val_loss = float("inf")
patience = 3
counter = 0
best_model_state = None

print("\n5. Starting deep learning training...")

for epoch in range(epochs):

    model.train()
    total_train_loss = 0.0

    for batch_X, batch_y in train_loader:

        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()

        logits = model(batch_X)
        loss = criterion(logits, batch_y)

        loss.backward()
        optimizer.step()

        total_train_loss += loss.item() * len(batch_X)

    train_loss = total_train_loss / len(train_dataset)

    # Validation
    model.eval()

    with torch.no_grad():

        val_logits = model(X_val_tensor.to(device))

        val_loss = criterion(
            val_logits,
            y_val_tensor.to(device)
        ).item()

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    print(
        f"Epoch {epoch + 1}/{epochs} | "
        f"Train Loss: {train_loss:.5f} | "
        f"Validation Loss: {val_loss:.5f}"
    )

    # Save the best model
    if val_loss < best_val_loss - 0.001:

        best_val_loss = val_loss
        counter = 0

        best_model_state = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }

    else:
        counter += 1

    # Stop if validation loss does not improve
    if counter >= patience:

        print("\nEarly stopping triggered!")
        break

# Restore the best model
if best_model_state is not None:

    model.load_state_dict(best_model_state)

    print("\nBest model restored.")
    print("Best validation loss:", round(best_val_loss, 5))
# ----------------------------------------------------------
# 7. VALIDATION EVALUATION
# ----------------------------------------------------------

print("\n6. Evaluating model...")

model.eval()

with torch.no_grad():

    val_logits = model(X_val_tensor.to(device))

    val_probabilities = torch.sigmoid(
        val_logits
    ).cpu().numpy().reshape(-1)

y_true = y_val.reshape(-1).astype(int)

roc_auc = roc_auc_score(
    y_true,
    val_probabilities
)

pr_auc = average_precision_score(
    y_true,
    val_probabilities
)

print("\nROC-AUC:", round(roc_auc, 4))
print("PR-AUC:", round(pr_auc, 4))

# ----------------------------------------------------------
# 8. FIND VALIDATION F1 THRESHOLD
# ----------------------------------------------------------

precision_values, recall_values, thresholds = precision_recall_curve(
    y_true,
    val_probabilities
)

f1_values = (
    2 * precision_values[:-1] * recall_values[:-1]
    / (
        precision_values[:-1]
        + recall_values[:-1]
        + 1e-12
    )
)

best_index = np.argmax(f1_values)
best_threshold = float(thresholds[best_index])

predictions = (
    val_probabilities >= best_threshold
).astype(int)

print("\n7. Selected validation threshold:", round(best_threshold, 5))

print("\nClassification report:")
print(
    classification_report(
        y_true,
        predictions,
        zero_division=0
    )
)

print("Confusion matrix:")
print(confusion_matrix(y_true, predictions))

# ----------------------------------------------------------
# 9. SAVE TRAINING LOSS GRAPH
# ----------------------------------------------------------

plt.figure(figsize=(9, 5))

plt.plot(train_losses, label="Training Loss")
plt.plot(val_losses, label="Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("PyTorch Deep MLP Training")

plt.legend()
plt.grid(True)

plt.savefig(
    OUTPUT_DIR / "pytorch_training_loss.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ----------------------------------------------------------
# 10. SAVE PRECISION-RECALL CURVE
# ----------------------------------------------------------

plt.figure(figsize=(9, 5))

plt.plot(
    recall_values,
    precision_values,
    label=f"PR-AUC = {pr_auc:.4f}"
)

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")

plt.legend()
plt.grid(True)

plt.savefig(
    OUTPUT_DIR / "pytorch_precision_recall.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ----------------------------------------------------------
# 11. SAVE CONFUSION MATRIX
# ----------------------------------------------------------

cm = confusion_matrix(y_true, predictions)

plt.figure(figsize=(6, 5))

plt.imshow(cm, interpolation="nearest")
plt.title("PyTorch MLP Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.xticks([0, 1], ["Normal", "Fraud"])
plt.yticks([0, 1], ["Normal", "Fraud"])

for i in range(2):
    for j in range(2):
        plt.text(
            j, i, str(cm[i, j]),
            ha="center",
            va="center"
        )

plt.colorbar()
plt.savefig(
    OUTPUT_DIR / "pytorch_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ----------------------------------------------------------
# 12. SAVE MODEL AND RESULTS
# ----------------------------------------------------------

torch.save(
    model.state_dict(),
    MODEL_DIR / "pytorch_mlp.pth"
)

results = {
    "model": "PyTorch Deep MLP",
    "device": str(device),
    "epochs": epochs,
    "roc_auc": float(roc_auc),
    "pr_auc": float(pr_auc),
    "best_validation_threshold": best_threshold
}

with open(
    OUTPUT_DIR / "pytorch_mlp_results.json",
    "w"
) as file:

    json.dump(results, file, indent=4)

print("\n8. Files saved successfully:")
print("models/pytorch_mlp.pth")
print("outputs/pytorch_mlp_results.json")
print("outputs/pytorch_training_loss.png")
print("outputs/pytorch_precision_recall.png")
print("outputs/pytorch_confusion_matrix.png")

print("\n" + "=" * 65)
print("       PYTORCH DEEP MLP COMPLETED!")
print("=" * 65)
