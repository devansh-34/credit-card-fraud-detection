
from pathlib import Path
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve
)

# ==========================================
# 1. PROJECT CONFIGURATION
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "processed"
MODEL_DIR = PROJECT_DIR / "models"
OUTPUT_DIR = PROJECT_DIR / "outputs"

MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

torch.manual_seed(42)
np.random.seed(42)

device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("=" * 60)
print("       LSTM CREDIT CARD FRAUD DETECTION")
print("=" * 60)
print("Using device:", device)

# ==========================================
# 2. LOAD SEQUENCE DATA
# ==========================================

print("\n1. Loading LSTM datasets...")

def load_array(filename):
    return np.load(DATA_DIR / filename)

X_train = load_array("X_train_lstm.npy")
y_train = load_array("y_train_lstm.npy")

X_val = load_array("X_val_lstm.npy")
y_val = load_array("y_val_lstm.npy")

X_test = load_array("X_test_lstm.npy")
y_test = load_array("y_test_lstm.npy")

print("Training:", X_train.shape)
print("Validation:", X_val.shape)
print("Testing:", X_test.shape)

# ==========================================
# 3. CONVERT DATA TO PYTORCH TENSORS
# ==========================================

print("\n2. Converting data to tensors...")

def make_loader(X, y, batch_size, shuffle=False):

    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0
    )


BATCH_SIZE = 1024

train_loader = make_loader(
    X_train, y_train, BATCH_SIZE, shuffle=True
)

val_loader = make_loader(
    X_val, y_val, BATCH_SIZE
)

test_loader = make_loader(
    X_test, y_test, BATCH_SIZE
)

# ==========================================
# 4. DEFINE LSTM NEURAL NETWORK
# ==========================================

print("\n3. Initializing LSTM model...")

class FraudLSTM(nn.Module):

    def __init__(
        self,
        input_size=30,
        hidden_size=32,
        num_layers=2
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )

        self.dropout = nn.Dropout(0.2)

        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):

        lstm_output, (hidden, cell) = self.lstm(x)

        # Use the final time-step output
        final_output = lstm_output[:, -1, :]

        final_output = self.dropout(final_output)

        logits = self.fc(final_output)

        return logits


model = FraudLSTM().to(device)

print(model)

# ==========================================
# 5. HANDLE CLASS IMBALANCE
# ==========================================

print("\n4. Handling fraud class imbalance...")

positive_count = float(np.sum(y_train == 1))
negative_count = float(np.sum(y_train == 0))

pos_weight = torch.tensor(
    [negative_count / positive_count],
    dtype=torch.float32,
    device=device
)

print("Normal transactions:", int(negative_count))
print("Fraud transactions:", int(positive_count))
print("Positive class weight:", round(pos_weight.item(), 2))

criterion = nn.BCEWithLogitsLoss(
    pos_weight=pos_weight
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.001,
    weight_decay=0.0001
)

# ==========================================
# 6. TRAINING FUNCTION
# ==========================================

def run_epoch(loader, training=True):

    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0

    for X_batch, y_batch in loader:

        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(training):

            logits = model(X_batch)

            loss = criterion(logits, y_batch)

            if training:
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=1.0
                )

                optimizer.step()

        total_loss += loss.item() * len(X_batch)

    return total_loss / len(loader.dataset)


# ==========================================
# 7. TRAIN THE LSTM
# ==========================================

print("\n5. Starting LSTM training...")

EPOCHS = 12

train_losses = []
val_losses = []

best_val_loss = float("inf")
best_model_path = MODEL_DIR / "lstm_model.pth"

for epoch in range(EPOCHS):

    train_loss = run_epoch(
        train_loader, training=True
    )

    val_loss = run_epoch(
        val_loader, training=False
    )

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        torch.save(
            model.state_dict(),
            best_model_path
        )

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Train Loss: {train_loss:.5f} | "
        f"Validation Loss: {val_loss:.5f}"
    )

# ==========================================
# 8. LOAD BEST MODEL
# ==========================================

print("\n6. Loading best model...")

model.load_state_dict(
    torch.load(best_model_path, map_location=device)
)

# ==========================================
# 9. GET PREDICTIONS
# ==========================================

def get_predictions(loader):

    model.eval()

    probabilities = []
    actual = []

    with torch.no_grad():

        for X_batch, y_batch in loader:

            X_batch = X_batch.to(device)

            logits = model(X_batch)

            probs = torch.sigmoid(logits)

            probabilities.extend(
                probs.cpu().numpy().flatten()
            )

            actual.extend(
                y_batch.numpy().flatten()
            )

    return np.array(actual), np.array(probabilities)


# ==========================================
# 10. VALIDATION THRESHOLD SELECTION
# ==========================================

print("\n7. Selecting threshold using validation data...")

val_actual, val_probs = get_predictions(val_loader)

precision_values, recall_values, thresholds = (
    precision_recall_curve(val_actual, val_probs)
)

f1_values = (
    2 * precision_values[:-1] * recall_values[:-1]
    / np.maximum(
        precision_values[:-1] + recall_values[:-1],
        1e-12
    )
)

best_index = int(np.argmax(f1_values))

best_threshold = float(thresholds[best_index])

print("Selected threshold:", round(best_threshold, 6))
print("Validation F1:", round(float(f1_values[best_index]), 4))

# ==========================================
# 11. FINAL TEST EVALUATION
# ==========================================

print("\n8. Evaluating on unseen test data...")

test_actual, test_probs = get_predictions(test_loader)

test_predictions = (
    test_probs >= best_threshold
).astype(int)

roc_auc = roc_auc_score(
    test_actual, test_probs
)

pr_auc = average_precision_score(
    test_actual, test_probs
)

print("\nROC-AUC:", round(roc_auc, 4))
print("PR-AUC:", round(pr_auc, 4))

print("\nClassification Report:")

report = classification_report(
    test_actual,
    test_predictions,
    zero_division=0
)

print(report)

cm = confusion_matrix(
    test_actual, test_predictions
)

print("Confusion Matrix:")
print(cm)

# ==========================================
# 12. SAVE TRAINING GRAPH
# ==========================================

plt.figure(figsize=(9, 5))

plt.plot(
    range(1, EPOCHS + 1),
    train_losses,
    label="Training Loss"
)

plt.plot(
    range(1, EPOCHS + 1),
    val_losses,
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("LSTM Training and Validation Loss")
plt.legend()
plt.grid(True)

plt.savefig(
    OUTPUT_DIR / "lstm_training_loss.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ==========================================
# 13. SAVE CONFUSION MATRIX
# ==========================================

plt.figure(figsize=(6, 5))

plt.imshow(cm, interpolation="nearest")

plt.title("LSTM Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")

plt.xticks([0, 1], ["Normal", "Fraud"])
plt.yticks([0, 1], ["Normal", "Fraud"])

for i in range(2):
    for j in range(2):
        plt.text(
            j, i, str(cm[i, j]),
            ha="center", va="center"
        )

plt.colorbar()

plt.savefig(
    OUTPUT_DIR / "lstm_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ==========================================
# 14. SAVE RESULTS
# ==========================================

results = {
    "model": "LSTM",
    "sequence_length": 10,
    "input_features": 30,
    "roc_auc": float(roc_auc),
    "pr_auc": float(pr_auc),
    "threshold": best_threshold,
    "confusion_matrix": cm.tolist(),
    "classification_report": report,
    "validation_f1": float(f1_values[best_index])
}

with open(
    OUTPUT_DIR / "lstm_results.json", "w"
) as file:

    json.dump(results, file, indent=4)

print("\n9. Files saved successfully:")

print("models/lstm_model.pth")
print("outputs/lstm_training_loss.png")
print("outputs/lstm_confusion_matrix.png")
print("outputs/lstm_results.json")

print("\n" + "=" * 60)
print("       LSTM TRAINING COMPLETED!")
print("=" * 60)
