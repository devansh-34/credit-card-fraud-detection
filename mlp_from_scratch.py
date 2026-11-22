
from pathlib import Path
import numpy as np

# ==================================================
# CREDIT CARD FRAUD DETECTION
# MLP IMPLEMENTED FROM SCRATCH USING NUMPY
# ==================================================

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "processed"
MODEL_DIR = PROJECT_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

print("=" * 60)
print("       MLP FROM SCRATCH - FRAUD DETECTION")
print("=" * 60)

# --------------------------------------------------
# 1. LOAD PREPROCESSED DATA
# --------------------------------------------------

print("\n1. Loading processed data...")

X_train = np.load(DATA_DIR / "X_train.npy").astype(np.float32)
X_val = np.load(DATA_DIR / "X_val.npy").astype(np.float32)

y_train = np.load(DATA_DIR / "y_train.npy").astype(np.float32).reshape(-1, 1)
y_val = np.load(DATA_DIR / "y_val.npy").astype(np.float32).reshape(-1, 1)

print("Training data:", X_train.shape)
print("Validation data:", X_val.shape)

# --------------------------------------------------
# 2. DEFINE ACTIVATION FUNCTIONS
# --------------------------------------------------

def relu(x):
    return np.maximum(0, x)


def sigmoid(x):
    x = np.clip(x, -50, 50)
    return 1 / (1 + np.exp(-x))


# --------------------------------------------------
# 3. INITIALIZE NETWORK PARAMETERS
# --------------------------------------------------

np.random.seed(42)

input_size = 30
hidden_size = 16
output_size = 1

W1 = (
    np.random.randn(input_size, hidden_size)
    * np.sqrt(2 / input_size)
).astype(np.float32)

b1 = np.zeros((1, hidden_size), dtype=np.float32)

W2 = (
    np.random.randn(hidden_size, output_size)
    * np.sqrt(2 / hidden_size)
).astype(np.float32)

b2 = np.zeros((1, output_size), dtype=np.float32)

print("\n2. Neural network initialized")
print("Input neurons:", input_size)
print("Hidden neurons:", hidden_size)
print("Output neurons:", output_size)

# --------------------------------------------------
# 4. CLASS WEIGHTS FOR IMBALANCED DATA
# --------------------------------------------------

positive_count = np.sum(y_train == 1)
negative_count = np.sum(y_train == 0)

positive_weight = negative_count / positive_count

print("\n3. Handling class imbalance")
print("Normal transactions:", int(negative_count))
print("Fraudulent transactions:", int(positive_count))
print("Fraud class weight:", round(float(positive_weight), 2))


# --------------------------------------------------
# 5. FORWARD PROPAGATION
# --------------------------------------------------

def forward(X):

    Z1 = np.dot(X, W1) + b1
    A1 = relu(Z1)

    Z2 = np.dot(A1, W2) + b2
    A2 = sigmoid(Z2)

    cache = (X, Z1, A1, Z2, A2)

    return A2, cache


# --------------------------------------------------
# 6. WEIGHTED BINARY CROSS-ENTROPY LOSS
# --------------------------------------------------

def calculate_loss(y, predictions):

    predictions = np.clip(
        predictions, 1e-7, 1 - 1e-7
    )

    weights = np.where(
        y == 1, positive_weight, 1.0
    )

    loss = -np.mean(
        weights * (
            y * np.log(predictions)
            + (1 - y) * np.log(1 - predictions)
        )
    )

    return loss


# --------------------------------------------------
# 7. BACKPROPAGATION
# --------------------------------------------------

def backward(y, cache):

    X, Z1, A1, Z2, A2 = cache

    batch_size = X.shape[0]

    sample_weights = np.where(
        y == 1, positive_weight, 1.0
    )

    # Output layer gradients
    dZ2 = (A2 - y) * sample_weights / batch_size

    dW2 = np.dot(A1.T, dZ2)
    db2 = np.sum(dZ2, axis=0, keepdims=True)

    # Hidden layer gradients
    dA1 = np.dot(dZ2, W2.T)

    dZ1 = dA1 * (Z1 > 0)

    dW1 = np.dot(X.T, dZ1)
    db1 = np.sum(dZ1, axis=0, keepdims=True)

    return dW1, db1, dW2, db2


# --------------------------------------------------
# 8. TRAINING USING MINI-BATCH GRADIENT DESCENT
# --------------------------------------------------

learning_rate = 0.001
epochs = 10
batch_size = 2048

print("\n4. Starting neural network training...")

for epoch in range(epochs):

    indices = np.random.permutation(len(X_train))

    X_shuffled = X_train[indices]
    y_shuffled = y_train[indices]

    for start in range(0, len(X_train), batch_size):

        end = start + batch_size

        X_batch = X_shuffled[start:end]
        y_batch = y_shuffled[start:end]

        # Forward propagation
        predictions, cache = forward(X_batch)

        # Backpropagation
        dW1, db1, dW2, db2 = backward(
            y_batch, cache
        )

        # Gradient descent updates
        W1[:] -= learning_rate * dW1
        b1[:] -= learning_rate * db1

        W2[:] -= learning_rate * dW2
        b2[:] -= learning_rate * db2

    # Calculate full training and validation loss
    train_predictions, _ = forward(X_train)
    val_predictions, _ = forward(X_val)

    train_loss = calculate_loss(
        y_train, train_predictions
    )

    val_loss = calculate_loss(
        y_val, val_predictions
    )

    print(
        f"Epoch {epoch + 1}/{epochs} | "
        f"Train Loss: {train_loss:.5f} | "
        f"Validation Loss: {val_loss:.5f}"
    )


# --------------------------------------------------
# 9. VALIDATION PREDICTIONS
# --------------------------------------------------

print("\n5. Validation evaluation")

val_predictions, _ = forward(X_val)

predicted_classes = (
    val_predictions >= 0.5
).astype(int)

actual_classes = y_val.astype(int)

true_positive = int(
    np.sum((predicted_classes == 1) & (actual_classes == 1))
)

true_negative = int(
    np.sum((predicted_classes == 0) & (actual_classes == 0))
)

false_positive = int(
    np.sum((predicted_classes == 1) & (actual_classes == 0))
)

false_negative = int(
    np.sum((predicted_classes == 0) & (actual_classes == 1))
)

print("True Positives:", true_positive)
print("True Negatives:", true_negative)
print("False Positives:", false_positive)
print("False Negatives:", false_negative)

precision = true_positive / max(
    true_positive + false_positive, 1
)

recall = true_positive / max(
    true_positive + false_negative, 1
)

f1_score = (
    2 * precision * recall /
    max(precision + recall, 1e-12)
)

print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1 Score:", round(f1_score, 4))


# --------------------------------------------------
# 10. SAVE MODEL PARAMETERS
# --------------------------------------------------

np.savez(
    MODEL_DIR / "mlp_from_scratch.npz",
    W1=W1,
    b1=b1,
    W2=W2,
    b2=b2
)

print("\n6. Model saved successfully!")
print("Location: models/mlp_from_scratch.npz")

print("\n" + "=" * 60)
print("       MLP TRAINING COMPLETED!")
print("=" * 60)
