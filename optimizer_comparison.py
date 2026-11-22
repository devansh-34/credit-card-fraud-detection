
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

# =========================================================
# CREDIT CARD FRAUD DETECTION
# COMPARISON: SGD, MOMENTUM AND NESTEROV
# =========================================================

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "processed"
OUTPUT_DIR = PROJECT_DIR / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)

np.random.seed(42)

print("=" * 65)
print("       OPTIMIZER COMPARISON - FRAUD DETECTION")
print("=" * 65)

# ---------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------

print("\n1. Loading processed data...")

X_train = np.load(DATA_DIR / "X_train.npy").astype(np.float32)
X_val = np.load(DATA_DIR / "X_val.npy").astype(np.float32)

y_train = np.load(DATA_DIR / "y_train.npy").astype(np.float32).reshape(-1, 1)
y_val = np.load(DATA_DIR / "y_val.npy").astype(np.float32).reshape(-1, 1)

print("Training shape:", X_train.shape)
print("Validation shape:", X_val.shape)

# ---------------------------------------------------------
# 2. NETWORK CONFIGURATION
# ---------------------------------------------------------

input_size = X_train.shape[1]
hidden_size = 16
output_size = 1

learning_rate = 0.001
epochs = 5
batch_size = 2048
momentum = 0.9

positive_count = np.sum(y_train == 1)
negative_count = np.sum(y_train == 0)

positive_weight = negative_count / positive_count

print("\n2. Network configuration")
print("Input neurons:", input_size)
print("Hidden neurons:", hidden_size)
print("Learning rate:", learning_rate)
print("Epochs:", epochs)
print("Batch size:", batch_size)
print("Fraud class weight:", round(float(positive_weight), 2))

# ---------------------------------------------------------
# 3. ACTIVATION FUNCTIONS
# ---------------------------------------------------------

def relu(x):
    return np.maximum(0, x)


def sigmoid(x):
    x = np.clip(x, -50, 50)
    return 1 / (1 + np.exp(-x))


# ---------------------------------------------------------
# 4. INITIALIZE MODEL
# ---------------------------------------------------------

def initialize_parameters():

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

    return {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2
    }


# ---------------------------------------------------------
# 5. FORWARD PROPAGATION
# ---------------------------------------------------------

def forward(X, params):

    Z1 = X @ params["W1"] + params["b1"]
    A1 = relu(Z1)

    Z2 = A1 @ params["W2"] + params["b2"]
    A2 = sigmoid(Z2)

    cache = (X, Z1, A1, A2)

    return A2, cache


# ---------------------------------------------------------
# 6. LOSS FUNCTION
# ---------------------------------------------------------

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

    return float(loss)


# ---------------------------------------------------------
# 7. BACKPROPAGATION
# ---------------------------------------------------------

def backward(y, cache, params):

    X, Z1, A1, A2 = cache

    batch_size_actual = X.shape[0]

    sample_weights = np.where(
        y == 1, positive_weight, 1.0
    )

    dZ2 = (
        (A2 - y) * sample_weights / batch_size_actual
    )

    dW2 = A1.T @ dZ2
    db2 = np.sum(dZ2, axis=0, keepdims=True)

    dA1 = dZ2 @ params["W2"].T
    dZ1 = dA1 * (Z1 > 0)

    dW1 = X.T @ dZ1
    db1 = np.sum(dZ1, axis=0, keepdims=True)

    return {
        "W1": dW1,
        "b1": db1,
        "W2": dW2,
        "b2": db2
    }


# ---------------------------------------------------------
# 8. TRAINING FUNCTION
# ---------------------------------------------------------

def train_optimizer(optimizer_name):

    print("\n" + "-" * 55)
    print("Training with:", optimizer_name)
    print("-" * 55)

    # Identical starting weights for fair comparison
    np.random.seed(42)
    params = initialize_parameters()

    velocities = {
        key: np.zeros_like(value)
        for key, value in params.items()
    }

    for epoch in range(epochs):

        indices = np.random.permutation(len(X_train))

        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]

        for start in range(0, len(X_train), batch_size):

            end = start + batch_size

            X_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]

            # Nesterov: calculate gradient at look-ahead weights
            if optimizer_name == "Nesterov":

                lookahead_params = {
                    key: params[key] - (
                        learning_rate * momentum * velocities[key]
                    )
                    for key in params
                }

                predictions, cache = forward(
                    X_batch, lookahead_params
                )

                gradients = backward(
                    y_batch, cache, lookahead_params
                )

            else:

                predictions, cache = forward(
                    X_batch, params
                )

                gradients = backward(
                    y_batch, cache, params
                )

            # Apply selected optimizer
            for key in params:

                if optimizer_name == "SGD":

                    params[key] -= (
                        learning_rate * gradients[key]
                    )

                elif optimizer_name == "Momentum":

                    velocities[key] = (
                        momentum * velocities[key]
                        + gradients[key]
                    )

                    params[key] -= (
                        learning_rate * velocities[key]
                    )

                elif optimizer_name == "Nesterov":

                    velocities[key] = (
                        momentum * velocities[key]
                        + gradients[key]
                    )

                    params[key] -= (
                        learning_rate * velocities[key]
                    )

        # Evaluate after each epoch
        train_predictions, _ = forward(X_train, params)
        val_predictions, _ = forward(X_val, params)

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

    # Final validation metrics
    val_predictions, _ = forward(X_val, params)

    roc_auc = roc_auc_score(
        y_val.ravel(), val_predictions.ravel()
    )

    pr_auc = average_precision_score(
        y_val.ravel(), val_predictions.ravel()
    )

    print("\nFinal validation results")
    print("ROC-AUC:", round(roc_auc, 4))
    print("PR-AUC:", round(pr_auc, 4))

    return {
        "optimizer": optimizer_name,
        "train_loss": train_loss,
        "validation_loss": val_loss,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc
    }


# ---------------------------------------------------------
# 9. COMPARE OPTIMIZERS
# ---------------------------------------------------------

print("\n3. Starting optimizer comparison...")

results = []

for optimizer in ["SGD", "Momentum", "Nesterov"]:

    result = train_optimizer(optimizer)
    results.append(result)


# ---------------------------------------------------------
# 10. DISPLAY COMPARISON
# ---------------------------------------------------------

print("\n" + "=" * 75)
print("                 OPTIMIZER COMPARISON RESULTS")
print("=" * 75)

print(
    f"{'Optimizer':<15}"
    f"{'Train Loss':<15}"
    f"{'Val Loss':<15}"
    f"{'ROC-AUC':<12}"
    f"{'PR-AUC':<12}"
)

for result in results:

    print(
        f"{result['optimizer']:<15}"
        f"{result['train_loss']:<15.5f}"
        f"{result['validation_loss']:<15.5f}"
        f"{result['roc_auc']:<12.4f}"
        f"{result['pr_auc']:<12.4f}"
    )

# Save results
import json

with open(
    OUTPUT_DIR / "optimizer_comparison.json",
    "w"
) as file:

    json.dump(results, file, indent=4)

print("\nResults saved to:")
print("outputs/optimizer_comparison.json")

print("\n" + "=" * 65)
print("             OPTIMIZER COMPARISON COMPLETED!")
print("=" * 65)
