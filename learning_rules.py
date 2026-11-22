
from pathlib import Path
import numpy as np

# ==========================================================
# CREDIT CARD FRAUD DETECTION
# IMPLEMENTATION OF FOUR LEARNING RULES
# ==========================================================

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "processed"
OUTPUT_DIR = PROJECT_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

np.random.seed(42)

print("=" * 65)
print("       NEURAL NETWORK LEARNING RULES")
print("=" * 65)

# ----------------------------------------------------------
# 1. LOAD TRAINING DATA
# ----------------------------------------------------------

X = np.load(DATA_DIR / "X_train.npy").astype(np.float32)
y = np.load(DATA_DIR / "y_train.npy").astype(np.int32).reshape(-1)

# Select equal numbers of fraud and normal transactions
fraud_indices = np.where(y == 1)[0]
normal_indices = np.where(y == 0)[0]

sample_count = min(500, len(fraud_indices), len(normal_indices))

selected_indices = np.concatenate([
    np.random.choice(fraud_indices, sample_count, replace=False),
    np.random.choice(normal_indices, sample_count, replace=False)
])

np.random.shuffle(selected_indices)

X = X[selected_indices]
y = y[selected_indices]

# Normalize features
X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
# Normalize each feature for stable learning
X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

print("\nDataset loaded:", X.shape)
print("Fraud samples:", int(np.sum(y == 1)))
print("Normal samples:", int(np.sum(y == 0)))


# ----------------------------------------------------------
# 2. HEBBIAN LEARNING
# "Neurons that fire together wire together"
# ----------------------------------------------------------

print("\n1. HEBBIAN LEARNING")

def hebbian_learning(data, learning_rate=0.00001, epochs=10):

    weights = (
        np.random.randn(data.shape[1]) * 0.01
    ).astype(np.float32)

    for _ in range(epochs):

        for sample in data:

            output = np.dot(sample, weights)

            weights += learning_rate * output * sample

            # Prevent numerical instability
            weights = np.clip(weights, -5, 5)

    return weights

# Use a small sample to keep the experiment efficient
hebbian_weights = hebbian_learning(X[:500])

print("Hebbian learning completed")
print("Weight vector shape:", hebbian_weights.shape)
print("Weight norm:", round(float(np.linalg.norm(hebbian_weights)), 4))


# ----------------------------------------------------------
# 3. MODIFIED HEBBIAN LEARNING (OJA'S RULE)
# A normalized form of Hebbian learning
# ----------------------------------------------------------

print("\n2. MODIFIED HEBBIAN LEARNING - OJA'S RULE")

def oja_learning(data, learning_rate=0.001, epochs=20):

    weights = np.random.randn(data.shape[1]).astype(np.float32)
    weights /= np.linalg.norm(weights) + 1e-8

    for _ in range(epochs):
        for sample in data[:500]:
            output = np.dot(sample, weights)

            weights += learning_rate * (
                output * sample - (output ** 2) * weights
            )

    return weights


oja_weights = oja_learning(X)

print("Oja learning completed")
print("Weight vector shape:", oja_weights.shape)
print("Weight norm:", round(float(np.linalg.norm(oja_weights)), 4))


# ----------------------------------------------------------
# 4. COMPETITIVE LEARNING
# Winner-takes-all clustering
# ----------------------------------------------------------

print("\n3. COMPETITIVE LEARNING")

def competitive_learning(data, clusters=3, epochs=10, learning_rate=0.05):

    indices = np.random.choice(len(data), clusters, replace=False)
    centers = data[indices].copy()

    for _ in range(epochs):

        for sample in data:

            distances = np.linalg.norm(centers - sample, axis=1)

            winner = np.argmin(distances)

            centers[winner] += learning_rate * (
                sample - centers[winner]
            )

    return centers


cluster_centers = competitive_learning(X)

print("Competitive learning completed")
print("Number of clusters:", len(cluster_centers))
print("Cluster center shape:", cluster_centers.shape)


# ----------------------------------------------------------
# 5. ERROR-CORRECTION LEARNING
# Supervised learning using prediction error
# ----------------------------------------------------------

print("\n4. ERROR-CORRECTION LEARNING")

def error_correction_learning(data, labels, learning_rate=0.01, epochs=10):

    weights = np.zeros(data.shape[1], dtype=np.float32)
    bias = 0.0

    # Balance the small training sample
    fraud_indices = np.where(labels == 1)[0]
    normal_indices = np.where(labels == 0)[0]

    if len(fraud_indices) == 0:
        raise ValueError("No fraud samples in the selected training data.")

    sample_count = min(len(fraud_indices), len(normal_indices))

    selected = np.concatenate([
        np.random.choice(fraud_indices, sample_count, replace=False),
        np.random.choice(normal_indices, sample_count, replace=False)
    ])

    np.random.shuffle(selected)

    train_X = data[selected]
    train_y = labels[selected]

    for _ in range(epochs):

        for sample, target in zip(train_X, train_y):

            z = np.dot(sample, weights) + bias
            prediction = 1 / (1 + np.exp(-np.clip(z, -50, 50)))

            error = target - prediction

            weights += learning_rate * error * sample
            bias += learning_rate * error

    return weights, bias


error_weights, error_bias = error_correction_learning(X, y)

print("Error-correction learning completed")
print("Weight vector shape:", error_weights.shape)
print("Bias:", round(float(error_bias), 6))


# ----------------------------------------------------------
# 6. SAVE EXPERIMENT RESULTS
# ----------------------------------------------------------

np.savez(
    OUTPUT_DIR / "learning_rules_results.npz",
    hebbian_weights=hebbian_weights,
    oja_weights=oja_weights,
    cluster_centers=cluster_centers,
    error_weights=error_weights,
    error_bias=error_bias
)

print("\nResults saved:")
print("outputs/learning_rules_results.npz")

print("\n" + "=" * 65)
print("       ALL FOUR LEARNING RULES COMPLETED!")
print("=" * 65)
