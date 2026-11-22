
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# ==================================================
# SYNTHETIC FRAUD DATA - QUALITY ANALYSIS
# ==================================================

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "processed"
OUTPUT_DIR = PROJECT_DIR / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)

print("=" * 60)
print("       SYNTHETIC FRAUD DATA ANALYSIS")
print("=" * 60)

# --------------------------------------------------
# 1. LOAD ORIGINAL AND SYNTHETIC DATA
# --------------------------------------------------

print("\n1. Loading datasets...")

X_train = np.load(DATA_DIR / "X_train.npy")
y_train = np.load(DATA_DIR / "y_train.npy").reshape(-1)

original_fraud = X_train[y_train == 1]

synthetic_fraud = np.load(
    OUTPUT_DIR / "synthetic_fraud.npy"
)

print("Original fraud shape:", original_fraud.shape)
print("Synthetic fraud shape:", synthetic_fraud.shape)

# --------------------------------------------------
# 2. CHECK DATA QUALITY
# --------------------------------------------------

print("\n2. Checking data quality...")

nan_count = int(np.isnan(synthetic_fraud).sum())
infinite_count = int(np.isinf(synthetic_fraud).sum())

duplicate_count = int(
    len(synthetic_fraud) -
    len(np.unique(synthetic_fraud, axis=0))
)

print("NaN values:", nan_count)
print("Infinite values:", infinite_count)
print("Duplicate synthetic rows:", duplicate_count)

# --------------------------------------------------
# 3. COMPARE FEATURE STATISTICS
# --------------------------------------------------

print("\n3. Comparing feature statistics...")

original_mean = np.mean(original_fraud, axis=0)
synthetic_mean = np.mean(synthetic_fraud, axis=0)

original_std = np.std(original_fraud, axis=0)
synthetic_std = np.std(synthetic_fraud, axis=0)

mean_difference = np.mean(
    np.abs(original_mean - synthetic_mean)
)

std_difference = np.mean(
    np.abs(original_std - synthetic_std)
)

print("Average absolute mean difference:",
      round(float(mean_difference), 4))

print("Average absolute std difference:",
      round(float(std_difference), 4))

# --------------------------------------------------
# 4. PCA VISUALIZATION
# --------------------------------------------------

print("\n4. Creating PCA visualization...")

combined = np.vstack([
    original_fraud,
    synthetic_fraud
])

pca = PCA(n_components=2)

reduced = pca.fit_transform(combined)

original_count = len(original_fraud)

original_2d = reduced[:original_count]
synthetic_2d = reduced[original_count:]

plt.figure(figsize=(10, 7))

plt.scatter(
    original_2d[:, 0],
    original_2d[:, 1],
    label="Original Fraud",
    alpha=0.6,
    s=20
)

plt.scatter(
    synthetic_2d[:, 0],
    synthetic_2d[:, 1],
    label="Synthetic Fraud",
    alpha=0.4,
    s=15
)

plt.title("Original vs Synthetic Fraud Transactions (PCA)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.grid(alpha=0.2)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "synthetic_fraud_pca.png",
    dpi=300
)

plt.close()

# --------------------------------------------------
# 5. FEATURE DISTRIBUTION COMPARISON
# --------------------------------------------------

print("\n5. Creating feature comparison graph...")

features_to_plot = [0, 1, 2, 3]

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

for i, ax in zip(features_to_plot, axes.flatten()):

    ax.hist(
        original_fraud[:, i],
        bins=30,
        alpha=0.6,
        label="Original"
    )

    ax.hist(
        synthetic_fraud[:, i],
        bins=30,
        alpha=0.6,
        label="Synthetic"
    )

    ax.set_title(f"Feature {i}")
    ax.legend()

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "synthetic_feature_comparison.png",
    dpi=300
)

plt.close()

# --------------------------------------------------
# 6. SAVE NUMERICAL RESULTS
# --------------------------------------------------

results = {
    "original_fraud_samples": int(len(original_fraud)),
    "synthetic_fraud_samples": int(len(synthetic_fraud)),
    "nan_values": nan_count,
    "infinite_values": infinite_count,
    "duplicate_synthetic_rows": duplicate_count,
    "mean_difference": float(mean_difference),
    "std_difference": float(std_difference),
    "pca_explained_variance": (
        pca.explained_variance_ratio_.tolist()
    )
}

with open(
    OUTPUT_DIR / "synthetic_fraud_analysis.json",
    "w"
) as file:

    json.dump(results, file, indent=4)

print("\n6. Analysis completed!")

print("Saved files:")
print("outputs/synthetic_fraud_pca.png")
print("outputs/synthetic_feature_comparison.png")
print("outputs/synthetic_fraud_analysis.json")

print("\n" + "=" * 60)
print("       SYNTHETIC DATA ANALYSIS COMPLETED!")
print("=" * 60)
