
from pathlib import Path
import json
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "dataset" / "creditcard.csv"
OUTPUT_DIR = PROJECT_DIR / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("       CREDIT CARD FRAUD - DATA PREPROCESSING")
print("=" * 60)

# --------------------------------------------------
# 2. LOAD ORIGINAL DATASET
# --------------------------------------------------

print("\n1. Loading original dataset...")

df = pd.read_csv(DATA_PATH)

print("Original shape:", df.shape)

# --------------------------------------------------
# 3. REMOVE EXACT DUPLICATES
# --------------------------------------------------

print("\n2. Removing exact duplicate rows...")

duplicate_count = df.duplicated().sum()

df = df.drop_duplicates().reset_index(drop=True)

print("Duplicates removed:", duplicate_count)
print("New dataset shape:", df.shape)

# --------------------------------------------------
# 4. SEPARATE FEATURES AND TARGET
# --------------------------------------------------

print("\n3. Separating features and target...")

X = df.drop(columns=["Class"])
y = df["Class"]

feature_names = X.columns.tolist()

print("Number of input features:", X.shape[1])
print("Target column: Class")

# --------------------------------------------------
# 5. SPLIT INTO TRAIN, VALIDATION AND TEST
# --------------------------------------------------

print("\n4. Splitting dataset...")

# 70% training, 30% temporary
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

# Split remaining 30% equally:
# 15% validation and 15% testing
X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

print("Training samples:", len(X_train))
print("Validation samples:", len(X_val))
print("Testing samples:", len(X_test))

# --------------------------------------------------
# 6. FEATURE SCALING
# --------------------------------------------------

print("\n5. Applying StandardScaler...")

scaler = StandardScaler()

# Learn scaling parameters ONLY from training data
X_train_scaled = scaler.fit_transform(X_train)

# Apply the same parameters to validation and test data
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print("Feature scaling completed.")

# --------------------------------------------------
# 7. SAVE PROCESSED DATA
# --------------------------------------------------

print("\n6. Saving processed data...")

np.save(OUTPUT_DIR / "X_train.npy", X_train_scaled)
np.save(OUTPUT_DIR / "X_val.npy", X_val_scaled)
np.save(OUTPUT_DIR / "X_test.npy", X_test_scaled)

np.save(OUTPUT_DIR / "y_train.npy", y_train.to_numpy())
np.save(OUTPUT_DIR / "y_val.npy", y_val.to_numpy())
np.save(OUTPUT_DIR / "y_test.npy", y_test.to_numpy())

joblib.dump(scaler, OUTPUT_DIR / "scaler.pkl")

with open(OUTPUT_DIR / "feature_names.json", "w") as file:
    json.dump(feature_names, file, indent=4)

# --------------------------------------------------
# 8. DISPLAY CLASS DISTRIBUTION
# --------------------------------------------------

print("\n7. CLASS DISTRIBUTION")

for name, labels in [
    ("Training", y_train),
    ("Validation", y_val),
    ("Testing", y_test)
]:
    print(f"\n{name}:")
    print("Normal:", int((labels == 0).sum()))
    print("Fraud:", int((labels == 1).sum()))

# --------------------------------------------------
# 9. FINAL SUMMARY
# --------------------------------------------------

print("\n8. FINAL DATA SHAPES")

print("X_train:", X_train_scaled.shape)
print("X_val:", X_val_scaled.shape)
print("X_test:", X_test_scaled.shape)

print("\nAll processed files saved in:", OUTPUT_DIR)

print("\n" + "=" * 60)
print("       PREPROCESSING COMPLETED SUCCESSFULLY!")
print("=" * 60)
