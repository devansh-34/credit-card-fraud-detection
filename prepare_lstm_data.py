
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler

# ==========================================
# LSTM DATA PREPARATION
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parent

DATA_PATH = PROJECT_DIR / "dataset" / "creditcard.csv"
OUTPUT_DIR = PROJECT_DIR / "processed"

OUTPUT_DIR.mkdir(exist_ok=True)

print("=" * 60)
print("       LSTM SEQUENCE DATA PREPARATION")
print("=" * 60)

# 1. Load dataset
print("\n1. Loading dataset...")

df = pd.read_csv(DATA_PATH)
df = df.drop_duplicates().reset_index(drop=True)

print("Dataset shape:", df.shape)

# 2. Sort transactions chronologically
print("\n2. Sorting transactions by time...")

df = df.sort_values("Time").reset_index(drop=True)

X = df.drop(columns=["Class"])
y = df["Class"].to_numpy()

# 3. Chronological split
print("\n3. Creating chronological splits...")

n = len(df)

train_end = int(n * 0.70)
val_end = int(n * 0.85)

X_train = X.iloc[:train_end]
X_val = X.iloc[train_end:val_end]
X_test = X.iloc[val_end:]

y_train = y[:train_end]
y_val = y[train_end:val_end]
y_test = y[val_end:]

print("Training rows:", len(X_train))
print("Validation rows:", len(X_val))
print("Testing rows:", len(X_test))

# 4. Scale features using training data only
print("\n4. Scaling features...")

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

joblib.dump(scaler, OUTPUT_DIR / "lstm_scaler.pkl")

# 5. Convert transactions into sequences
print("\n5. Creating transaction sequences...")

SEQUENCE_LENGTH = 10

def create_sequences(X, y, sequence_length):

    sequences = []
    labels = []

    for i in range(sequence_length - 1, len(X)):

        sequence = X[
            i - sequence_length + 1 : i + 1
        ]

        sequences.append(sequence)

        # Predict whether the final transaction is fraud
        labels.append(y[i])

    return (
        np.array(sequences, dtype=np.float32),
        np.array(labels, dtype=np.float32)
    )


X_train_seq, y_train_seq = create_sequences(
    X_train, y_train, SEQUENCE_LENGTH
)

X_val_seq, y_val_seq = create_sequences(
    X_val, y_val, SEQUENCE_LENGTH
)

X_test_seq, y_test_seq = create_sequences(
    X_test, y_test, SEQUENCE_LENGTH
)

# 6. Save LSTM data
print("\n6. Saving sequence datasets...")

np.save(OUTPUT_DIR / "X_train_lstm.npy", X_train_seq)
np.save(OUTPUT_DIR / "y_train_lstm.npy", y_train_seq)

np.save(OUTPUT_DIR / "X_val_lstm.npy", X_val_seq)
np.save(OUTPUT_DIR / "y_val_lstm.npy", y_val_seq)

np.save(OUTPUT_DIR / "X_test_lstm.npy", X_test_seq)
np.save(OUTPUT_DIR / "y_test_lstm.npy", y_test_seq)

# 7. Display results
print("\n7. FINAL SEQUENCE SHAPES")

print("Training:", X_train_seq.shape)
print("Validation:", X_val_seq.shape)
print("Testing:", X_test_seq.shape)

print("\nTraining fraud samples:", int(y_train_seq.sum()))
print("Validation fraud samples:", int(y_val_seq.sum()))
print("Testing fraud samples:", int(y_test_seq.sum()))

print("\nSequence length:", SEQUENCE_LENGTH)
print("Features per transaction:", X_train_seq.shape[2])

print("\nLSTM data saved successfully!")

print("=" * 60)
print("       LSTM DATA PREPARATION COMPLETED")
print("=" * 60)
