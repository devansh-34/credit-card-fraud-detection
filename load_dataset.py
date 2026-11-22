
from pathlib import Path
import pandas as pd

# Find the main project folder
PROJECT_DIR = Path(__file__).resolve().parent
DATASET_PATH = PROJECT_DIR / "dataset" / "creditcard.csv"

print("=" * 55)
print("       CREDIT CARD FRAUD DETECTION")
print("             DATASET LOADER")
print("=" * 55)

# Check whether the CSV file exists
if not DATASET_PATH.exists():
    print("\nERROR: creditcard.csv was not found!")
    print("Expected location:", DATASET_PATH)
    raise FileNotFoundError("Please put creditcard.csv inside the dataset folder.")

# Load dataset
print("\nLoading dataset...")
df = pd.read_csv(DATASET_PATH)

# Basic information
print("\n1. DATASET SHAPE")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

print("\n2. FIRST FIVE TRANSACTIONS")
print(df.head())

print("\n3. COLUMN NAMES")
print(df.columns.tolist())

print("\n4. DATA TYPES")
print(df.dtypes)

print("\n5. MISSING VALUES")
print(df.isnull().sum().sum())

print("\n6. DUPLICATE ROWS")
print(df.duplicated().sum())

# Fraud and normal transaction counts
print("\n7. TRANSACTION CLASS DISTRIBUTION")
class_counts = df["Class"].value_counts().sort_index()
print(class_counts.rename(index={0: "Normal", 1: "Fraud"}))

print("\n8. CLASS PERCENTAGES")
print((df["Class"].value_counts(normalize=True) * 100).sort_index())

print("\n9. FRAUD SUMMARY")
print("Normal transactions:", (df["Class"] == 0).sum())
print("Fraudulent transactions:", (df["Class"] == 1).sum())

fraud_percentage = (df["Class"] == 1).mean() * 100
print(f"Fraud percentage: {fraud_percentage:.3f}%")

print("\n10. TRANSACTION AMOUNT SUMMARY")
print(df["Amount"].describe())

print("\n" + "=" * 55)
print("       DATASET LOADED SUCCESSFULLY!")
print("=" * 55)
