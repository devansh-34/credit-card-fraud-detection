"""
Baseline comparison: Logistic Regression and Random Forest.
Run from project root: python baseline_comparison.py
Requires processed/X_train.npy, y_train.npy, X_val.npy, y_val.npy.
Saves CSV, JSON, and comparison plot in outputs/.
"""
from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix)
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "processed"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
SEED = 42

def load(name):
    p = DATA / f"{name}.npy"
    if not p.exists():
        raise FileNotFoundError(f"Missing file: {p}")
    return np.load(p)

def best_threshold(y, prob):
    thresholds = np.linspace(.05, .95, 181)
    scores = [f1_score(y, (prob >= t).astype(int), zero_division=0) for t in thresholds]
    i = int(np.argmax(scores))
    return float(thresholds[i]), float(scores[i])

def evaluate(name, model, Xtr, ytr, Xv, yv):
    print(f"\nTraining {name}...")
    model.fit(Xtr, ytr)
    prob = model.predict_proba(Xv)[:, 1]
    pred = (prob >= .5).astype(int)
    threshold, tuned_f1 = best_threshold(yv, prob)
    tuned_pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(yv, pred, labels=[0, 1]).ravel()
    row = {
        "Model": name,
        "Accuracy_0.5": float(accuracy_score(yv, pred)),
        "Precision_0.5": float(precision_score(yv, pred, zero_division=0)),
        "Recall_0.5": float(recall_score(yv, pred, zero_division=0)),
        "F1_0.5": float(f1_score(yv, pred, zero_division=0)),
        "ROC_AUC": float(roc_auc_score(yv, prob)),
        "PR_AUC": float(average_precision_score(yv, prob)),
        "Best_F1_Threshold_Validation": threshold,
        "Best_F1_Validation": tuned_f1,
        "Precision_BestThreshold": float(precision_score(yv, tuned_pred, zero_division=0)),
        "Recall_BestThreshold": float(recall_score(yv, tuned_pred, zero_division=0)),
        "TN_0.5": int(tn), "FP_0.5": int(fp), "FN_0.5": int(fn), "TP_0.5": int(tp),
    }
    print(f"ROC-AUC={row['ROC_AUC']:.4f} | PR-AUC={row['PR_AUC']:.4f} | F1@0.5={row['F1_0.5']:.4f}")
    return row

def main():
    X_train, y_train = load("X_train"), load("y_train").reshape(-1)
    X_val, y_val = load("X_val"), load("y_val").reshape(-1)
    print("Train:", X_train.shape, "Validation:", X_val.shape)
    print("Train fraud:", int((y_train == 1).sum()), "Validation fraud:", int((y_val == 1).sum()))
    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(class_weight="balanced", max_iter=2000, random_state=SEED))
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2, class_weight="balanced_subsample",
            n_jobs=-1, random_state=SEED
        ),
    }
    rows = [evaluate(n, m, X_train, y_train, X_val, y_val) for n, m in models.items()]
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "baseline_comparison.csv", index=False)
    with open(OUT / "baseline_comparison.json", "w") as f:
        json.dump({"evaluation_split": "validation", "note": "Threshold optimization is on validation data; these are not final test results.", "results": rows}, f, indent=2)
    cols = ["ROC_AUC", "PR_AUC", "Precision_0.5", "Recall_0.5", "F1_0.5"]
    ax = df.set_index("Model")[cols].plot(kind="bar", figsize=(12, 7), ylim=(0, 1))
    ax.set_title("Baseline Model Comparison (Validation)")
    ax.set_ylabel("Score"); ax.set_xlabel("Model"); ax.grid(axis="y", alpha=.25)
    plt.xticks(rotation=0); plt.tight_layout()
    plt.savefig(OUT / "baseline_comparison.png", dpi=300); plt.close()
    print("\nSaved outputs/baseline_comparison.csv, .json, and .png")
    print("These are validation results, not test results.")

if __name__ == "__main__":
    main()
