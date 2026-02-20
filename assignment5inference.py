import pandas as pd
import numpy as np
from tensorflow import keras
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle

# ----------------------------
# Settings
# ----------------------------
DATA_PATH = "data/tourism.csv"
MODEL_PATH = "examples/assignment.h5"
PREPROCESS_PATH = "examples/assignment_preprocess.pkl"
OUT_DIR = "data/output"
os.makedirs(OUT_DIR, exist_ok=True)

# ----------------------------
# Load model
# ----------------------------
model = keras.models.load_model(MODEL_PATH, compile=False)
print("Model loaded successfully!")
model.summary()

print(f"Using dataset for inference: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)

X = df.drop("ProdTaken", axis=1)
y = df["ProdTaken"].astype(int).values

# ----------------------------
# Load preprocess (feature columns + scaler + best_threshold)
# ----------------------------
with open(PREPROCESS_PATH, "rb") as f:
    preprocess = pickle.load(f)

expected_columns = preprocess["feature_columns"]   # ✅ FIXED KEY
scaler = preprocess["scaler"]

# Use threshold from training/VAL (recommended: no leakage)
best_t = float(preprocess.get("best_threshold", 0.5))

# ----------------------------
# One-hot encode + align columns exactly like training
# ----------------------------
categorical_columns = X.select_dtypes(include=["object"]).columns.tolist()
X_encoded = pd.get_dummies(X, columns=categorical_columns, drop_first=True)
X_encoded = X_encoded.reindex(columns=expected_columns, fill_value=0)

# Scale using training scaler
X_scaled = scaler.transform(X_encoded)

# Predict probabilities
y_pred_proba = model.predict(X_scaled, verbose=0).flatten()

# ----------------------------
# OPTION 1 (recommended): use saved threshold from training/VAL
# ----------------------------
y_pred = (y_pred_proba >= best_t).astype(int)

# ----------------------------
# Metrics (weighted, because prof cares weighted avg)
# ----------------------------
accuracy = accuracy_score(y, y_pred)
auc_score = roc_auc_score(y, y_pred_proba)

w_precision = precision_score(y, y_pred, average="weighted", zero_division=0)
w_recall = recall_score(y, y_pred, average="weighted", zero_division=0)
w_f1 = f1_score(y, y_pred, average="weighted", zero_division=0)

print("\n" + "=" * 60)
print("INFERENCE RESULTS (Tourism)")
print("=" * 60)
print(f"Threshold used (from training/VAL): {best_t:.3f}")
print(f"\nAccuracy:  {accuracy:.4f}")
print(f"AUC Score: {auc_score:.4f}")
print(f"Weighted Precision: {w_precision:.4f}")
print(f"Weighted Recall:    {w_recall:.4f}")
print(f"Weighted F1-Score:  {w_f1:.4f}")

print("\nClassification Report:")
print(classification_report(y, y_pred))

cm = confusion_matrix(y, y_pred)
print("\nConfusion Matrix:")
print(cm)

# ----------------------------
# Plots
# ----------------------------
plt.figure(figsize=(9, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar_kws={"label": "Count"})
plt.title("Confusion Matrix - Tourism Model", fontsize=14, fontweight="bold")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.tight_layout()
out_cf = os.path.join(OUT_DIR, "tourism_result_cf.jpg")
plt.savefig(out_cf, dpi=300, bbox_inches="tight")
print(f"\nConfusion matrix plot saved to: {out_cf}")

plt.figure(figsize=(9, 7))
cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_normalized, annot=True, fmt=".2%", cmap="Greens", cbar_kws={"label": "Percentage"})
plt.title("Normalized Confusion Matrix - Tourism Model", fontsize=14, fontweight="bold")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.tight_layout()
out_cfn = os.path.join(OUT_DIR, "tourism_result_cfn.jpg")
plt.savefig(out_cfn, dpi=300, bbox_inches="tight")
print(f"Normalized confusion matrix plot saved to: {out_cfn}")

# ----------------------------
# Save text report
# ----------------------------
report_path = os.path.join(OUT_DIR, "tourism_inference_report.txt")
with open(report_path, "w") as f:
    f.write("=" * 70 + "\n")
    f.write("Inference Results (Tourism)\n")
    f.write("=" * 70 + "\n")
    f.write(f"Dataset:   {DATA_PATH}\n")
    f.write(f"Model:     {MODEL_PATH}\n")
    f.write(f"Threshold: {best_t:.3f}\n\n")
    f.write(f"Accuracy:  {accuracy:.4f}\n")
    f.write(f"AUC:       {auc_score:.4f}\n")
    f.write(f"Weighted Precision: {w_precision:.4f}\n")
    f.write(f"Weighted Recall:    {w_recall:.4f}\n")
    f.write(f"Weighted F1-Score:  {w_f1:.4f}\n\n")
    f.write("Confusion Matrix:\n")
    f.write(str(cm) + "\n\n")
    f.write("Classification Report:\n")
    f.write(classification_report(y, y_pred))

print(f"Inference report saved to: {report_path}")
