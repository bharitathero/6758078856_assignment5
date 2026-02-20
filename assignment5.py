import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import callbacks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score

# ----------------------------
# Config
# ----------------------------
DATA_PATH = "data/tourism.csv"          # you renamed cleaned file to tourism.csv already
MODEL_PATH = "examples/assignment.h5"
PREPROCESS_PATH = "examples/assignment_preprocess.pkl"
REPORT_PATH = "data/output/tourism_training_report.txt"

os.makedirs("data/output", exist_ok=True)
os.makedirs("examples", exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.2            # from train split
EPOCHS = 300
BATCH_SIZE = 32
LEARNING_RATE = 1e-3      # slightly higher can converge better; ReduceLROnPlateau will tame it

# ----------------------------
# Utility: find best threshold for WEIGHTED F1
# ----------------------------
def best_threshold_for_weighted_f1(y_true, y_prob, thresholds=None):
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 991)  # step ~0.005
    best_t, best_f1 = 0.5, -1.0
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        f1w = f1_score(y_true, y_pred, average="weighted")
        if f1w > best_f1:
            best_f1 = f1w
            best_t = float(t)
    return best_t, float(best_f1)

# ----------------------------
# Callback: compute val weighted F1 + choose best threshold
# ----------------------------
class WeightedF1Callback(callbacks.Callback):
    def __init__(self, X_val, y_val):
        super().__init__()
        self.X_val = X_val
        self.y_val = y_val
        self.best_val_f1w = -1.0
        self.best_threshold = 0.5

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        y_prob = self.model.predict(self.X_val, verbose=0).reshape(-1)
        t, f1w = best_threshold_for_weighted_f1(self.y_val, y_prob)

        logs["val_weighted_f1"] = f1w  # so you can see it in logs if needed

        # Track best
        if f1w > self.best_val_f1w:
            self.best_val_f1w = f1w
            self.best_threshold = t

        print(f" — val_weighted_f1: {f1w:.4f} (best_t={t:.3f})")

# ----------------------------
# Load data
# ----------------------------
print("Loading data...")
print(f"Using dataset: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)

print(f"Dataset shape: {df.shape}")
print(f"\nColumn names: {df.columns.tolist()}")
print(f"\nTarget distribution:\n{df['ProdTaken'].value_counts()}")

X = df.drop("ProdTaken", axis=1)
y = df["ProdTaken"].astype(int).values

# One-hot encode categoricals
categorical_columns = X.select_dtypes(include=["object"]).columns.tolist()
print(f"\nCategorical columns: {categorical_columns}")

X_encoded = pd.get_dummies(X, columns=categorical_columns, drop_first=True)
feature_columns = X_encoded.columns.tolist()

print(f"Features after encoding: {X_encoded.shape[1]}")
print(X_encoded.head())

# ----------------------------
# Split: train/test (stratified)
# ----------------------------
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X_encoded, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

# Split train into train/val (stratified)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=VAL_SIZE, random_state=RANDOM_STATE, stratify=y_train_full
)

# Standardize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print(f"\nTraining set size: {X_train_scaled.shape[0]}")
print(f"Validation set size: {X_val_scaled.shape[0]}")
print(f"Testing set size: {X_test_scaled.shape[0]}")

# ----------------------------
# Model (slightly less dropout -> better overall fit)
# ----------------------------
model = keras.Sequential([
    keras.layers.Input(shape=(X_train_scaled.shape[1],)),

    keras.layers.Dense(512, activation="relu"),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.12),

    keras.layers.Dense(256, activation="relu"),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.12),

    keras.layers.Dense(128, activation="relu"),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.12),

    keras.layers.Dense(64, activation="relu"),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.12),

    keras.layers.Dense(32, activation="relu"),
    keras.layers.BatchNormalization(),

    keras.layers.Dense(16, activation="relu"),
    keras.layers.BatchNormalization(),

    keras.layers.Dense(1, activation="sigmoid"),
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss="binary_crossentropy",
    metrics=[
        keras.metrics.AUC(name="auc"),
        keras.metrics.AUC(curve="PR", name="pr_auc"),  # more sensitive than ROC-AUC in imbalance
        "accuracy",
    ],
)

model.summary()

# ----------------------------
# Callbacks
# ----------------------------
f1_cb = WeightedF1Callback(X_val_scaled, y_val)

callbacks_list = [
    callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True),
    callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=6, min_lr=1e-5, verbose=1),
    f1_cb,
]

# ----------------------------
# Train
# ----------------------------
print("\nTraining the model...")
history = model.fit(
    X_train_scaled, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_val_scaled, y_val),
    verbose=1,
    callbacks=callbacks_list,
)

# ----------------------------
# Evaluate on TEST using best threshold from VAL (NO leakage)
# ----------------------------
print("\nEvaluating the model on TEST set...")

y_test_prob = model.predict(X_test_scaled, verbose=0).reshape(-1)

best_t = f1_cb.best_threshold
y_test_pred = (y_test_prob >= best_t).astype(int)

test_f1w = f1_score(y_test, y_test_pred, average="weighted")

print(f"\nBest threshold by WEIGHTED F1 on VAL: {best_t:.3f}")
print(f"Weighted F1 (TEST): {test_f1w:.4f}")

print("\nClassification Report (TEST):")
print(classification_report(y_test, y_test_pred))

print("\nConfusion Matrix (TEST):")
print(confusion_matrix(y_test, y_test_pred))

# ----------------------------
# Save model + preprocess bundle
# ----------------------------
model.save(MODEL_PATH)
print(f"\nModel saved to: {MODEL_PATH}")

preprocess = {
    "scaler": scaler,
    "feature_columns": feature_columns,
    "best_threshold": best_t,
}

with open(PREPROCESS_PATH, "wb") as f:
    pickle.dump(preprocess, f)

print(f"Preprocess saved to: {PREPROCESS_PATH}")

# ----------------------------
# Save training report (your prof-style text output)
# ----------------------------
with open(REPORT_PATH, "w", encoding="utf-8") as f:
    f.write("======================================================================\n")
    f.write("Model Evaluation Results (Tourism)\n")
    f.write("======================================================================\n\n")
    f.write(f"Model: {MODEL_PATH}\n")
    f.write(f"Dataset: {DATA_PATH}\n")
    f.write(f"Samples: {len(df)}\n\n")

    f.write("Best threshold chosen on validation (weighted F1): "
            f"{best_t:.3f}\n")
    f.write(f"Weighted F1 on TEST: {test_f1w:.4f}\n\n")

    f.write("======================================================================\n")
    f.write("Classification Report (TEST)\n")
    f.write("======================================================================\n")
    f.write(classification_report(y_test, y_test_pred))
    f.write("\n\n")

    f.write("======================================================================\n")
    f.write("Confusion Matrix (TEST)\n")
    f.write("======================================================================\n")
    f.write(str(confusion_matrix(y_test, y_test_pred)))
    f.write("\n")

print(f"\nTraining report saved to: {REPORT_PATH}")
