# train.py
import os
import sys
import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report, accuracy_score

DATA_PATH = "data/samples.csv"
MODEL_PATH = "models/sign_model.joblib"
COUNTS_PATH = "data/class_counts.csv"

# ---------- Settings ----------
MIN_SAMPLES_PER_CLASS = 5   # <-- change to 2 or 3 if your dataset is small

# ---------- Load ----------
if not os.path.exists(DATA_PATH):
    sys.exit("❌ data/samples.csv not found. Run prepare_dataset.py first.")

df = pd.read_csv(DATA_PATH)
if "label" not in df.columns:
    sys.exit("❌ 'label' column not found in data/samples.csv")

X = df.drop(columns=["label"]).values
y = df["label"].astype(str).values

# ---------- Inspect & filter tiny classes ----------
counts = pd.Series(y).value_counts().sort_values(ascending=True)
os.makedirs("data", exist_ok=True)
counts.to_csv(COUNTS_PATH, header=["count"])
print("📊 Class counts (saved to data/class_counts.csv):")
print(counts)

small_classes = counts[counts < MIN_SAMPLES_PER_CLASS].index.tolist()
if small_classes:
    print(f"\n⚠️ Removing classes with < {MIN_SAMPLES_PER_CLASS} samples:")
    for c in small_classes:
        print(f" - {c} ({counts[c]} samples)")
    mask = ~pd.Series(y).isin(small_classes).values
    X, y = X[mask], y[mask]

# Need at least 2 classes after filtering
unique_classes = np.unique(y)
if len(unique_classes) < 2:
    sys.exit("❌ Not enough classes with sufficient samples after filtering. "
             f"Have {len(unique_classes)} class(es). "
             "Collect more data or lower MIN_SAMPLES_PER_CLASS.")

# ---------- Stratified split ----------
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(sss.split(X, y))
X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

# ---------- Model (Scaler + KNN) ----------
model = Pipeline(steps=[
    ("scaler", StandardScaler(with_mean=False)),  # with_mean=False for sparse-like numeric arrays
    ("knn", KNeighborsClassifier(n_neighbors=5, weights="distance"))
])

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"\n✅ Accuracy: {acc:.3f}")
print("\nClassification report:")
print(classification_report(y_test, y_pred, zero_division=0))

os.makedirs("models", exist_ok=True)
joblib.dump(model, MODEL_PATH)
print(f"\n💾 Saved model to: {MODEL_PATH}")

print("\n📝 Notes:")
print(f"- Classes used for training: {sorted(np.unique(y))}")
print(f"- Filter threshold MIN_SAMPLES_PER_CLASS = {MIN_SAMPLES_PER_CLASS} "
      "(raise it for cleaner training; lower it if your dataset is small)")

