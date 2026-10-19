"""Train FraudShield model from the included CSV.
Run from the project root: python train_model.py
"""
from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report, confusion_matrix, average_precision_score,
    roc_auc_score, precision_score, recall_score, f1_score
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from backend.config import FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET, DATA_PATH, MODEL_PATH

ROOT = Path(__file__).resolve().parent
data_path = ROOT / DATA_PATH
model_path = ROOT / MODEL_PATH
model_path.parent.mkdir(parents=True, exist_ok=True)

print(f"Loading dataset: {data_path}")
df = pd.read_csv(data_path)
required = FEATURES + [TARGET]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Dataset is missing columns: {missing}")

# Exclude rows without a target. The target is never used as an input.
df = df[required].dropna(subset=[TARGET]).copy()
X = df[FEATURES]
y = df[TARGET].astype(int)
print(f"Rows: {len(df):,}; features: {len(FEATURES)}")
print("Target distribution:")
print(y.value_counts(normalize=True).rename("fraction").to_string())

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])
preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, NUMERIC_FEATURES),
    ("categorical", categorical_pipeline, CATEGORICAL_FEATURES)
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=250,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1
    ))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print("Training model...")
model.fit(X_train, y_train)
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

print("\\n=== Evaluation on held-out test set ===")
print(classification_report(y_test, predictions, digits=4, zero_division=0))
print("Confusion matrix [ [TN, FP], [FN, TP] ]:")
print(confusion_matrix(y_test, predictions))
print(f"ROC-AUC: {roc_auc_score(y_test, probabilities):.4f}")
print(f"PR-AUC (average precision): {average_precision_score(y_test, probabilities):.4f}")
print(f"Precision: {precision_score(y_test, predictions, zero_division=0):.4f}")
print(f"Recall:    {recall_score(y_test, predictions, zero_division=0):.4f}")
print(f"F1-score:  {f1_score(y_test, predictions, zero_division=0):.4f}")

joblib.dump({
    "pipeline": model,
    "features": FEATURES,
    "threshold": 0.50,
    "model_name": "Random Forest"
}, model_path)
print(f"\\nSaved trained pipeline to: {model_path}")
print("The saved pipeline includes preprocessing and the classifier.")
