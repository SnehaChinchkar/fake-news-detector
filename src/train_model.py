import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from preprocess import load_and_clean_data

# -----------------------------
# Load and preprocess data
# -----------------------------
print("🔹 Loading and preprocessing data...")
df = load_and_clean_data()
X = df.drop(columns=["label"])
y = df["label"]

print(f"Dataset shape: {X.shape}, Labels: {y.value_counts().to_dict()}")

# -----------------------------
# Define columns
# -----------------------------
text_col = "content"
numeric_cols = [
    "headline_len", "punct_count", "upper_ratio",
    "exclamation_ratio", "clickbait_score", "age_days", "source_cred"
]

# -----------------------------
# Preprocessor
# -----------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("text", TfidfVectorizer(max_features=8000, ngram_range=(1, 2)), text_col),
        ("num", StandardScaler(), numeric_cols),
    ],
    remainder="drop"
)

# -----------------------------
# Build model pipeline
# -----------------------------
pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("clf", RandomForestClassifier(
        n_estimators=250,
        max_depth=None,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ))
])

# -----------------------------
# Train/test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("🔹 Training model...")
pipe.fit(X_train, y_train)

acc = pipe.score(X_test, y_test)
print(f"✅ Model accuracy: {acc:.4f}")

# -----------------------------
# Save trained model
# -----------------------------
os.makedirs("model", exist_ok=True)
joblib.dump(pipe, "model/pipeline_with_meta.pkl")
print("💾 Saved trained model to model/pipeline_with_meta.pkl")
