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

def train_model_with_meta(model_path="model/pipeline_with_meta.pkl"):
    # -----------------------------
    # Load and preprocess data
    # -----------------------------
    data_path = os.getenv("DATA_PATH", "/var/data/custom_short_texts.csv")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"❌ Dataset not found at {data_path}")

    print(f"🔹 Loading and preprocessing data from: {data_path}")
    df = load_and_clean_data(data_path)
    print(f"✅ Loaded {len(df)} samples from {data_path}")

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
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(pipe, model_path)
    print(f"💾 Saved trained model to {model_path}")


if __name__ == "__main__":
    train_model_with_meta()
