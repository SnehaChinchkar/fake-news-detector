import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from src.preprocess import load_and_clean_data


def safe_cleanup_dataset_files(base_folder):
    """
    Safely remove True.csv, Fake.csv, and custom_dataset.csv recursively from base_folder.
    This ensures cleanup works whether files are in /data/ or in KaggleHub cache.
    """
    targets = {"True.csv", "Fake.csv", "custom_dataset.csv"}
    removed = set()

    for root, _, files in os.walk(base_folder):
        for f in files:
            if f in targets:
                file_path = os.path.join(root, f)
                try:
                    os.remove(file_path)
                    removed.add(f)
                    print(f"🧹 Removed: {file_path}")
                except Exception as e:
                    print(f"⚠️ Failed to remove {file_path}: {e}")

    for t in targets - removed:
        print(f"ℹ️ File not found (skip): {t}")


def train_model_with_meta(model_path="storage/fake_news_model.joblib"):
    """
    Standalone training function for local dev testing.
    Loads dataset from DATA_PATH env or default local CSV,
    trains model, and saves pipeline to storage/.
    """
    data_path = os.getenv(
        "DATA_PATH",
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "custom_short_texts.csv")
    )

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"❌ Dataset not found at {data_path}")

    print(f"[INFO] Loading and preprocessing data from: {data_path}")
    df = load_and_clean_data(os.path.dirname(data_path))

    print(f"[INFO] Loaded {len(df)} samples")

    X = df.drop(columns=["label"])
    y = df["label"]

    text_col = "content"
    numeric_cols = [
        "headline_len", "punct_count", "upper_ratio",
        "exclamation_ratio", "clickbait_score", "age_days", "source_cred"
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("text", TfidfVectorizer(max_features=8000, ngram_range=(1, 2)), text_col),
            ("num", StandardScaler(), numeric_cols),
        ],
        remainder="drop"
    )

    clf = RandomForestClassifier(
        n_estimators=250,
        max_depth=None,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("clf", clf)
    ])

    short_text_mask = X["headline_len"] < 10  # short/custom samples
    sample_weights = pd.Series(1.0, index=X.index)
    sample_weights[short_text_mask] = 0.5  # reduce weight for short samples

    X_train, X_test, y_train, y_test, sw_train, sw_test = train_test_split(
        X, y, sample_weights, test_size=0.2, random_state=42
    )

    print("[INFO] Training model with sample weights...")
    pipe.fit(X_train, y_train, clf__sample_weight=sw_train)

    acc = pipe.score(X_test, y_test)
    print(f"[OK] Model accuracy: {acc:.4f}")

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(pipe, model_path)
    print(f"[SAVED] Trained model stored at: {model_path}")


def train_and_save(data_folder, model_path):
    """
    Universal training wrapper for cloud or local use.
    Automatically finds the Kaggle dataset CSV in the folder.
    """
    print(f"[INFO] Preparing to train using data folder: {data_folder}")

    csv_candidates = [f for f in os.listdir(data_folder) if f.endswith(".csv")]
    if not csv_candidates:
        raise FileNotFoundError(f"No CSV files found in {data_folder}")
    data_path = os.path.join(data_folder, csv_candidates[0])
    print(f"[INFO] Found dataset: {data_path}")

    df = load_and_clean_data(os.path.dirname(data_path))
    print(f"[INFO] Loaded {len(df)} rows from Kaggle dataset")

    X = df.drop(columns=["label"])
    y = df["label"]

    text_col = "content"
    numeric_cols = [
        "headline_len", "punct_count", "upper_ratio",
        "exclamation_ratio", "clickbait_score", "age_days", "source_cred"
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("text", TfidfVectorizer(max_features=8000, ngram_range=(1, 2)), text_col),
            ("num", StandardScaler(), numeric_cols),
        ],
        remainder="drop"
    )

    clf = RandomForestClassifier(
        n_estimators=250,
        max_depth=None,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("clf", clf)
    ])

    short_text_mask = X["headline_len"] < 10
    sample_weights = pd.Series(1.0, index=X.index)
    sample_weights[short_text_mask] = 0.5  # half weight for short/custom samples

    print("[INFO] Training RandomForestClassifier pipeline with sample weights...")
    pipe.fit(X, y, clf__sample_weight=sample_weights)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(pipe, model_path)
    print(f"[SAVED] Model successfully stored at: {model_path}")

    # ✅ Cleanup (now works for KaggleHub)
    try:
        safe_cleanup_dataset_files(data_folder)
        print("✅ Cleanup complete.")
    except Exception as e:
        print(f"⚠️ Cleanup failed: {e}")


if __name__ == "__main__":
    print("[INFO] Running standalone training (local test)...")
    train_model_with_meta()
