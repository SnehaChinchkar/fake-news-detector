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


def safe_remove(file_path: str):
    """Remove a file safely if it exists."""
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
            print(f"🧹 Removed {file_path}")
        except Exception as e:
            print(f"⚠️ Failed to remove {file_path}: {e}")
    else:
        print(f"ℹ️ File not found (skip): {file_path}")


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

    short_text_mask = X["headline_len"] < 10
    sample_weights = pd.Series(1.0, index=X.index)
    sample_weights[short_text_mask] = 0.5

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

    # 🧹 Cleanup custom short dataset (since we only use it in meta training)
    safe_remove(data_path)


def train_and_save(data_folder, model_path):
    """
    Universal training wrapper for cloud or local use.
    Automatically finds the Kaggle dataset CSVs in the folder,
    loads, trains, and cleans up files progressively.
    """
    print(f"[INFO] Preparing to train using data folder: {data_folder}")

    # Locate datasets
    true_path = os.path.join(data_folder, "True.csv")
    fake_path = os.path.join(data_folder, "Fake.csv")
    custom_path = os.path.join(data_folder, "custom_short_texts.csv")

    if not (os.path.exists(true_path) and os.path.exists(fake_path)):
        raise FileNotFoundError(f"❌ Missing one or both CSV files in {data_folder}")

    print(f"[INFO] Found True/Fake datasets at: {data_folder}")

    # 🧠 Load and preprocess
    df = load_and_clean_data(data_folder)
    print(f"[INFO] Loaded {len(df)} rows from Kaggle + custom datasets")

    # 🧹 Cleanup Kaggle originals (after loading into memory)
    safe_remove(true_path)
    safe_remove(fake_path)

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
    sample_weights[short_text_mask] = 0.5

    print("[INFO] Training RandomForestClassifier pipeline with sample weights...")
    pipe.fit(X, y, clf__sample_weight=sample_weights)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(pipe, model_path)
    print(f"[SAVED] Model successfully stored at: {model_path}")

    safe_remove(custom_path)


if __name__ == "__main__":
    print("[INFO] Running standalone training (local test)...")
    train_model_with_meta()
