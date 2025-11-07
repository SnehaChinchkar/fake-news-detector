import os
import pandas as pd
import re

MIN_WORDS = 5

def clean_text(text: str) -> str:
    """Basic cleaning for news text."""
    text = re.sub(r"http\S+", "", str(text))
    text = re.sub(r"[^A-Za-z0-9\s\.,!?'-]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()

def create_custom_dataset(dataset_folder: str = None) -> str:
    """
    Create a balanced short-text dataset (custom_short_texts.csv)
    using titles from True.csv and Fake.csv in the given dataset folder.

    Args:
        dataset_folder (str): Path to folder containing True.csv and Fake.csv.
                              If None, defaults to ./data

    Returns:
        str: Path to the saved custom_short_texts.csv file
    """

    data_dir = dataset_folder or os.getenv("DATA_PATH", os.path.join(os.path.dirname(__file__), "data"))
    true_path = os.path.join(data_dir, "True.csv")
    fake_path = os.path.join(data_dir, "Fake.csv")
    output_path = os.path.join(data_dir, "custom_short_texts.csv")

    if not (os.path.exists(true_path) and os.path.exists(fake_path)):
        raise FileNotFoundError(f"❌ Missing one or both CSV files: {true_path}, {fake_path}")

    print(f"✅ Loading datasets from: {data_dir}")

    true_df = pd.read_csv(true_path)
    fake_df = pd.read_csv(fake_path)

    true_df["title"] = true_df["title"].fillna("").apply(clean_text)
    fake_df["title"] = fake_df["title"].fillna("").apply(clean_text)

    true_filtered = true_df[true_df["title"].str.split().str.len() >= MIN_WORDS]["title"]
    fake_filtered = fake_df[fake_df["title"].str.split().str.len() >= MIN_WORDS]["title"]

    print(f"[INFO] True titles kept: {len(true_filtered)}, Fake titles kept: {len(fake_filtered)} (≥{MIN_WORDS} words)")

    n_samples = min(len(true_filtered), len(fake_filtered))
    real_titles = true_filtered.sample(n_samples, random_state=42)
    fake_titles = fake_filtered.sample(n_samples, random_state=42)

    custom_df = pd.DataFrame({
        "text": pd.concat([real_titles, fake_titles], ignore_index=True),
        "label": [1] * n_samples + [0] * n_samples
    })

    # === SAVE TO CSV ===
    os.makedirs(data_dir, exist_ok=True)
    custom_df.to_csv(output_path, index=False)

    print(f"✅ Saved {output_path} with {len(custom_df)} samples (balanced, ≥{MIN_WORDS} words).")
    return output_path

if __name__ == "__main__":
    try:
        create_custom_dataset()
    except FileNotFoundError as e:
        print(str(e))
