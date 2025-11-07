import os
import pandas as pd
import re

# === CONFIG ===
DATA_DIR = os.getenv("DATA_PATH", os.path.join(os.path.dirname(__file__), "data"))
TRUE_PATH = os.path.join(DATA_DIR, "True.csv")
FAKE_PATH = os.path.join(DATA_DIR, "Fake.csv")
OUTPUT_PATH = os.path.join(DATA_DIR, "custom_short_texts.csv")

MIN_WORDS = 5

# === UTILITIES ===
def clean_text(text):
    """Basic cleaning for news text."""
    text = re.sub(r"http\S+", "", str(text))
    text = re.sub(r"[^A-Za-z0-9\s\.,!?'-]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()

# === LOAD DATA ===
if not (os.path.exists(TRUE_PATH) and os.path.exists(FAKE_PATH)):
    raise FileNotFoundError(f"❌ Missing one or both CSV files: {TRUE_PATH}, {FAKE_PATH}")

print(f"✅ Loading datasets from: {DATA_DIR}")

true_df = pd.read_csv(TRUE_PATH)
fake_df = pd.read_csv(FAKE_PATH)

# === FILTER SHORT TITLES ===
true_df["title"] = true_df["title"].fillna("").apply(clean_text)
fake_df["title"] = fake_df["title"].fillna("").apply(clean_text)

true_filtered = true_df[true_df["title"].str.split().str.len() >= MIN_WORDS]["title"]
fake_filtered = fake_df[fake_df["title"].str.split().str.len() >= MIN_WORDS]["title"]

print(f"[INFO] True titles kept: {len(true_filtered)}, Fake titles kept: {len(fake_filtered)} (≥{MIN_WORDS} words)")

# === BALANCE DATA ===
n_samples = min(len(true_filtered), len(fake_filtered))
real_titles = true_filtered.sample(n_samples, random_state=42)
fake_titles = fake_filtered.sample(n_samples, random_state=42)

# === BUILD CUSTOM DATAFRAME ===
custom_df = pd.DataFrame({
    "text": pd.concat([real_titles, fake_titles], ignore_index=True),
    "label": [1] * n_samples + [0] * n_samples
})

# === SAVE TO CSV ===
os.makedirs(DATA_DIR, exist_ok=True)
custom_df.to_csv(OUTPUT_PATH, index=False)

print(f"✅ Saved {OUTPUT_PATH} with {len(custom_df)} samples (balanced, ≥{MIN_WORDS} words).")
