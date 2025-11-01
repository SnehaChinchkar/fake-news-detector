import pandas as pd

MIN_WORDS = 5

true_df = pd.read_csv("data/True.csv")
fake_df = pd.read_csv("data/Fake.csv")

true_filtered = true_df[true_df['title'].str.split().str.len() >= MIN_WORDS]['title'].dropna()
fake_filtered = fake_df[fake_df['title'].str.split().str.len() >= MIN_WORDS]['title'].dropna()

n_samples = min(len(true_filtered), len(fake_filtered))
real_titles = true_filtered.sample(n_samples, random_state=42)
fake_titles = fake_filtered.sample(n_samples, random_state=42)

custom_df = pd.DataFrame({
    'text': pd.concat([real_titles, fake_titles], ignore_index=True),
    'label': [1]*n_samples + [0]*n_samples
})

# Save to CSV
custom_df.to_csv("data/custom_short_texts.csv", index=False)
print(f"✅ Saved data/custom_short_texts.csv with {len(custom_df)} samples (balanced, ≥{MIN_WORDS} words).")
