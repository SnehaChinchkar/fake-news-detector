# import pandas as pd

# # Load your current datasets
# true_df = pd.read_csv("data/True.csv")
# fake_df = pd.read_csv("data/Fake.csv")

# # Pick only 'title' column for short text
# real_titles = true_df['title'].dropna().sample(400, random_state=42)
# fake_titles = fake_df['title'].dropna().sample(400, random_state=42)

# # Combine into one DataFrame
# custom_df = pd.DataFrame({
#     'text': pd.concat([real_titles, fake_titles], ignore_index=True),
#     'label': [1]*len(real_titles) + [0]*len(fake_titles)
# })

# # Save to CSV
# custom_df.to_csv("data/custom_short_texts.csv", index=False)
# print("✅ Saved data/custom_short_texts.csv with", len(custom_df), "samples.")
import pandas as pd

MIN_WORDS = 5

# Load datasets
true_df = pd.read_csv("data/True.csv")
fake_df = pd.read_csv("data/Fake.csv")

# Filter titles with at least MIN_WORDS words
true_filtered = true_df[true_df['title'].str.split().str.len() >= MIN_WORDS]['title'].dropna()
fake_filtered = fake_df[fake_df['title'].str.split().str.len() >= MIN_WORDS]['title'].dropna()

# Sample as many as possible while keeping a balanced dataset
n_samples = min(len(true_filtered), len(fake_filtered))
real_titles = true_filtered.sample(n_samples, random_state=42)
fake_titles = fake_filtered.sample(n_samples, random_state=42)

# Combine into one DataFrame
custom_df = pd.DataFrame({
    'text': pd.concat([real_titles, fake_titles], ignore_index=True),
    'label': [1]*n_samples + [0]*n_samples
})

# Save to CSV
custom_df.to_csv("data/custom_short_texts.csv", index=False)
print(f"✅ Saved data/custom_short_texts.csv with {len(custom_df)} samples (balanced, ≥{MIN_WORDS} words).")
