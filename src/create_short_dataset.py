import pandas as pd

# Load your current datasets
true_df = pd.read_csv("data/True.csv")
fake_df = pd.read_csv("data/Fake.csv")

# Pick only 'title' column for short text
real_titles = true_df['title'].dropna().sample(200, random_state=42)
fake_titles = fake_df['title'].dropna().sample(200, random_state=42)

# Combine into one DataFrame
custom_df = pd.DataFrame({
    'text': pd.concat([real_titles, fake_titles], ignore_index=True),
    'label': [1]*len(real_titles) + [0]*len(fake_titles)
})

# Save to CSV
custom_df.to_csv("data/custom_short_texts.csv", index=False)
print("✅ Saved data/custom_short_texts.csv with", len(custom_df), "samples.")
