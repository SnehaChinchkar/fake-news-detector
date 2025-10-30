import os
import pandas as pd
import re
from datetime import datetime
from urllib.parse import urlparse

# -----------------------------
# ✅ Step 1 — Environment-aware data path
# -----------------------------
DATA_DIR = os.getenv("DATA_PATH", "data")

# -----------------------------
# Domain + credibility helpers
# -----------------------------
def extract_domain(url_or_source):
    """Extract domain from URL or text source."""
    if pd.isna(url_or_source):
        return ""
    try:
        parsed = urlparse(str(url_or_source))
        domain = parsed.netloc if parsed.netloc else parsed.path
        domain = domain.replace("www.", "")
        return domain.lower()
    except:
        return str(url_or_source).lower()


# Strengthened credibility lookup table
CRED_LOOKUP = {
    # Real / high-credibility outlets
    "reuters.com": 1.5,
    "bbc.co.uk": 1.5,
    "nytimes.com": 1.5,
    "apnews.com": 1.5,
    "npr.org": 1.5,
    "theguardian.com": 1.5,
    "washingtonpost.com": 1.5,
    "bloomberg.com": 1.2,
    "cnn.com": 1.2,
    "politico.com": 1.0,
    "nasa.gov": 1.5,
    # Moderate / regional credible
    "cnbc.com": 1.0,
    "usatoday.com": 1.0,
    "abcnews.go.com": 1.0,
    "dw.com": 1.0,
    "indiatimes.com": 0.8,
    "thehindu.com": 1.0,
    "hindustantimes.com": 0.8,
    "ndtv.com": 1.0,
    # Low-credibility or fake sources
    "thepoliticalinsider.com": -1.2,
    "politicususa.com": -1.0,
    "dailybuzzlive.com": -1.5,
    "libertywritersnews.com": -1.5,
    "worldnewsdailyreport.com": -1.5,
    "news4ktla.com": -1.5,
    "empirenews.net": -1.5,
    "beforeitsnews.com": -1.5,
    "theonion.com": -1.5,  # satire
    "buzzfeed.com": -1.0,
    "dailymail.co.uk": -1.0,
    "abcnews.com.co": -1.5,
}

def source_cred_score(domain, default=-0.2):
    """Return credibility score for a domain, or penalty for unknown domains."""
    if not domain:
        return default
    for k, v in CRED_LOOKUP.items():
        if k in domain:
            return v
    return default


# -----------------------------
# Text + metadata preprocessing
# -----------------------------
def clean_text(text):
    """Basic cleaning for article text."""
    text = re.sub(r"http\S+", "", str(text))
    text = re.sub(r"[^A-Za-z0-9\s\.,!?'-]", "", text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()


# Clickbait-related keywords
CLICKBAIT_WORDS = [
    "shocking", "breaking", "amazing", "you won't believe", "exclusive",
    "unbelievable", "surprising", "secret", "revealed", "crazy", "must see"
]

def clickbait_score(text):
    """Simple clickbait score based on keyword count."""
    txt = text.lower()
    return sum(1 for word in CLICKBAIT_WORDS if word in txt)


def make_metadata(df):
    """Compute derived metadata features for each article."""
    if 'source' in df.columns:
        df['source_domain'] = df['source'].apply(extract_domain)
    elif 'link' in df.columns:
        df['source_domain'] = df['link'].apply(extract_domain)
    else:
        df['source_domain'] = ""

    df['headline_len'] = df['content'].fillna('').apply(lambda s: len(str(s).split()))
    df['punct_count'] = df['content'].fillna('').apply(lambda s: sum(1 for c in str(s) if c in '!?.,'))
    df['upper_ratio'] = df['content'].fillna('').apply(
        lambda s: sum(1 for c in str(s) if c.isupper()) / max(1, len(str(s)))
    )
    df['exclamation_ratio'] = df['content'].fillna('').apply(
        lambda s: s.count('!') / max(1, len(s))
    )
    df['clickbait_score'] = df['content'].fillna('').apply(clickbait_score)

    if 'date' in df.columns:
        df['publish_date'] = pd.to_datetime(df['date'], errors='coerce')
        now = pd.Timestamp.now()
        df['age_days'] = (now - df['publish_date']).dt.days.fillna(99999).astype(int)
    else:
        df['age_days'] = 99999

    return df


# -----------------------------
# Load + prepare dataset
# -----------------------------
def load_and_clean_data(include_custom=True):
    """Load, clean, and combine True/Fake news + optional short samples."""
    # ✅ Step 2 — Use environment-aware paths
    true_df = pd.read_csv(os.path.join(DATA_DIR, "True.csv"))
    fake_df = pd.read_csv(os.path.join(DATA_DIR, "Fake.csv"))

    true_df['label'] = 1
    fake_df['label'] = 0

    min_len = min(len(true_df), len(fake_df))
    true_df = true_df.sample(min_len, random_state=42)
    fake_df = fake_df.sample(min_len, random_state=42)

    df = pd.concat([true_df, fake_df], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)

    df['content'] = (df['title'].fillna('') + " " + df['text'].fillna('')).apply(clean_text)

    df = make_metadata(df)
    df['source_cred'] = df['source_domain'].apply(source_cred_score)

    keep_cols = [
        'content', 'source_domain', 'headline_len', 'punct_count',
        'upper_ratio', 'exclamation_ratio', 'clickbait_score',
        'age_days', 'source_cred', 'label'
    ]
    df = df[keep_cols]

    # ✅ Step 2 (continued) — Custom short examples from environment path
    if include_custom:
        try:
            custom = pd.read_csv(os.path.join(DATA_DIR, "custom_short_texts.csv"))
            custom['label'] = custom['label'].astype(int)
            custom['content'] = custom['text'].apply(clean_text)
            custom['source_domain'] = ""
            custom['headline_len'] = custom['content'].apply(lambda s: len(str(s).split()))
            custom['punct_count'] = custom['content'].apply(lambda s: sum(1 for c in str(s) if c in '!?.,'))
            custom['upper_ratio'] = custom['content'].apply(lambda s: sum(1 for c in str(s) if c.isupper()) / max(1, len(s)))
            custom['exclamation_ratio'] = custom['content'].apply(lambda s: s.count('!') / max(1, len(s)))
            custom['clickbait_score'] = custom['content'].apply(clickbait_score)
            custom['age_days'] = 99999
            custom['source_cred'] = 0.0

            df = pd.concat([df, custom[keep_cols]], ignore_index=True)
            print(f"✅ Added {len(custom)} custom short examples from {os.path.join(DATA_DIR, 'custom_short_texts.csv')}")
        except FileNotFoundError:
            print("⚠️ No custom_short_texts.csv found — skipping short examples.")

    return df
