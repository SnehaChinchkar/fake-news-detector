import joblib
import pandas as pd
from datetime import datetime
from src.preprocess import clean_text, extract_domain, source_cred_score, clickbait_score

# Load trained pipeline
pipeline = joblib.load("model/pipeline_with_meta.pkl")

def make_input_row(text, source=None, publish_date=None):
    now = pd.Timestamp.now()
    content = clean_text(text)
    domain = extract_domain(source) if source else "unknown.com"

    # Date → age
    try:
        pd_date = pd.to_datetime(publish_date) if publish_date else pd.NaT
    except Exception:
        pd_date = pd.NaT
    age_days = (now - pd_date).days if not pd.isna(pd_date) else 99999

    # Metadata features
    headline_len = len(str(content).split())
    punct_count = sum(1 for c in str(content) if c in "!?.,")
    upper_ratio = sum(1 for c in str(content) if c.isupper()) / max(1, len(content))
    exclamation_ratio = content.count("!") / max(1, len(content))
    clickbait_val = clickbait_score(content)
    source_cred = source_cred_score(domain)

    row = {
        "content": content,
        "source_domain": domain,
        "headline_len": headline_len,
        "punct_count": punct_count,
        "upper_ratio": upper_ratio,
        "exclamation_ratio": exclamation_ratio,
        "clickbait_score": clickbait_val,
        "age_days": age_days,
        "source_cred": source_cred,
    }
    return pd.DataFrame([row])

def predict_text(text, source=None, publish_date=None):
    X = make_input_row(text, source=source, publish_date=publish_date)
    proba = pipeline.predict_proba(X)[0]
    pred = pipeline.predict(X)[0]
    label = "Real" if pred == 1 else "Fake"
    confidence = float(max(proba) * 100)
    return {"label": label, "confidence": round(confidence, 2)}

if __name__ == "__main__":
    print(predict_text(
        "Scientists discover water on Mars!",
        source="nytimes.com",
        publish_date="2024-10-01"
    ))
