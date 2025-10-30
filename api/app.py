# api/app.py
from flask import Flask, request, jsonify
import joblib
import pandas as pd
from datetime import datetime
from src.preprocess import clean_text, extract_domain, source_cred_score

app = Flask(__name__)

# -----------------------------
# Load trained model
# -----------------------------
MODEL_PATH = "model/pipeline_with_meta.pkl"
pipeline = joblib.load(MODEL_PATH)

# -----------------------------
# Root route
# -----------------------------
@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "message": "📰 Fake News Detector API",
        "usage": "POST to /predict with JSON {text, source (optional), publish_date (optional)}"
    })

# -----------------------------
# Predict route
# -----------------------------
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)

    text = data.get("text", "").strip()
    source = data.get("source", "")
    publish_date = data.get("publish_date", None)

    if not text:
        return jsonify({"error": "Missing 'text' field"}), 400

    # Clean + prepare
    content = clean_text(text)
    domain = extract_domain(source)
    cred = source_cred_score(domain)

    # Compute derived metadata
    headline_len = len(content.split())
    punct_count = sum(1 for c in content if c in "!?.,")
    upper_ratio = sum(1 for c in content if c.isupper()) / max(1, len(content))
        # Add these near where you compute upper_ratio, etc.
    exclamation_ratio = content.count('!') / max(1, len(content))
    clickbait_words = [
        "shocking", "breaking", "amazing", "you won't believe", "exclusive",
        "unbelievable", "surprising", "secret", "revealed", "crazy", "must see"
    ]
    clickbait_score = sum(1 for word in clickbait_words if word in content.lower())

    if publish_date:
        try:
            pd_date = pd.to_datetime(publish_date)
            age_days = (pd.Timestamp.now() - pd_date).days
        except:
            age_days = 99999
    else:
        age_days = 99999

    # Prepare input row
    X = pd.DataFrame([{
        "content": content,
        "source_domain": domain,
        "headline_len": headline_len,
        "punct_count": punct_count,
        "upper_ratio": upper_ratio,
        "exclamation_ratio": exclamation_ratio,
        "clickbait_score": clickbait_score,
        "age_days": age_days,
        "source_cred": cred
    }])

    # Predict
    pred = pipeline.predict(X)[0]
    proba = pipeline.predict_proba(X)[0]
    label = "Real" if pred == 1 else "Fake"
    confidence = round(float(max(proba)) * 100, 2)

    return jsonify({"label": label, "confidence": confidence})

if __name__ == "__main__":
    app.run(debug=True)
