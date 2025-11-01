# api/app.py
import os
import joblib
import pandas as pd
from datetime import datetime
from flask import Flask, request, jsonify

from create_kaggle_config import ensure_kaggle_config_from_env
from src.preprocess import clean_text, extract_domain, source_cred_score
from src.train_model import train_model_with_meta
from flask_cors import CORS

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(ROOT_DIR, "storage", "fake_news_model.joblib")

app = Flask(__name__)
ALLOWED_ORIGIN = os.getenv("ALLOWED_ORIGIN", "*")  

CORS(app, origins=[ALLOWED_ORIGIN])
ensure_kaggle_config_from_env()  


@app.route("/")
def health():
    return "📰 Fake News Detector API is running with metadata support"


@app.route("/init", methods=["POST"])
def init_train():
    """
    Trigger dataset download + model training.
    Protected by INIT_TOKEN environment variable.
    """
    INIT_TOKEN = os.getenv("INIT_TOKEN")
    auth = request.headers.get("Authorization", "")

    if INIT_TOKEN and auth != f"Bearer {INIT_TOKEN}":
        return jsonify({"error": "unauthorized"}), 401

    try:
        train_model_with_meta(MODEL_PATH)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({"status": "training complete"})



@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict whether given text (and optional metadata) is Fake or Real news.
    """
    if not os.path.exists(MODEL_PATH):
        return jsonify({"error": "model not found. run /init first."}), 400

    data = request.get_json(force=True)

    text = data.get("text", "").strip()
    source = data.get("source", "")
    publish_date = data.get("publish_date", None)

    if not text:
        return jsonify({"error": "Missing 'text' field"}), 400

    # Preprocess text and extract metadata
    content = clean_text(text)
    domain = extract_domain(source)
    cred = source_cred_score(domain)

    headline_len = len(content.split())
    punct_count = sum(1 for c in content if c in "!?.,")
    upper_ratio = sum(1 for c in content if c.isupper()) / max(1, len(content))
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
        except Exception:
            age_days = 99999
    else:
        age_days = 99999

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

    # Load model and predict
    pipeline = joblib.load(MODEL_PATH)
    pred = pipeline.predict(X)[0]
    proba = pipeline.predict_proba(X)[0]

    label = "Real" if pred == 1 else "Fake"
    confidence = round(float(max(proba)) * 100, 2)

    return jsonify({"label": label, "confidence": confidence})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
