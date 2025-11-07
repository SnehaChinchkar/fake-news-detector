import os
import streamlit as st
import requests
from datetime import datetime

# API_URL = "http://127.0.0.1:5000/predict"
API_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:5000") + "/predict"

st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

st.title("📰 Fake News Detector")
st.markdown("Enter a news article below and see if it’s likely **Fake or Real**.")

text = st.text_area("🗞️ News content:", height=200)
source = st.text_input("🌐 Source domain (optional):", "")
publish_date = st.date_input("📅 Publish date (optional):", value=None)

if st.button("🔍 Analyze"):
    if not text.strip():
        st.warning("Please enter some text to analyze.")
    else:
        payload = {"text": text, "source": source or None}
        if publish_date:
            payload["publish_date"] = publish_date.strftime("%Y-%m-%d")

        try:
            res = requests.post(API_URL, json=payload)
            if res.status_code == 200:
                data = res.json()
                label = data.get("label", "?")
                conf = data.get("confidence", 0)

                if label.lower() == "real":
                    st.success(f" Predicted as **Real** ({conf:.2f}% confidence)")
                else:
                    st.error(f" Predicted as **Fake** ({conf:.2f}% confidence)")
            else:
                st.error(f"API Error: {res.text}")

        except Exception as e:
            st.error(f"Connection failed: {e}")
