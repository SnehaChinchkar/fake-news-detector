from predict import predict_text

texts = [
    "NASA confirms discovery of new exoplanet.",
    "The vaccine rollout begins next month.",
    "Aliens have taken over the White House.",
    "Stock market crashes due to global coffee shortage.",
    "Scientists discover cure for aging."
]

for text in texts:
    print(f"{text} -> {predict_text(text)}")
