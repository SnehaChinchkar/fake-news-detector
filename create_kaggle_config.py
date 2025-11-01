# create_kaggle_config.py
import os, json, stat

def ensure_kaggle_config_from_env():
    username = os.getenv("KAGGLE_USERNAME")
    key = os.getenv("KAGGLE_KEY")

    if not username or not key:
        # no env vars set — caller should handle fallback (local testing)
        return False

    kaggle_dir = os.path.expanduser("~/.kaggle")
    os.makedirs(kaggle_dir, exist_ok=True)
    kaggle_json_path = os.path.join(kaggle_dir, "kaggle.json")

    kaggle_data = {"username": username, "key": key}
    with open(kaggle_json_path, "w") as f:
        json.dump(kaggle_data, f)
    os.chmod(kaggle_json_path, stat.S_IRUSR | stat.S_IWUSR)  # 0o600
    return True

if __name__ == "__main__":
    r = ensure_kaggle_config_from_env()
    print("Kaggle config written from env:", r)
