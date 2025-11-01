import os
import shutil
import kagglehub

from create_kaggle_config import ensure_kaggle_config_from_env

KAGGLE_DATASET = "clmentbisaillon/fake-and-real-news-dataset"
# DATA_DIR = os.path.abspath("data")
# MODEL_PATH = os.path.abspath("storage/fake_news_model.joblib")
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data")
MODEL_PATH = os.path.join(ROOT_DIR, "storage", "fake_news_model.joblib")

def prepare_dirs():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

def download_dataset():
    print("Downloading dataset with kagglehub...")
    # path = kagglehub.dataset_download(KAGGLE_DATASET, force=False)
    path = kagglehub.dataset_download(KAGGLE_DATASET)


    if os.path.isfile(path) and path.endswith(".zip"):
        import zipfile
        os.makedirs(DATA_DIR, exist_ok=True)
        with zipfile.ZipFile(path, "r") as z:
            z.extractall(DATA_DIR)
        return DATA_DIR
    else:
        return path

def run_training():
    prepare_dirs()
    ensure_kaggle_config_from_env()
    dataset_folder = download_dataset()
    print("Dataset ready at:", dataset_folder)

    from src.train_model import train_and_save
    train_and_save(dataset_folder, MODEL_PATH)
    print("Model saved to:", MODEL_PATH)

if __name__ == "__main__":
    run_training()
