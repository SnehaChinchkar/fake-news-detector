# Fake News Detector
## Installation
Install dependencies using:
```bash
pip install -r requirements.txt
```

## Usage
### Training the Model
```bash
python src/train_model.py
```

### Running the Application
#### API (Flask)
```bash
python -m api.app
```

#### Web Interface (Streamlit)
```bash
streamlit run src/app.py
```

## Dataset
The project uses the [Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset?select=True.csv) from Kaggle.

### Data Processing
- Created custom dataset (`custom_short_texts.csv`) from original dataset to improve performance on short texts
- Enhanced model with metadata features extracted from news content
- Optimized feature selection to prevent overfitting and improve generalization
- Tested using both cURL commands and Python scripts
- Added Streamlit frontend for user interaction 
