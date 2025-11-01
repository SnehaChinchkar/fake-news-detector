# Fake News Detector

## Installation
Install dependencies using:
```bash
pip install -r requirements.txt
```

## Dataset
This project uses the **Fake and Real News Dataset** from Kaggle:  
[Fake and Real News Dataset on Kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)

**Author:** Clément Bisaillon  
**License:** [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)  
The dataset is provided for non-commercial research and educational purposes.

## Data Processing Features
- Custom dataset (`custom_short_texts.csv`) for improved short text performance
- Metadata features from news content
- Optimized feature selection for reduced overfitting
- Testing via cURL and Python scripts
- Streamlit frontend interface

## Model Improvements
- Reduce weight of custom samples
- Ignore very short texts in preprocessing (only for custom dataset)

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
