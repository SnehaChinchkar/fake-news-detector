# Fake News Detector
## Installing the dependencies:
use requirements.txt

## Training the Model
```bash
python src/train_model.py
```

## Running the Application
### API (Flask)
```bash
python -m api.app
```

### User Interface (Streamlit)
```bash
streamlit run src/app.py
```

## Dataset
This project uses the [Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset?select=True.csv) from Kaggle.

what i did was initially train model on original dataset but it was performing low on small text size so i created a custom dataset(custom_short_texts.csv) from original dataset of fake and real news using create_short_dataset. After that I made advancement in my model so that it uses metadata (some of it obtained by news content itself). Later, I dropped few columns in training as they were causing poor performance, as user will not always have all the data used in training as input (overfititng or underfitting cause this?) but it was giving poor generalisation. I tested the model using bash commands such as :
curl ..
later on i also tested using a temporary python file.
after that i added the frontend ui. 
