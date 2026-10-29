# Movie Review Sentiment Analyzer (NLP Mini Project)

## Requirements
- Python 3.10–3.12 recommended
- Internet connection for the first-time dataset download

## Run on Windows
1. Extract this folder.
2. Open the folder in VS Code.
3. Open Terminal > New Terminal.
4. Create a virtual environment (recommended):
   `python -m venv .venv`
5. Activate it:
   PowerShell: `.\.venv\Scripts\Activate.ps1`
   If PowerShell blocks activation, use Command Prompt and run:
   `.venv\Scripts\activate.bat`
6. Install libraries:
   `python -m pip install -r requirements.txt`
7. Train the NLP model (first run downloads the NLTK dataset):
   `python train_model.py`
8. Start the web application:
   `python app.py`
9. Open Chrome and visit:
   `http://127.0.0.1:5000`

## How it works
The training script uses the NLTK movie_reviews dataset (2,000 labelled reviews).
TF-IDF converts review text into numeric features; Logistic Regression learns to
classify reviews as positive (`pos`) or negative (`neg`). The trained pipeline is
saved to `sentiment_model.pkl`. Flask serves the web page and prediction results.

## Project files
- `app.py`: Flask server and prediction route
- `train_model.py`: downloads data, trains model, prints test metrics, saves model
- `templates/index.html`: application interface
- `static/style.css`: responsive styling
- `requirements.txt`: Python dependencies

This is an educational demo. Model confidence is not a guarantee of correctness.
