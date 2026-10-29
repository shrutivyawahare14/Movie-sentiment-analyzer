
from flask import Flask, render_template, request
import os
import pickle
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

app = Flask(__name__)

MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "sentiment_model.pkl"
)

analyzer = SentimentIntensityAnalyzer()

# Load your trained Logistic Regression model
model = None

if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)


def predict_sentiment(review):
    # Analyze sentiment using VADER
    scores = analyzer.polarity_scores(review)
    compound = scores["compound"]

    # Calculate relative positive and negative scores
    total = scores["pos"] + scores["neg"]

    if total > 0:
        positive_score = scores["pos"] / total * 100
        negative_score = scores["neg"] / total * 100
    else:
        positive_score = 50.0
        negative_score = 50.0

    # Use VADER when the sentiment is clear
    if compound >= 0.05:
        sentiment = "Positive"
        confidence = positive_score

    elif compound <= -0.05:
        sentiment = "Negative"
        confidence = negative_score

    # If unclear, use the trained machine learning model
    elif model is not None:
        prediction = model.predict([review])[0]
        probabilities = model.predict_proba([review])[0]
        classes = list(model.classes_)

        positive_score = float(
            probabilities[classes.index("pos")] * 100
        )
        negative_score = float(
            probabilities[classes.index("neg")] * 100
        )

        sentiment = (
            "Positive" if prediction == "pos" else "Negative"
        )
        confidence = max(positive_score, negative_score)

    else:
        sentiment = "Neutral"
        confidence = 50.0

    return {
        "sentiment": sentiment,
        "confidence": round(confidence, 1),
        "positive_probability": round(positive_score, 1),
        "negative_probability": round(negative_score, 1),
    }


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    review = ""
    error = None

    if request.method == "POST":
        review = request.form.get("review", "").strip()

        if not review:
            error = "Please enter a movie review first."
        else:
            result = predict_sentiment(review)
            result["review"] = review

    return render_template(
        "index.html",
        result=result,
        review=review,
        error=error
    )


if __name__ == "__main__":
    app.run(debug=True)
