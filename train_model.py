"""Train a simple NLP sentiment model using the NLTK movie reviews dataset."""
import pickle
import random
import nltk
from nltk.corpus import movie_reviews
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

print("Downloading/checking the NLTK movie review dataset...")
nltk.download("movie_reviews")

documents = []
for category in movie_reviews.categories():
    for fileid in movie_reviews.fileids(category):
        documents.append((movie_reviews.raw(fileid), category))

random.seed(42)
random.shuffle(documents)
X = [review for review, label in documents]
y = [label for review, label in documents]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

model = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True, ngram_range=(1, 2), min_df=2, max_features=10000
    )),
    ("classifier", LogisticRegression(max_iter=1000))
])

print("Training the model...")
model.fit(X_train, y_train)
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"\nTest accuracy: {accuracy * 100:.2f}%")
print("\nClassification report:")
print(classification_report(
    y_test, predictions, labels=["neg", "pos"],
    target_names=["Negative", "Positive"], zero_division=0
))

with open("sentiment_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("\nSaved model as sentiment_model.pkl")
print("Now run: python app.py")
