import os
import pathlib
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, accuracy_score


def main():
    root = pathlib.Path(__file__).parent
    data_path = root / "data" / "news_headlines.csv"
    model_dir = root / "model"
    model_dir.mkdir(exist_ok=True)

    if not data_path.exists():
        print(f"Training data not found at {data_path}")
        return

    df = pd.read_csv(data_path)
    df = df.dropna(subset=["text", "label"]).sample(frac=1, random_state=42)

    X = df["text"].astype(str)
    y = df["label"].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_df=0.9)
    clf = LogisticRegression(solver="liblinear")

    pipeline = make_pipeline(vectorizer, clf)
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    probs = pipeline.predict_proba(X_test)

    print("Accuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds))

    # Save pipeline directly
    joblib.dump(pipeline, model_dir / "pipeline.pkl")
    print(f"Saved trained pipeline to {model_dir / 'pipeline.pkl'}")


if __name__ == "__main__":
    main()
