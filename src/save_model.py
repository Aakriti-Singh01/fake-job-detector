import joblib
from train_model import (
    load_processed_data,
    train_xgboost,
    vectorize_text,
    combine_features
)
from sklearn.model_selection import train_test_split


def train_and_save():
    df = load_processed_data()

    X_text = df['combined_text']
    y = df['fraudulent']

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        X_text, y, test_size=0.2, random_state=42, stratify=y
    )

    # 🔹 TF-IDF
    X_train_tfidf, vectorizer = vectorize_text(X_train_text)

    # 🔹 Combine ALL features (text + structured + advanced)
    X_train = combine_features(
        X_train_tfidf,
        df.loc[X_train_text.index]
    )

    # 🔹 Train model
    model = train_xgboost(X_train, y_train)

    # 🔹 Save artifacts
    joblib.dump(model, "models/xgb_model.pkl")
    joblib.dump(vectorizer, "models/vectorizer.pkl")

    print("✅ Model and vectorizer saved!")


if __name__ == "__main__":
    train_and_save()