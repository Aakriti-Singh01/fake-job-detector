import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from scipy.sparse import hstack
from xgboost import XGBClassifier



def load_processed_data():
    from feature_engineering import feature_engineering
    from preprocessing import preprocess_data

    file_path = "data/fake_job_postings.csv"
    df = preprocess_data(file_path)
    df = feature_engineering(df)

    return df


def vectorize_text(X_text):
    """
    Convert text to TF-IDF features
    """
    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words='english'
    )

    X_tfidf = vectorizer.fit_transform(X_text)

    return X_tfidf, vectorizer


def combine_features(X_tfidf, df):
    """
    Combine TF-IDF with structured + advanced features
    """
    structured_features = df[
        [
            'telecommuting',
            'has_company_logo',
            'has_questions',
            'spam_score',
            'salary_flag',
            'text_length',
            'missing_info',
            'email_flag',
            'phone_flag',
            'urgency_score',
            'remote_flag'
        ]
    ].values

    X_final = hstack([X_tfidf, structured_features])

    return X_final


def train_model(X, y):
    """
    Train Logistic Regression model
    """
    model = LogisticRegression(max_iter=2000,class_weight='balanced')

    model.fit(X, y)

    return model

def train_xgboost(X, y):
    """
    Train XGBoost model
    """
    model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=10,  # handles imbalance
        eval_metric='logloss'
    )

    model.fit(X, y)

    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance
    """
    y_pred = model.predict(X_test)

    print("\n🔹 Classification Report:\n")
    print(classification_report(y_test, y_pred))

    print("\n🔹 Confusion Matrix:\n")
    print(confusion_matrix(y_test, y_pred))

def evaluate_with_threshold(model, X_test, y_test, threshold=0.3):
    y_probs = model.predict_proba(X_test)[:, 1]
    y_pred = (y_probs >= threshold).astype(int)

    from sklearn.metrics import classification_report, confusion_matrix

    print(f"\n🔹 Threshold = {threshold}")
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix:\n")
    print(confusion_matrix(y_test, y_pred))


if __name__ == "__main__":
    df = load_processed_data()

    X_text = df['combined_text']
    y = df['fraudulent']

    # Split BEFORE vectorization (important)
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        X_text, y, test_size=0.2, random_state=42, stratify=y
    )

    # Vectorize
    X_train_tfidf, vectorizer = vectorize_text(X_train_text)
    X_test_tfidf = vectorizer.transform(X_test_text)

    # Combine structured features
    X_train = combine_features(X_train_tfidf, df.loc[X_train_text.index])
    X_test = combine_features(X_test_tfidf, df.loc[X_test_text.index])

    # -----  Logistic Regression -----
    model_lr = train_model(X_train, y_train)
    print("✅ Model trained")

    #evaluate
    evaluate_model(model_lr, X_test, y_test)
    # Threshold tuning
    evaluate_with_threshold(model_lr, X_test, y_test, threshold=0.3)


    # -------- XGBoost --------
    model_xgb = train_xgboost(X_train, y_train)
    print("\n✅ XGBoost model trained")

    evaluate_model(model_xgb, X_test, y_test)
    evaluate_with_threshold(model_xgb, X_test, y_test, threshold=0.3)