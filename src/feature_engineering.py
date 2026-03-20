import pandas as pd
import re


def add_spam_score(df):
    """
    Count presence of spam keywords
    """
    spam_keywords = [
        "urgent", "immediate", "apply now", "limited",
        "earn money", "no experience", "whatsapp",
        "join within", "hiring now", "work from home",
        "walk-in", "hurry", "shortlisted"
    ]

    def count_spam(text):
        return sum(1 for word in spam_keywords if word in text)

    df['spam_score'] = df['combined_text'].apply(count_spam)

    return df


def add_salary_flag(df):
    """
    Detect suspicious salary patterns
    """
    def has_salary_pattern(text):
        patterns = [
            r'\$\d+',
            r'₹\d+',
            r'\d+\s*lpa',        # 🔥 NEW (important)
            r'\d+\s*per\s*week',
            r'\d+\s*per\s*day'
        ]
        return int(any(re.search(p, text) for p in patterns))

    df['salary_flag'] = df['combined_text'].apply(has_salary_pattern)

    return df


def add_text_length(df):
    """
    Add length of text
    """
    df['text_length'] = df['combined_text'].apply(len)
    return df

def add_missing_info_flag(df):
    def check_missing(text):
        text = text.lower()

        missing_company = "company:" not in text or len(text.split("company:")[-1].strip()) < 10
        missing_requirements = "requirements:" not in text or len(text.split("requirements:")[-1].strip()) < 20

        return int(missing_company or missing_requirements)

    df['missing_info'] = df['combined_text'].apply(check_missing)

    return df


# 🔥 IMPORTANT: Apply advanced features to dataframe
def add_advanced_features(df):
    """
    Extract real-world fraud signals
    """

    def extract(text):
        text = text.lower()

        # Email detection
        email_flag = 1 if ("gmail.com" in text or "yahoo.com" in text) else 0

        # Phone number detection (10-digit)
        import re
        phone_flag = 1 if re.search(r'\b\d{10}\b', text) else 0

        # Urgency signals
        urgency_score = sum(word in text for word in ["urgent", "immediate", "hurry", "now"])

        # Remote job flag
        remote_flag = 1 if "work from home" in text else 0

        return pd.Series([email_flag, phone_flag, urgency_score, remote_flag])

    df[['email_flag', 'phone_flag', 'urgency_score', 'remote_flag']] = \
        df['combined_text'].apply(extract)

    return df


def feature_engineering(df):
    """
    Apply all feature engineering steps
    """
    df = add_spam_score(df)
    print("✅ Spam score added")

    df = add_salary_flag(df)
    print("✅ Salary flag added")

    df = add_text_length(df)
    print("✅ Text length added")

    df = add_missing_info_flag(df)
    print("✅ Missing info flag added")

    # 🔥 NEW (CRITICAL)
    df = add_advanced_features(df)
    print("✅ Advanced features added")

    return df


if __name__ == "__main__":
    from preprocessing import preprocess_data

    file_path = "data/fake_job_postings.csv"
    df = preprocess_data(file_path)

    df = feature_engineering(df)

    print("\n🔹 Final Dataset Shape:", df.shape)
    print("\n🔹 Sample Data:")
    print(df.head())