import pandas as pd


def handle_missing_values(df):
    """
    Fill missing values appropriately
    """
    # Text columns → fill with empty string
    text_cols = ['title', 'description', 'requirements', 'company_profile']
    for col in text_cols:
        df[col] = df[col].fillna("")

    # Structured columns → fill with 0
    structured_cols = ['telecommuting', 'has_company_logo', 'has_questions']
    for col in structured_cols:
        df[col] = df[col].fillna(0).astype(int)

    return df


def combine_text_features(df):
    """
    Combine text columns into a single feature with labels
    """
    df['combined_text'] = (
        "TITLE: " + df['title'] + " " +
        "DESCRIPTION: " + df['description'] + " " +
        "REQUIREMENTS: " + df['requirements'] + " " +
        "COMPANY: " + df['company_profile']
    )
    return df


def select_features(df):
    """
    Keep only required columns
    """
    df = df[[
        'combined_text',
        'telecommuting',
        'has_company_logo',
        'has_questions',
        'fraudulent'
    ]]
    return df


def preprocess_data(file_path):
    """
    Full preprocessing pipeline
    """
    df = pd.read_csv(file_path, encoding='utf-8', engine='python')

    print("✅ Data loaded")

    df = handle_missing_values(df)
    print("✅ Missing values handled")

    df = combine_text_features(df)
    print("✅ Text features combined")

    # 🔥 NEW: normalize text (important for feature engineering)
    df['combined_text'] = df['combined_text'].str.lower()

    df = select_features(df)
    print("✅ Features selected")

    return df


if __name__ == "__main__":
    file_path = "data/fake_job_postings.csv"

    df = preprocess_data(file_path)

    print("\n🔹 Final Dataset Shape:", df.shape)
    print("\n🔹 Sample Data:")
    print(df.head())