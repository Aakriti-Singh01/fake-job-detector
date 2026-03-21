import streamlit as st
import joblib
import numpy as np
import xgboost as xgb
import sys
import os
import pandas as pd
import re

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from train_model import combine_features

st.set_page_config(page_title="Fake Job Detector", layout="wide")

st.title("🕵️ Fake Job Detection System")
st.write("Detect whether a job posting is **Real or Fraudulent** using Machine Learning + Rule Intelligence + External Validation.")

# Load model
model = joblib.load("models/xgb_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")


# ---------------- FEATURE ENGINEERING ---------------- #
def create_features(text):
    text_lower = text.lower()

    spam_keywords = [
        "urgent", "immediate", "apply now", "limited",
        "earn money", "no experience", "whatsapp",
        "join within", "hiring now", "work from home",
        "walk-in", "hurry", "shortlisted"
    ]

    spam_score = sum(word in text_lower for word in spam_keywords)

    salary_flag = int(bool(re.search(r'\$\d+|₹\d+|\d+\s*lpa', text_lower)))

    text_length = len(text)

    missing_info = 1 if len(text.strip()) < 50 else 0

    email_flag = 1 if ("gmail.com" in text_lower or "yahoo.com" in text_lower) else 0

    domain_flag = 0
    if "@" in text_lower:
        if not any(domain in text_lower for domain in [".com", ".org", ".in", ".co"]):
            domain_flag = 1

    phone_flag = 1 if re.search(r'\b\d{10}\b', text) else 0

    urgency_score = sum(word in text_lower for word in ["urgent", "immediate", "hurry", "now"])

    remote_flag = 1 if "work from home" in text_lower else 0

    suspicious_company_words = ["solutions", "global", "international", "private", "ltd"]
    company_suspicious = sum(word in text_lower for word in suspicious_company_words)

    return {
        "telecommuting": 0,
        "has_company_logo": 0,
        "has_questions": 0,
        "spam_score": spam_score,
        "salary_flag": salary_flag,
        "text_length": text_length,
        "missing_info": missing_info,
        "email_flag": email_flag,
        "domain_flag": domain_flag,
        "phone_flag": phone_flag,
        "urgency_score": urgency_score,
        "remote_flag": remote_flag,
        "company_suspicious": company_suspicious
    }


# ---------------- EXTERNAL VALIDATION ---------------- #
def external_validation(text):
    text_lower = text.lower()

    issues = []

    if "gmail.com" in text_lower:
        issues.append("Uses generic email instead of company domain")

    if "whatsapp" in text_lower:
        issues.append("Relies on WhatsApp communication")

    if not any(word in text_lower for word in ["www.", ".com", ".org"]):
        issues.append("No official company website mentioned")

    suspicious_words = ["global", "international", "solutions"]
    if sum(word in text_lower for word in suspicious_words) >= 2:
        issues.append("Company name looks generic/suspicious")

    return issues


# ---------------- RULE ENGINE ---------------- #
def fraud_rule_engine(features, model_prob):
    score = 0

    if model_prob > 0.5:
        score += 1

    if features["email_flag"] and features["phone_flag"]:
        score += 4

    if features["urgency_score"] >= 2 and features["salary_flag"]:
        score += 3

    if features["spam_score"] >= 3 and features["urgency_score"] >= 2:
        score += 2

    if features["company_suspicious"] >= 3 and features["email_flag"]:
        score += 2

    if features["phone_flag"]:
        score += 1

    if features["salary_flag"]:
        score += 1

    if features["remote_flag"]:
        score += 0.5

    if score >= 6:
        return "fraud", score, "HIGH RISK 🚨"
    elif score >= 3.5:
        return "fraud", score, "MEDIUM RISK ⚠️"
    else:
        return "real", score, "LOW RISK ✅"


# ---------------- LLM STYLE EXPLANATION ---------------- #
def generate_reasoning(features, external_issues, result):
    reasons = []

    if features["email_flag"]:
        reasons.append("Uses personal email (gmail/yahoo) instead of official domain")

    if features["phone_flag"]:
        reasons.append("Includes direct phone number for hiring")

    if features["urgency_score"] >= 2:
        reasons.append("Creates urgency (e.g., immediate joining, hurry)")

    if features["salary_flag"]:
        reasons.append("Mentions unusually high or highlighted salary")

    if features["remote_flag"]:
        reasons.append("Offers work-from-home (commonly used in scams)")

    reasons.extend(external_issues)

    if result == "fraud":
        summary = "⚠️ This job shows multiple scam-like patterns."
    else:
        summary = "✅ This job looks relatively safe with minor concerns."

    return summary, reasons


# ---------------- SHAP ---------------- #
def get_shap_explanation(model, X, feature_names):
    dmatrix = xgb.DMatrix(X)

    shap_values = model.get_booster().predict(
        dmatrix,
        pred_contribs=True
    )

    values = shap_values[0][:-1]

    feature_impact = list(zip(feature_names, values))

    feature_impact = sorted(
        feature_impact,
        key=lambda x: abs(x[1]),
        reverse=True
    )

    return feature_impact[:10]


# ---------------- UI ---------------- #
user_input = st.text_area("📄 Paste Job Description Here", height=250)

if st.button("🔍 Detect Fraud"):

    if user_input.strip() == "":
        st.warning("Please enter a job description")
    else:
        X_text = vectorizer.transform([user_input])

        features = create_features(user_input)
        df_features = pd.DataFrame([features])

        X_final = combine_features(X_text, df_features)

        feature_names = list(vectorizer.get_feature_names_out()) + list(features.keys())

        dmatrix = xgb.DMatrix(X_final)
        prob = model.get_booster().predict(dmatrix)[0]

        result, score, risk_level = fraud_rule_engine(features, prob)

        external_issues = external_validation(user_input)

        summary, reasons = generate_reasoning(features, external_issues, result)

        # -------- FINAL OUTPUT -------- #
        st.subheader("🧾 Final Verdict")

        if result == "fraud":
            st.error("🚨 Fraudulent Job")
        else:
            st.success("✅ Likely Real Job")

        st.write(f"**Risk Level:** {risk_level}")
        st.write(f"**Risk Score:** {score}")

        st.progress(min(score / 10, 1.0))

        # -------- LLM EXPLANATION -------- #
        st.subheader("🧠 AI Reasoning")

        st.write(summary)

        for r in reasons:
            st.markdown(f"- {r}")

        # -------- SHAP -------- #
        st.subheader("🔍 Model Explanation")

        top_features = get_shap_explanation(model, X_final, feature_names)

        shap_df = pd.DataFrame(top_features, columns=["Feature", "Impact"])

        shap_df[" Type"] = shap_df["Impact"].apply(
            lambda x: "Fraud Signal 🚨" if x > 0 else "Legit Signal ✅"
        )

        st.dataframe(
            shap_df.style
            .background_gradient("Impact", cmap="RdYlGn_r")
            .format({"Impact": "{:.3f}"})
        )

        # -------- TOP DRIVERS -------- #
        st.subheader("🚀 Key Drivers")

        for name, value in top_features[:3]:
            if value > 0:
                st.markdown(f"🔴 **{name}** indicates fraud (+{value:.3f})")
            else:
                st.markdown(f"🟢 **{name}** supports legitimacy ({value:.3f})")