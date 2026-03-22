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
st.write("Detect whether a job posting is **Real or Fraudulent** using Machine Learning + Rule Intelligence.")

# Load model
model = joblib.load("models/xgb_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")


# ---------------- FEATURE ENGINEERING ---------------- #
def create_features(text):
    text_lower = text.lower()

    spam_keywords = [
        "urgent", "immediate", "apply now", "limited",
        "earn money", "no experience", "whatsapp",
        "join today", "hiring now", "work from home",
        "walk-in", "hurry", "shortlisted", "asap"
    ]

    spam_score = sum(word in text_lower for word in spam_keywords)

    salary_flag = int(bool(re.search(r'\$\d+|₹\d+|\d+\s*lpa', text_lower)))
    high_salary_flag = int(bool(re.search(r'(₹\s?\d{2,}|\d+\s?lpa)', text_lower)))

    buzzwords = ["ai", "blockchain", "nft", "web3", "crypto", "generative ai"]
    buzzword_count = sum(word in text_lower for word in buzzwords)

    text_length = len(text)
    missing_info = 1 if len(text.strip()) < 80 else 0

    email_flag = 1 if ("gmail.com" in text_lower or "yahoo.com" in text_lower) else 0
    phone_flag = 1 if re.search(r'\b\d{10}\b', text) else 0

    urgency_score = sum(word in text_lower for word in ["urgent", "immediate", "hurry", "now", "asap"])

    remote_flag = 1 if "work from home" in text_lower else 0

    suspicious_company_words = ["solutions", "global", "international", "private", "ltd"]
    company_suspicious = sum(word in text_lower for word in suspicious_company_words)

    return {
        "telecommuting": 0,
        "has_company_logo": 0,
        "has_questions": 0,
        "spam_score": spam_score,
        "salary_flag": salary_flag,
        "high_salary_flag": high_salary_flag,
        "buzzword_count": buzzword_count,
        "text_length": text_length,
        "missing_info": missing_info,
        "email_flag": email_flag,
        "phone_flag": phone_flag,
        "urgency_score": urgency_score,
        "remote_flag": remote_flag,
        "company_suspicious": company_suspicious
    }


# ---------------- RULE ENGINE ---------------- #
def fraud_rule_engine(features, model_prob):
    score = 0

    if model_prob > 0.4:
        score += 2

    if features["email_flag"] and features["phone_flag"]:
        score += 4

    if features["urgency_score"] >= 1 and features["salary_flag"]:
        score += 4

    if features["spam_score"] >= 2:
        score += 2

    if features["company_suspicious"] >= 2:
        score += 2

    if features["remote_flag"] and features["salary_flag"]:
        score += 2

    if features["high_salary_flag"] and features["buzzword_count"] >= 2:
        score += 3

    if features["phone_flag"]:
        score += 1.5

    if features["email_flag"]:
        score += 1.5

    if features["missing_info"]:
        score += 2

    if score >= 7:
        return "fraud", score, "HIGH RISK 🚨"
    elif score >= 4:
        return "fraud", score, "MEDIUM RISK ⚠️"
    else:
        return "real", score, "LOW RISK ✅"


# ---------------- EXPLANATION ---------------- #
def generate_reasoning(features, result):
    reasons = []

    if features["email_flag"]:
        reasons.append("Uses personal email instead of official domain")

    if features["phone_flag"]:
        reasons.append("Includes direct phone number")

    if features["urgency_score"] >= 1:
        reasons.append("Creates urgency (ASAP / immediate hiring)")

    if features["salary_flag"]:
        reasons.append("Highlights salary prominently")

    if features["buzzword_count"] >= 2:
        reasons.append("Uses hype tech buzzwords (AI/Web3/Blockchain)")

    if features["remote_flag"]:
        reasons.append("Work-from-home offer")

    if result == "fraud":
        summary = "⚠️ This job shows strong scam patterns."
    else:
        summary = "✅ This job looks relatively safe."

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
        # -------- ML PIPELINE -------- #
        X_text = vectorizer.transform([user_input])
        features = create_features(user_input)
        df_features = pd.DataFrame([features])

        X_final = combine_features(X_text, df_features)

        feature_names = list(vectorizer.get_feature_names_out()) + list(features.keys())

        dmatrix = xgb.DMatrix(X_final)
        prob = model.get_booster().predict(dmatrix)[0]

        result, score, risk_level = fraud_rule_engine(features, prob)

        summary, reasons = generate_reasoning(features, result)

        # -------- FINAL OUTPUT -------- #
        st.subheader("🧾 Final Verdict")

        if result == "fraud":
            st.error("🚨 Fraudulent Job")
        else:
            st.success("✅ Likely Real Job")

        st.write(f"**Risk Level:** {risk_level}")
        st.write(f"**Risk Score:** {score}")
        st.progress(min(score / 10, 1.0))

        # -------- REASONING -------- #
        st.subheader("🧠 AI Reasoning")
        st.write(summary)

        for r in reasons:
            st.markdown(f"- {r}")

        # -------- SHAP -------- #
        st.subheader("🔍 Model Explanation")

        top_features = get_shap_explanation(model, X_final, feature_names)

        shap_df = pd.DataFrame(top_features, columns=["Feature", "Impact"])

        shap_df["Type"] = shap_df["Impact"].apply(
            lambda x: "🚨 Fraud Signal" if x > 0 else "✅ Legit Signal"
        )

        shap_df = shap_df.sort_values(by="Impact", key=abs, ascending=False)

        # 📊 Chart
        st.markdown("### 📊 Feature Impact Overview")
        st.bar_chart(shap_df.set_index("Feature")["Impact"])

        # 📋 Table
        st.markdown("### 📋 Detailed Breakdown")
        st.dataframe(shap_df)

        

        # 🚀 Drivers
        st.subheader("🚀 Key Drivers")
        for name, value in top_features[:3]:
            if value > 0:
                st.markdown(f"🔴 **{name}** indicates fraud (+{value:.3f})")
            else:
                st.markdown(f"🟢 **{name}** supports legitimacy ({value:.3f})")