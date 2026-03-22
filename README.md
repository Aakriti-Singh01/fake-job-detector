# 🕵️ Fake Job Detection System

An intelligent web application that detects whether a job posting is **Real or Fraudulent** using a combination of **Machine Learning, Rule-Based Intelligence, and Explainable AI (XAI)**.

🔗 **Live App:** https://fake-job-detector-aakriti01.streamlit.app/
📦 **Repository:** https://github.com/Aakriti-Singh01/fake-job-detector

---

## 🚀 Overview

Online job scams are increasing rapidly, especially on informal platforms. This project aims to tackle that problem by building a **hybrid AI system** that analyzes job descriptions and flags potential fraud.

Unlike basic classifiers, this system:

* Combines **ML predictions + domain-specific fraud rules**
* Provides **human-readable explanations**
* Highlights **key risk signals** behind each prediction

---

## ✨ Features

* 🔍 **Fraud Detection Engine**
  Classifies job postings as *Real* or *Fraudulent*

* 🧠 **Hybrid Intelligence**
  Combines:

  * XGBoost Machine Learning model
  * Custom fraud detection rules

* 📊 **Explainable AI (SHAP)**
  Shows feature impact and reasoning behind predictions

* ⚠️ **Risk Scoring System**
  Provides:

  * Risk Score
  * Risk Level (Low / Medium / High)

* 🧾 **AI Reasoning Output**
  Generates human-friendly explanations for decisions

* 🌐 **Deployed Web App**
  Built using Streamlit and deployed on Streamlit Cloud

---

## 🛠️ Tech Stack

* **Frontend/UI:** Streamlit
* **Machine Learning:** XGBoost, Scikit-learn
* **Data Processing:** Pandas, NumPy
* **Model Explainability:** SHAP (via XGBoost contribs)
* **Deployment:** Streamlit Cloud

---

## 🧠 How It Works

1. **Text Input**
   User pastes a job description

2. **Feature Engineering**
   Extracts:

   * Spam keywords
   * Salary patterns
   * Contact details (email/phone)
   * Urgency signals
   * Suspicious company indicators

3. **ML Prediction**

   * TF-IDF vectorization
   * XGBoost model predicts fraud probability

4. **Rule-Based Scoring**

   * Applies domain-specific fraud rules
   * Generates final risk score

5. **Final Output**

   * Verdict: Real / Fraud
   * Risk Level
   * Explanation
   * Feature importance (SHAP)

---

## 📸 Sample Output

* ✅ Likely Real Job
* 🚨 Fraudulent Job
* 📊 Feature Impact Chart
* 🔎 Key  Drivers
<img width="913" height="841" alt="image" src="https://github.com/user-attachments/assets/6f01fe56-31fe-451c-8294-b5afa3bafd83" />

---

## ⚙️ Installation & Setup

```bash
# Clone the repository
git clone https://github.com/Aakriti-Singh01/fake-job-detector.git

# Navigate to project
cd fake-job-detector

# Create virtual environment
python -m venv venv

# Activate environment
venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app/streamlit_app.py
```

---

## 📂 Project Structure

```
fake-job-detector/
│
├── app/
│   └── streamlit_app.py
│
├── models/
│   ├── xgb_model.pkl
│   └── vectorizer.pkl
│
├── src/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── train_model.py
│   └── save_model.py
│
├── requirements.txt
└── README.md
```

---

## 🎯 Key Highlights

* Hybrid approach (ML + Rules) improves real-world detection
* Explainable outputs increase trust and transparency
* Designed with practical fraud patterns (not just academic data)
* Fully deployed and accessible online

---

## 📌 Future Improvements

* Improve dataset with real-world scam examples
* Add company verification API
* Enhance NLP with deep learning models (BERT)
* User feedback loop for continuous learning

---

## 👩‍💻 Author

**Aariti Singh**
**AI & Data Science | ML Engineer | Data Analytics**

---

## ⭐ If you found this project useful

Give it a ⭐ on GitHub and feel free to contribute!
