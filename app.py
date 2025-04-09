import streamlit as st
import pickle

# App Title
st.set_page_config(page_title="Email Classification App")
st.title("📧 Smart Email Categorizer")
st.write("This app classifies your email using Random Forest, SVM, and Naive Bayes models.")

# Load models and vectorizer
@st.cache_resource
def load_all():
    with open("models/rf_model.pkl", "rb") as f:
        rf = pickle.load(f)
    with open("models/svm_model.pkl", "rb") as f:
        svm = pickle.load(f)
    with open("models/nb_model.pkl", "rb") as f:
        nb = pickle.load(f)
    with open("models/tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open("models/label_encoder.pkl", "rb") as f:
        encoder = pickle.load(f)
    return rf, svm, nb, vectorizer, encoder

rf_model, svm_model, nb_model, tfidf_vectorizer, label_encoder = load_all()

# Input email content
email_input = st.text_area("📥 Enter your email content here:", height=250)

# Predict
if st.button("🔍 Classify"):
    if not email_input.strip():
        st.warning("Please enter an email to classify.")
    else:
        email_tfidf = tfidf_vectorizer.transform([email_input])

        rf_pred = label_encoder.inverse_transform(rf_model.predict(email_tfidf))[0]
        svm_pred = label_encoder.inverse_transform(svm_model.predict(email_tfidf))[0]
        nb_pred = label_encoder.inverse_transform(nb_model.predict(email_tfidf))[0]

        st.subheader("🔎 Predictions")
        st.markdown(f"**🧠 Random Forest Prediction:** `{rf_pred}`")
        st.markdown(f"**💻 SVM Prediction:** `{svm_pred}`")
        st.markdown(f"**📘 Naive Bayes Prediction:** `{nb_pred}`")
