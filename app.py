import streamlit as st
import pickle

# Load models
with open("models/rf_model.pkl", "rb") as f:
    rf_model = pickle.load(f)
with open("models/svm_model.pkl", "rb") as f:
    svm_model = pickle.load(f)
with open("models/nb_model.pkl", "rb") as f:
    nb_model = pickle.load(f)

# Load TF-IDF and Label Encoder
with open("models/tfidf_vectorizer.pkl", "rb") as f:
    tfidf_vectorizer = pickle.load(f)
with open("models/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Streamlit App
st.set_page_config(page_title="📧 Email Categorizer", layout="centered")
st.title("📧 Smart Email Categorizer")
st.write("This app categorizes your email using 3 machine learning models.")

email_text = st.text_area("✍️ Enter email content below:")

if st.button("🔍 Predict Category"):
    if email_text.strip() == "":
        st.warning("Please enter some email text.")
    else:
        # Vectorize
        email_tfidf = tfidf_vectorizer.transform([email_text])

        # Predictions
        rf_pred = label_encoder.inverse_transform(rf_model.predict(email_tfidf))[0]
        svm_pred = label_encoder.inverse_transform(svm_model.predict(email_tfidf))[0]
        nb_pred = label_encoder.inverse_transform(nb_model.predict(email_tfidf))[0]

        # Results
        st.subheader("🔎 Predictions:")
        st.write(f"**Random Forest:** {rf_pred}")
        st.write(f"**SVM:** {svm_pred}")
        st.write(f"**Naive Bayes:** {nb_pred}")
