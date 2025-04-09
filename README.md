# 📧 AI Email Categorizer with Streamlit

This is a Streamlit web application that uses **Machine Learning** models to automatically classify emails into categories such as **Promotion**, **Social**, **Spam**, **Finance**, and more.

🚀 **Live App**: [Click here to try the app](https://email-sorter.streamlit.app/)

---

## 🔍 Features

- 🔤 **Text Classification** using ML algorithms
- ✅ Models Used:
  - Random Forest Classifier
  - Support Vector Machine (SVM)
  - Naive Bayes Classifier
- 📊 Display predictions from all three models side-by-side
- 📬 Supports user input and **built-in sample email testing**
- 📈 **Displays model accuracy** for comparison
- 🧠 Preprocessing with **TF-IDF Vectorization**
- 🏷️ **Labels encoded** using `LabelEncoder`

---

## 📊 Model Accuracies

| Model                  | Accuracy Score |
|------------------------|----------------|
| Random Forest Classifier | 97.2%          |
| Support Vector Machine   | 95.6%          |
| Naive Bayes Classifier   | 92.4%          |

*(These scores are based on a labeled dataset of categorized emails.)*

---

## 💬 Sample Categories

Test your own email or try out one of the predefined categories:

- 🛍️ Promotion: "*Get 50% off on your next purchase!*"
- 👥 Social: "*You have a new friend request.*"
- 💰 Finance: "*Your account balance is low.*"
- 🗑️ Spam: "*You’ve won a million dollars! Click here now.*"
- 🧾 Updates: "*Your package has been shipped.*"

---

## 🛠 How it Works

1. The email text is vectorized using a **TF-IDF vectorizer**
2. The vectorized input is passed to all three models
3. Predictions are decoded using a label encoder
4. The output shows which category each model thinks the email belongs to
5. Accuracy for each model is displayed on the page

---



