from flask import Flask, request, render_template
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load models
with open("models/rf_model.pkl", "rb") as f:
    rf_model = pickle.load(f)
with open("models/svm_model.pkl", "rb") as f:
    svm_model = pickle.load(f)
with open("models/nb_model.pkl", "rb") as f:
    nb_model = pickle.load(f)

# Load vectorizer and label encoder
with open("models/tfidf_vectorizer.pkl", "rb") as f:
    tfidf_vectorizer = pickle.load(f)
with open("models/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        email_text = request.form["email"]
        email_tfidf = tfidf_vectorizer.transform([email_text])

        # Get predictions
        rf_pred = label_encoder.inverse_transform(rf_model.predict(email_tfidf))[0]
        svm_pred = label_encoder.inverse_transform(svm_model.predict(email_tfidf))[0]
        nb_pred = label_encoder.inverse_transform(nb_model.predict(email_tfidf))[0]

        return render_template("result.html", email=email_text, rf=rf_pred, svm=svm_pred, nb=nb_pred)
    
    return render_template("index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

