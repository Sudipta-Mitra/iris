import streamlit as st
import joblib
import re
import nltk
import pathlib
import os
from nltk.corpus import stopwords

# Absolute path to models
PROJECT_ROOT = pathlib.Path(__file__).parent.parent.resolve()
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')

# Load model and vectorizer
model = joblib.load(os.path.join(MODELS_DIR, 'sentiment_model.pkl'))
vectorizer = joblib.load(os.path.join(MODELS_DIR, 'tfidf_vectorizer.pkl'))

# Download stopwords if not already
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub('[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

# --- Streamlit UI ---

st.set_page_config(page_title="🎬 Movie Review Sentiment Analyzer", page_icon="🎯", layout="wide")

# Background color
st.markdown("""
    <style>
    .main {
        background-color: #f7f7f7;
    }
    </style>
    """, unsafe_allow_html=True)

# Colorful header
st.markdown("""
    <div style="background: linear-gradient(to right, #ff416c, #ff4b2b); padding: 20px; border-radius: 15px;">
        <h1 style="color: white; text-align: center;">🎬 Sentiment Analyzer</h1>
        <p style="color: white; text-align: center; font-size: 18px;">Using ML | NLP | DL | Deployed with Streamlit 🚀</p>
    </div>
""", unsafe_allow_html=True)

st.write("")
st.markdown("<h4 style='color:#444;'>📝 Enter your movie review:</h4>", unsafe_allow_html=True)

# Text Input
user_input = st.text_area("", height=200, placeholder="Ex: The movie was outstanding, truly a masterpiece!", key="input_area")

# Predict button
predict_button = st.button("Predict Sentiment")

if predict_button:
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a review to analyze.")
    else:
        processed_input = preprocess_text(user_input)
        vectorized_input = vectorizer.transform([processed_input]).toarray()
        prediction = model.predict(vectorized_input)

        if prediction[0] == 1:
            st.success(" This review is **Positive**")
            st.balloons()
        else:
            st.error(" This review is **Negative**")

# Footer
st.markdown("""
    <hr>
    <div style="text-align:center; color: grey;">
        <p>Made by Sudipta | Ardent Project 2025</p>
    </div>

""", unsafe_allow_html=True)
