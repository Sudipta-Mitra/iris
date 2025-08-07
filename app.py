import streamlit as st
import pickle

# Load model
with open("sentiment_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("Sentiment Analysis App")
text = st.text_area("Enter text to analyze sentiment:")

if st.button("Predict"):
    if text:
        result = model.predict([text])[0]
        st.success(f"Sentiment: {result}")
    else:
        st.warning("Please enter some text.")
