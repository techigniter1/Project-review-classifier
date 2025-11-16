import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json
import numpy as np

# -----------------------------
# Load tokenizer
# -----------------------------
with open("tokenizer.json") as f:
    data = json.load(f)
    tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(data)

# -----------------------------
# Load trained LSTM model
# -----------------------------
model = tf.keras.models.load_model("review_classifier.h5")

max_length = 100

# -----------------------------
# Streamlit Page Configuration
# -----------------------------
st.set_page_config(page_title="E-Commerce Review Classifier", page_icon="🛒")

st.markdown("""
    <h1 style="text-align:center; color:#2F80ED;">🛒 E-Commerce Review Sentiment Classifier</h1>
    <p style="text-align:center;">Analyze customer reviews instantly using Deep Learning</p>
""", unsafe_allow_html=True)

st.write("---")

# ======================================================
#            SINGLE REVIEW PREDICTION SECTION
# ======================================================
st.subheader("✨ Single Review Prediction")

review = st.text_area("Enter a review:", placeholder="Type something like: 'The product quality is amazing!'")

if st.button("Predict Sentiment"):
    if review.strip() == "":
        st.warning("⚠ Please enter a review first.")
    else:
        seq = tokenizer.texts_to_sequences([review])
        padded = pad_sequences(seq, maxlen=max_length, padding='post')

        pred = model.predict(padded)[0][0]

        # Output formatting
        if pred > 0.5:
            sentiment = "Positive 😊"
            color = "#A7F3D0"
        else:
            sentiment = "Negative 😡"
            color = "#FCA5A5"

        st.markdown(
            f"""
            <div style="padding:15px; border-radius:10px; background-color:{color}">
                <h3>Prediction: {sentiment}</h3>
                <p><b>Confidence Score:</
