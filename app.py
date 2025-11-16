import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json
import numpy as np


# Load tokenizer properly as JSON string
with open("tokenizer.json", "r") as f:
    tokenizer_json = f.read()

tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(tokenizer_json)


# Load model
model = tf.keras.models.load_model("review_classifier.h5")

max_length = 100

st.set_page_config(page_title="Review Classifier", page_icon="🛒")

st.markdown(
    """
    <h1 style='text-align:center; color:#2F80ED;'>🛒 E-Commerce Review Sentiment Classifier</h1>
    <p style='text-align:center;'>Analyze customer reviews instantly using Deep Learning.</p>
    """,
    unsafe_allow_html=True
)

st.write("---")

st.subheader("✨ Single Review Prediction")

review = st.text_area("Enter a review:", placeholder="Type something like: 'The product quality is amazing!'")

if st.button("Predict Sentiment"):
    if review.strip() == "":
        st.warning("⚠ Please enter a review first.")
    else:
        seq = tokenizer.texts_to_sequences([review])
        padded = pad_sequences(seq, maxlen=max_length, padding='post')
        pred = model.predict(padded)[0][0]

        if pred > 0.5:
            sentiment = "Positive 😊"
            bg_color = "#A7F3D0"
        else:
            sentiment = "Negative 😡"
            bg_color = "#FCA5A5"

        st.markdown(
            "<div style='padding:15px; border-radius:10px; background-color:{};'>"
            "<h3>Prediction: {}</h3>"
            "<p><b>Confidence Score:</b> {:.4f}</p>"
            "</div>".format(bg_color, sentiment, pred),
            unsafe_allow_html=True
        )

        st.write("### Confidence Meter")
        st.progress(float(pred))

st.write("---")

st.subheader("📂 Batch Prediction (Upload CSV)")

uploaded_file = st.file_uploader("Upload a CSV file with a 'review' column", type=['csv'])

if uploaded_file is not None:
    import pandas as pd

    df = pd.read_csv(uploaded_file)

    if "review" not in df.columns:
        st.error("❌ CSV must contain a column named 'review'.")
    else:
        st.success("✔ File uploaded successfully!")

        seqs = tokenizer.texts_to_sequences(df["review"].tolist())
        padded = pad_sequences(seqs, maxlen=max_length, padding="post")

        preds = model.predict(padded)

        df["sentiment"] = ["Positive" if p > 0.5 else "Negative" for p in preds]
        df["confidence"] = preds

        st.write("### Results Preview")
        st.dataframe(df.head())

        csv_out = df.to_csv(index=False).encode('utf-8')

        st.download_button(
            label="⬇ Download Full Results CSV",
            data=csv_out,
            file_name="predicted_reviews.csv",
            mime="text/csv"
        )
