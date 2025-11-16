import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json
import numpy as np
import keras
from keras.layers import InputLayer

# ---- FIX 1: Custom InputLayer to remove batch_shape ----
class CustomInputLayer(InputLayer):
    def __init__(self, **kwargs):
        kwargs.pop("batch_shape", None)  # Remove unsupported argument
        super().__init__(**kwargs)

custom_objects = {"InputLayer": CustomInputLayer}

# ---- FIX 2: Load tokenizer from JSON correctly ----
with open("tokenizer.json", "r") as f:
    tokenizer_json = f.read()

tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(tokenizer_json)

# ---- FIX 3: Load model safely ----
model = tf.keras.models.load_model(
    "review_classifier.h5",
    custom_objects=custom_objects,
    compile=False
)

# ---- App UI ----
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
        st.warning("Please enter a review!")
    else:
        seq = tokenizer.texts_to_sequences([review])
        padded = pad_sequences(seq, maxlen=max_length, padding='post')
        prediction = model.predict(padded)[0][0]

        sentiment = "Positive 😊" if prediction > 0.5 else "Negative 😡"
        score = float(prediction)

        st.success(f"### Sentiment: **{sentiment}**")
        st.info(f"Prediction score: **{score:.4f}**")

st.write("---")
st.caption("Built with TensorFlow • Streamlit • Railway")
