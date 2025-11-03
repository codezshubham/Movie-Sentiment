# Step 1: Import Libraries and Load the Model
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

# Load the IMDB dataset word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Load the pre-trained model with ReLU activation
model = load_model('simple_rnn_imdb.h5')

# Step 2: Helper Functions
# Function to decode reviews
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

# Function to preprocess user input
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review


# import streamlit as st
# ## streamlit app
# # Streamlit app
# st.title('IMDB Movie Review Sentiment Analysis')
# st.write('Enter a movie review to classify it as positive or negative.')

# # User input
# user_input = st.text_area('Movie Review')

# if st.button('Classify'):

#     preprocessed_input=preprocess_text(user_input)

#     ## MAke prediction
#     prediction=model.predict(preprocessed_input)
#     sentiment='Positive' if prediction[0][0] > 0.5 else 'Negative'

#     # Display the result
#     st.write(f'Sentiment: {sentiment}')
#     st.write(f'Prediction Score: {prediction[0][0]}')
# else:
#     st.write('Please enter a movie review.')

import streamlit as st

# Streamlit UI Setup
st.set_page_config(page_title="IMDB Sentiment Classifier", page_icon="🎬", layout="centered")

# Custom CSS to style the app
st.markdown("""
    <style>
    body {background-color: #f8f9fa;}
    .main-title {
        font-size: 36px;
        font-weight: 700;
        color: #4B7BEC;
        text-align: center;
        margin-bottom: 10px;
    }
    .sub-text {
        text-align: center;
        font-size: 16px;
        color: #555;
        margin-bottom: 30px;
    }
    .prediction-box {
        padding: 18px;
        border-radius: 10px;
        text-align: center;
        font-size: 20px;
        font-weight: 600;
        margin-top: 15px;
    }
    .positive { background-color: #d4edda; color: #155724; border: 2px solid #c3e6cb;}
    .negative { background-color: #f8d7da; color: #721c24; border: 2px solid #f5c6cb;}
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-title">🎬 IMDB Movie Review Sentiment Analysis</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-text">Type your movie review below and let the model predict its sentiment!</div>', unsafe_allow_html=True)

# Input Box
user_input = st.text_area("✍️ Enter Movie Review Here", height=180)

# Button
if st.button("🔍 Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please write something in the review box.")
    else:
        preprocessed_input = preprocess_text(user_input)
        prediction = model.predict(preprocessed_input)[0][0]
        
        sentiment = "Positive" if prediction > 0.5 else "Negative"
        sentiment_class = "positive" if sentiment == "Positive" else "negative"

        st.markdown(
            f'<div class="prediction-box {sentiment_class}">'
            f"Sentiment: {sentiment} <br> Confidence Score: {prediction:.3f}"
            f'</div>',
            unsafe_allow_html=True
        )
else:
    st.info("Enter a review and click *Analyze Sentiment* to see result.")
