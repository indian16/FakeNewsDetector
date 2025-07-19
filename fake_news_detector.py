import streamlit as st
import joblib
import os
from url_extractor import extract_article_from_url
import re
import matplotlib.pyplot as plt
import pandas as pd


def load_model_and_vectorizer():
    model = joblib.load("models/fast_model.pkl")
    vectorizer = joblib.load("models/vectorizer.pkl")
    return model, vectorizer


def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z ]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.lower().strip()


def predict(text, model, vectorizer):
    text = clean_text(text)
    vec = vectorizer.transform([text])
    prediction = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0].max()
    return prediction, prob


def visualize_result(label, confidence):
    result = 'FAKE' if label == 1 else 'REAL'
    other = 'REAL' if result == 'FAKE' else 'FAKE'
    df = pd.DataFrame({
        'Category': [result, other],
        'Confidence': [confidence, 1 - confidence]
    })

    fig, ax = plt.subplots()
    bars = ax.bar(df['Category'], df['Confidence'], color=['red' if result == 'FAKE' else 'green', 'gray'])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Confidence Score")
    ax.set_title("Prediction Confidence")

    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02, f'{height:.2f}', ha='center', va='bottom')

    st.pyplot(fig)


st.set_page_config(page_title="Fake News Detector", layout="centered")
st.title("📰 Fake News Detector")
st.markdown("Enter a news article URL OR paste article text to detect if it's fake or real.")

model, vectorizer = load_model_and_vectorizer()

input_type = st.radio("Choose input type:", ("URL", "Raw Text"))

user_input = st.text_area("Paste news URL or article text:")

if st.button("Analyze"):
    if not user_input.strip():
        st.error("Please provide input.")
    else:
        if input_type == "URL":
            with st.spinner("Extracting article from URL..."):
                article_text = extract_article_from_url(user_input)
                if not article_text:
                    st.error("Failed to extract article from URL.")
                else:
                    label, confidence = predict(article_text, model, vectorizer)
                    st.success(f"Prediction: {'FAKE' if label == 1 else 'REAL'} ({confidence:.2f} confidence)")
                    visualize_result(label, confidence)
        else:
            label, confidence = predict(user_input, model, vectorizer)
            st.success(f"Prediction: {'FAKE' if label == 1 else 'REAL'} ({confidence:.2f} confidence)")
            visualize_result(label, confidence)





