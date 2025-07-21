import streamlit as st
import pickle
import os
from bs4 import BeautifulSoup
import requests
import re

# Load the model and vectorizer
model_path = os.path.join("model", "fake_news_model.pkl")
vectorizer_path = os.path.join("model", "vectorizer.pkl")

with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)

with open(vectorizer_path, 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Function to clean the text
def clean_text(text):
    text = re.sub(r'\W', ' ', text)  # remove all non-word characters
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)  # remove single characters
    text = re.sub(r'\s+', ' ', text)  # remove extra spaces
    return text.lower()

# Function to extract text from a URL
def get_article_text(url):
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all('p')
        content = ' '.join([p.get_text() for p in paragraphs])
        return content
    except:
        return None

# Streamlit App
st.set_page_config(page_title="TruthLens - Fake News Detector")
st.title("🕵️ TruthLens - Fake News Detector")
st.markdown("Enter a **news article URL** to find out whether it's Real or Fake!")

# Input field
url_input = st.text_input("🔗 Paste the news URL here:")

# Predict button
if st.button("🔍 Analyze"):
    if url_input:
        with st.spinner("Extracting and analyzing the article..."):
            article_text = get_article_text(url_input)
            if article_text:
                cleaned_text = clean_text(article_text)
                vectorized_text = vectorizer.transform([cleaned_text])
                prediction = model.predict(vectorized_text)[0]
                if prediction.upper() == "FAKE":
                    st.error("❌ The news appears to be **Fake**.")
                else:
                    st.success("✅ The news appears to be **Real**.")
            else:
                st.warning("⚠️ Could not extract article content. Try a different URL.")
    else:
        st.warning("⚠️ Please enter a URL.")
