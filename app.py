import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the trained model and vectorizer
model_path = "model/fake_news_model .pkl"
vectorizer_path = "model/vectorizer .pkl"

with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)

with open(vectorizer_path, 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Streamlit app title and intro
st.title("📰 TruthLens: Fake News Detection")
st.write("Enter a news article below to check whether it is **Fake** or **Real**.")

# Text input
input_text = st.text_area("📝 News Article Content")

# Prediction function
def predict(text):
    transformed_text = vectorizer.transform([text])
    prediction = model.predict(transformed_text)
    return prediction[0]

# Predict button
if st.button("Detect"):
    if input_text.strip() == "":
        st.warning("Please enter a news article.")
    else:
        result = predict(input_text)
        if result == "FAKE":
            st.error("🛑 This news is predicted to be **FAKE**.")
        else:
            st.success("✅ This news is predicted to be **REAL**.")
