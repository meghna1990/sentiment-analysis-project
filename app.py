import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

nltk.download('stopwords')

@st.cache_data
def load_data():
    return pd.read_json("Sarcasm_Headlines_Dataset.json")

df = load_data()
df['sentiment'] = df['is_sarcastic'].apply(lambda x: 'Negative' if x==1 else 'Positive')

stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = [w for w in text.split() if w not in stop_words]
    return " ".join(words)

df['cleaned'] = df['headline'].apply(clean_text)

@st.cache_resource
def train():
    tfidf = TfidfVectorizer(max_features=5000)
    X = tfidf.fit_transform(df['cleaned'])
    y = df['sentiment']
    model = MultinomialNB()
    model.fit(X, y)
    return tfidf, model

tfidf, model = train()

st.title("📰 News Sentiment Analyzer")

text = st.text_input("Enter a news headline:")

if st.button("Predict"):
    if text.strip() == "":
        st.warning("Please enter a headline")
    else:
        cleaned = clean_text(text)
        vector = tfidf.transform([cleaned])
        result = model.predict(vector)[0]

        if result == "Positive":
            st.success(f"Sentiment: {result}")
        else:
            st.error(f"Sentiment: {result}")