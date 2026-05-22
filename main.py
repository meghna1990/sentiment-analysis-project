print("Program started")
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Download stopwords (only needs to happen once)
nltk.download('stopwords')

# 1. Load dataset
# Ensure your JSON file is formatted as a proper list: [{}, {}]
df = pd.read_json("Sarcasm_Headlines_Dataset.json")
print("Dataset loaded. Shape:", df.shape)

# 2. Convert to sentiment labels
df['sentiment'] = df['is_sarcastic'].apply(lambda x: 'Negative' if x==1 else 'Positive')

# 3. Clean text
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

print("Cleaning text...")
df['cleaned'] = df['headline'].apply(clean_text)

# 4. TF-IDF Vectorization
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(df['cleaned']).toarray()
y = df['sentiment']
print("TF-IDF vectorization complete")

# 5. Train-test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Data split done")

# 6. Initialize and Train Model (ORDER MATTERS HERE)
model = MultinomialNB() # Initialize first
print("Training model...")
model.fit(X_train, y_train) # Then fit

# 7. Predict and Evaluate
print("Predicting...")
y_pred = model.predict(X_test)

print("--- Results ---")
print("Accuracy Score:", accuracy_score(y_test, y_pred))