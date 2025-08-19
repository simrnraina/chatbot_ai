# train_chatbot.py (No TensorFlow)
import json
import pickle
import random
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

# Make sure you have nltk data
nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

# Load intents
with open("intents.json", "r") as f:
    intents = json.load(f)

# Collect training data
sentences = []
labels = []
classes = []

for intent in intents["intents"]:
    tag = intent["tag"]
    if tag not in classes:
        classes.append(tag)
    for pattern in intent["patterns"]:
        sentences.append(pattern)
        labels.append(tag)

# Convert labels to indices
class_to_idx = {c: i for i, c in enumerate(classes)}
y = [class_to_idx[label] for label in labels]

# Vectorize sentences
vectorizer = CountVectorizer(tokenizer=nltk.word_tokenize)
X = vectorizer.fit_transform(sentences)

# Train Logistic Regression model
model = LogisticRegression(max_iter=200)
model.fit(X, y)

# Save model, vectorizer, and classes
pickle.dump(model, open("chatbot_model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))
pickle.dump(classes, open("classes.pkl", "wb"))

print("Training complete. Model saved as chatbot_model.pkl, vectorizer.pkl, classes.pkl")
