"""
SMS Spam Classification
------------------------
A simple text classification model that detects whether an SMS message
is "spam" or "ham" (not spam).

Author: Maryam Yaqoob (FA23-BAI-025)
"""

import re
import string
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# ---------------------------------------------------------------------
# 1. Load Dataset
# ---------------------------------------------------------------------
# Dataset: SMS Spam Collection (5,572 labeled SMS messages: spam/ham)
# Source: https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv
df = pd.read_csv("sms.tsv", sep="\t", header=None, names=["label", "text"])
print(f"Dataset shape: {df.shape}")
print(df["label"].value_counts())

# ---------------------------------------------------------------------
# 2. Preprocessing
# ---------------------------------------------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)          # remove URLs
    text = re.sub(r"\d+", "", text)                      # remove numbers
    text = text.translate(str.maketrans("", "", string.punctuation))  # remove punctuation
    text = re.sub(r"\s+", " ", text).strip()              # remove extra whitespace
    return text

df["clean_text"] = df["text"].apply(clean_text)
df["label_num"] = df["label"].map({"ham": 0, "spam": 1})

# ---------------------------------------------------------------------
# 3. Train/Test Split
# ---------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df["clean_text"], df["label_num"],
    test_size=0.2, random_state=42, stratify=df["label_num"]
)

# ---------------------------------------------------------------------
# 4. Feature Extraction (TF-IDF)
# ---------------------------------------------------------------------
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# ---------------------------------------------------------------------
# 5. Train Model (Logistic Regression)
# ---------------------------------------------------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

# ---------------------------------------------------------------------
# 6. Evaluate Model
# ---------------------------------------------------------------------
y_pred = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n===== Model Performance =====")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-Score : {f1:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=["ham", "spam"]))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# ---------------------------------------------------------------------
# 7. Try the model on a few custom examples
# ---------------------------------------------------------------------
samples = [
    "Congratulations! You won a free iPhone, click here to claim now!!!",
    "Hey, are we still meeting for lunch tomorrow?",
]
samples_clean = [clean_text(s) for s in samples]
samples_tfidf = vectorizer.transform(samples_clean)
preds = model.predict(samples_tfidf)
for s, p in zip(samples, preds):
    print(f"\nMessage: {s}\nPrediction: {'SPAM' if p == 1 else 'HAM'}")
