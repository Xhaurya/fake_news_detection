# FakeNewsDetection.py

import pandas as pd
import nltk
from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# ---------------------------------------------------
# 1. NLTK stopwords
# ---------------------------------------------------
nltk.download('stopwords')
print(stopwords.words('english'))

# ---------------------------------------------------
# 2. Load fake.csv and true.csv (from Kaggle)
# ---------------------------------------------------
news_dataset = pd.read_csv("fake_news_final_small.csv")

print(news_dataset.head())
print(news_dataset["label"].value_counts())

# ---------------------------------------------------
# 3. Prepare features (X) and labels (Y)
# ---------------------------------------------------
# Use the "text" column (if your dataset uses a different name,
# change 'text' to that column, e.g., 'content' or 'article')
X = news_dataset["text"]
Y = news_dataset["label"]

# Handle missing values
X = X.fillna("")

# ---------------------------------------------------
# 4. TF-IDF Vectorization
# ---------------------------------------------------
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
X = vectorizer.fit_transform(X)

# ---------------------------------------------------
# 5. Train-Test Split
# ---------------------------------------------------
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y,
    test_size=0.2,
    random_state=42,
    stratify=Y
)

# ---------------------------------------------------
# 6. Train Naive Bayes model
# ---------------------------------------------------
model = MultinomialNB()
model.fit(X_train, Y_train)

# ---------------------------------------------------
# 7. Accuracy
# ---------------------------------------------------
Y_train_pred = model.predict(X_train)
Y_test_pred = model.predict(X_test)

train_acc = accuracy_score(Y_train, Y_train_pred)
test_acc = accuracy_score(Y_test, Y_test_pred)

print("Accuracy score of the training data :", train_acc)
print("Accuracy score of the test data     :", test_acc)

# ---------------------------------------------------
# 8. Prediction on one test example
# ---------------------------------------------------
X_new = X_test[3]
prediction = model.predict(X_new)

print("Raw prediction for X_test[3]:", prediction)

if prediction[0] == 1:
    print("The news is Real")
else:
    print("The news is Fake")

print("Actual label:", Y_test.iloc[3])

# ---------------------------------------------------
# 9. Custom input prediction
# ---------------------------------------------------
input_news = ["Trump is not running for president in 2024"]

input_news = [
    "Government announces new education policy for students",
    "Aliens have landed in New York City",
    "The stock market reached an all-time high today",
    "Drinking salt water cures all diseases"
]

input_vec = vectorizer.transform(input_news)
predictions = model.predict(input_vec)

for news, pred in zip(input_news, predictions):
    print("\nNews:", news)
    if pred == 1:
        print("Prediction: The news is Real")
    else:
        print("Prediction: The news is Fake")
