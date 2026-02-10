# Fake News Detection using Machine Learning

This project implements a Fake News Detection system using Machine Learning and Natural Language Processing (NLP). The model classifies news articles as Real or Fake based on their textual content.

---

## Project Overview

Fake news spreads rapidly on online platforms and can cause serious misinformation. This project uses TF-IDF text vectorization and a Multinomial Naive Bayes classifier to automatically detect fake news articles.

---

## Machine Learning Model

- Model: Multinomial Naive Bayes  
- Learning Type: Supervised Learning  
- Task: Binary Classification (Real / Fake)

---

## Dataset

- Source: Kaggle Fake News Dataset  
- File: fake_news_finaldataset_small.csv  

### Columns
- text – News article content  
- label –  
  - 1 = Real News  
  - 0 = Fake News  

---

## Technologies Used

- Python  
- Pandas  
- NLTK  
- Scikit-learn  

---

## Methodology

1. Load the dataset  
2. Handle missing values  
3. Remove stopwords using NLTK  
4. Convert text to numerical form using TF-IDF  
5. Split data into training and testing sets  
6. Train Multinomial Naive Bayes model  
7. Evaluate accuracy  
8. Predict Real or Fake news  

---

## How to Run

### Step 1: Clone the Repository
```bash
git clone https://github.com/your-username/fake-news-detection.git
cd fake-news-detection
