# 📊 Twitter Sentiment Analysis using Machine Learning

> Automatically classifies tweets as **positive**, **negative**, or **neutral** using NLP and machine learning. Built to understand how public opinion can be extracted from social media data at scale.

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-154f3c?style=flat-square&logo=python&logoColor=white)

---

## 🧩 The Problem

Twitter generates millions of tweets every day. Manually reading them to understand public sentiment is impossible. Businesses, researchers, and governments need a way to **automatically gauge opinion** at scale — whether about a product, a political event, or a public health crisis.

This project automates that process using classical NLP and machine learning models, turning raw tweet text into actionable sentiment labels.

---

## 🏗️ How It Works

```
Raw Tweets
    │
    ▼
Text Preprocessing
(Cleaning, Tokenization, Stopword Removal, Lemmatization)
    │
    ▼
Feature Extraction
(TF-IDF Vectorization / Count Vectorizer)
    │
    ▼
ML Classification Model
(Logistic Regression / Naïve Bayes / SVM)
    │
    ▼
Sentiment Label: Positive / Negative / Neutral
    │
    ▼
Visualization (Word Clouds, Charts)
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core programming language |
| NLTK & SpaCy | Text preprocessing (tokenization, lemmatization, stopwords) |
| Scikit-learn | ML models (Logistic Regression, Naïve Bayes, SVM) + TF-IDF |
| Pandas & NumPy | Data loading, manipulation, and numerical ops |
| Matplotlib & Seaborn | Sentiment distribution charts |
| WordCloud | Visual representation of frequent words |
| Jupyter Notebook | Interactive development & visualization |

---

## 📁 Dataset

- **Source:** [Kaggle — Twitter Sentiment Dataset](https://www.kaggle.com/)
- **Contents:** Tweets pre-labeled as positive, negative, or neutral
- **Format:** CSV with tweet text and sentiment label columns

---

## 🚀 How to Run

### Step 1 — Clone the Repo
```bash
git clone https://github.com/sneha020902/Twitter-sentiment-analysis.git
cd Twitter-sentiment-analysis
```

### Step 2 — Install Dependencies
```bash
pip install pandas numpy scikit-learn nltk matplotlib seaborn wordcloud
```

### Step 3 — Download NLTK Data
```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
```

### Step 4 — Open the Notebook
```bash
jupyter notebook
```
Open `twitter-sentiment-analysis.ipynb` and run all cells.

---

## 📈 Model Performance

| Model | Accuracy |
|---|---|
| Logistic Regression | ~82% |
| Naïve Bayes | ~78% |
| SVM | ~84% |

> Results may vary depending on dataset split and preprocessing choices.

---

## 💡 What I Learned

- How **NLP preprocessing** works: cleaning text, removing noise, tokenizing, lemmatizing
- The difference between **TF-IDF** and **Count Vectorization** and when to use each
- How to train, evaluate, and compare multiple **ML classifiers** on the same task
- How to interpret **precision, recall, F1-score** beyond just accuracy
- How to generate **word clouds** and visualize class distributions
- The importance of **balanced datasets** in classification problems

---

## 🔮 Future Improvements

- [ ] Use **BERT / Transformers** for deep learning-based sentiment analysis
- [ ] Connect to the **Twitter API** for real-time tweet analysis
- [ ] Build a **Flask/FastAPI** web app to demo the model live
- [ ] Add **multi-language** support

---

## 👩‍💻 Author

**Sneha Agrawal** — Aspiring Cloud & DevOps Engineer
🔗 [LinkedIn](https://www.linkedin.com/in/-snehaagrawal/) · [GitHub](https://github.com/sneha020902) · [Portfolio](https://sneha020902.github.io)
