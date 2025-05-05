# 📰 Fake News Classifier

This project is a machine learning-based Fake News Classifier built as a capstone for an Applied Machine Learning course. It leverages NLP techniques and two classification models to detect whether a news article is **real** or **fake**, with deployment via a simple Gradio web interface.

## 📌 Problem Statement

The rise of misinformation and biased news in modern media presents a critical challenge. This project aims to identify fake news articles using machine learning, helping users evaluate the credibility of the content they read.

## 📊 Dataset

- **Source**: [Kaggle Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
- Contains over **45,000** labeled articles split into `Fake.csv` and `True.csv`.

## 🧪 Features & Workflow

### ✅ Preprocessing
- Tokenization
- Lowercasing
- Removal of punctuation and digits
- Stopword filtering
- Lemmatization

### ✅ Models Used
- **Naive Bayes (Fast Mode)**
- **Logistic Regression (Accurate Mode)**

### ✅ Performance
| Model               | Accuracy | Precision | Recall | F1 Score |
|--------------------|----------|-----------|--------|----------|
| Multinomial NB      | ~93%     | 0.93      | 0.93   | 0.93     |
| Logistic Regression | ~99%     | 0.99      | 0.99   | 0.99     |

### ✅ Deployment
- Interactive web UI built with **Gradio**
- Users can paste an article and select:
  - 🔹 Fast Mode (Naive Bayes)
  - 🔸 Accurate Mode (Logistic Regression)
- Output includes **classification** (Fake/True) and **confidence score**

## 🚀 Running Locally

```bash
git clone https://github.com/amechi-aduba/newsclassifier.git
cd fake-news-classifier

# Install dependencies
pip install -r requirements.txt

# Run the Gradio app
python app.py
