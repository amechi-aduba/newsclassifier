# 📰 Fake News Classifier — Naive Bayes | TF-IDF LogReg | MiniLM Embeddings

Access at: (https://amechi21-fake-news-detector.hf.space/)


Capstone project for **Applied Machine Learning** — detects whether a news article is **True** or **Fake** via three NLP pipelines, exposed through a Gradio web interface.

---

## 📌 Problem Statement
Misinformation spreads faster than fact-checking.  
This project offers an **instant credibility check** so readers can decide whether to trust, ignore, or further verify online articles.

---

## 📊 Dataset
| Item | Details |
|------|---------|
| **Source** | [Kaggle – Fake & Real News](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) |
| **Size**  | ≈ 44 k labelled articles (`True.csv`, `Fake.csv`) |
| **Years** | 2015 – 2017 |
| **Columns used** | `text` and `label` |

---

## 🧪 Features & Workflow

### ✅ Pre-processing
| Step | Purpose |
|------|---------|
| Lower-case, remove punctuation/digits | Normalise text |
| Tokenisation + NLTK stop-word filter | Drop glue words |
| WordNet lemmatisation | Merge word variants |
| **TF-IDF (5 k terms)** | For classical models |
| **MiniLM sentence embedding (384-dim)** | For contextual model |

### ✅ Models Implemented
| Mode | Pipeline | Relative Speed | Strengths |
|------|----------|----------------|-----------|
| **Fast** | TF-IDF → Multinomial Naive Bayes | ⚡ ms | Tiny, instant |
| **Accurate** | TF-IDF → Logistic Regression | 🔸 tens ms | 99 % in-dataset accuracy |
| **Contextual** | MiniLM embedding → Logistic Regression | 🔹 ~150-200 ms | Best generalisation to unseen outlets |

### ✅ Performance (20 % hold-out)
| Model | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|----|
| Naive Bayes | 0.93 | 0.93 | 0.93 | 0.93 |
| TF-IDF LogReg | **0.99** | 0.99 | 0.99 | 0.99 |
| MiniLM + LogReg | **0.98** | 0.98 | 0.98 | 0.98 |

*MiniLM sacrifices ~1 % in-dataset accuracy for much stronger robustness to brand-new sources.*

### ✅ Deployment
- **Gradio** web app (`app.py`)
- Paste article → choose mode  
  - 🔹 Fast (Naive Bayes)  
  - 🔸 Accurate (TF-IDF LogReg)  
  - 🔶 Contextual (MiniLM LogReg)  
- Returns **label** (True / Fake) **+ confidence %**

## future work:
  - "Data freshness — add 2020-2025 multilingual news to stay current."
  - "Explainability — integrate SHAP/LIME to highlight sentences driving predictions."
  - "Active learning — let users flag mistakes and retrain on new feedback."
  - "Cloud deployment — package as a Hugging Face Space or Docker image for easy sharing."

## credits:
  dataset: "Clément Bisaillon — Fake & Real News (Kaggle)"
  miniLM: "Sentence-Transformers — Reimers & Gurevych (2020)"
  gradio: "Abubakar Abid et al."
  author: "Created by **Amechi Aduba** as a capstone for *Applied Machine Learning* @ Clark University"
---

## 🚀 Running Locally

```bash
git clone https://github.com/<your-username>/fake-news-classifier.git
cd fake-news-classifier

# 1️⃣  (Recommended) create clean env
conda create -n fake-news python=3.10 -y
conda activate fake-news

# 2️⃣  install dependencies
pip install -r requirements.txt      # downloads MiniLM (~90 MB)

# 3️⃣  launch Gradio UI
python app.py                        # opens http://127.0.0.1:7860


