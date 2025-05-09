## "Fake News Classifier"
# "Detect true vs. fake articles with Naive Bayes, TF-IDF LogReg, and MiniLM + LogReg"
table_of_contents:
  - Motivation
  - Dataset
  - Project Structure
  - Installation
  - How It Works
  - Results
  - Running the App
  - Limitations & Future Work
  - Credits

motivation: |
  Misinformation travels faster than fact-checking.  
  This project offers an instant credibility check so readers can decide whether to trust, ignore, or further verify online news.

dataset:
  name: "Kaggle Fake & Real News"
  link: "https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset"
  size: "≈ 44 k articles"
  classes:
    true: 1
    fake: 0
  columns_used: ["text", "label"]
  years: "2015 – 2017"

project_structure:
  notebook: "Jupyter analysis + training"
  miniLM: "saved Sentence-Transformer encoder"
  miniLM_lr_pkl: "Logistic-Regression on embeddings"
  nb_pkl: "Naive-Bayes (TF-IDF) model"
  lr_pkl: "LogReg (TF-IDF) model"
  tfidf_pkl: "TF-IDF vectorizer"
  app_py: "Gradio interface"
  requirements_txt: "pip/conda dependencies"

installation:
  steps:
    - "git clone https://github.com/<user>/fake-news-classifier.git && cd fake-news-classifier"
    - |
      conda create -n fake-news python=3.10 -y  
      conda activate fake-news
    - "pip install -r requirements.txt  # downloads MiniLM (~90 MB)"

how_it_works:
  models:
    - mode: "Fast"
      pipeline: "TF-IDF → Naive Bayes"
      pros: "Instant predictions; tiny model"
      cons: "Lower F1 (≈ 93 %)"
    - mode: "Accurate"
      pipeline: "TF-IDF → Logistic Regression"
      pros: "98–99 % in-dataset accuracy"
      cons: "Vocab-dependent; mislabels unfamiliar outlets"
    - mode: "Contextual"
      pipeline: "MiniLM embeddings → Logistic Regression"
      pros: "≈ 98 % F1; robust; CPU-friendly"
      cons: "90 MB encoder; embeds add ~100 ms"

results:
  metrics:
    - model: "Naive Bayes"
      accuracy: 0.93
      f1: 0.93
    - model: "TF-IDF LogReg"
      accuracy: 0.99
      f1: 0.99
    - model: "MiniLM + LogReg"
      accuracy: 0.98
      f1: 0.98
  note: "Metrics computed on a 20 % hold-out split."

running_app:
  command: "python app.py  # opens http://127.0.0.1:7860"
  ui: "Paste article → select Fast / Accurate / Contextual → receive label + confidence"

limitations_future_work:
  - "Dataset skews to U.S. politics; expand to newer/global sources"
  - "Add explainability (SHAP, LIME) to show which sentences drive decisions"
  - "Extend to multilingual news"

credits:
  dataset: "Clément Bisaillon (Kaggle)"
  miniLM: "Sentence-Transformers: Reimers & Gurevych, 2020"
  note: "Built as a capstone for Applied Machine Learning @ Clark University"
