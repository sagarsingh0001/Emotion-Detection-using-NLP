# 🧠 Emotion Detection using NLP

A Streamlit web app that reads a piece of text and predicts the emotion behind it, powered by a machine learning model trained on labelled text data.

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Model-F7931E?logo=scikitlearn&logoColor=white)
![NLP](https://img.shields.io/badge/NLP-Text%20Classification-8A2BE2)

<!-- Replace with your own screenshot or GIF -->
![App Screenshot](assets/UI.png)

**🔗 Live demo:** _https://emotion-detection-using-nlp-ml.streamlit.app/_

---

## 📌 Overview

Understanding the emotion in text is useful for customer feedback analysis, social media monitoring, conversational AI and user experience research. Reading large volumes of text by hand doesn't scale, so this project builds a machine learning pipeline that predicts the emotion expressed in a piece of text.

The model predicts one of six classes:

| Label | Emotion  | Emoji |
|-------|----------|-------|
| 0     | Sadness  | 😢 |
| 1     | Anger    | 😠 |
| 2     | Love     | ❤️ |
| 3     | Surprise | 😲 |
| 4     | Fear     | 😨 |
| 5     | Joy      | 😄 |

The final model is a **Logistic Regression classifier on Bag-of-Words (unigram + bigram) features**, reaching **90.53% accuracy** and a **88.22% macro F1-score** on a held-out test set. The Streamlit app shows the emotion name (not the numeric label), a confidence score and a probability chart.

## ✨ Features

- Modern UI with an emotion-coloured result card and emoji
- Predicted emotion shown by name, with confidence and a probability chart for all six classes
- Same text preprocessing as training, applied automatically to every input
- Option to view the cleaned text that is actually sent to the model
- One-click example sentences and a recent-predictions history

## 📊 Dataset

- **Size:** 16,000 labelled English text samples (`train.txt`, semicolon-separated: `text;emotion`)
- **Classes:** 6 emotions
- **Text length:** 2 to 66 words per sample (mean about 19)
- **Data quality:** no missing values; 1 duplicate row and 31 duplicate texts (about 0.19%); a few identical texts carry conflicting labels (for example joy vs love)

| Emotion  | Samples | Share |
|----------|--------:|------:|
| Joy      | 5,362 | 33.5% |
| Sadness  | 4,666 | 29.2% |
| Anger    | 2,159 | 13.5% |
| Fear     | 1,937 | 12.1% |
| Love     | 1,304 | 8.2% |
| Surprise | 572   | 3.6% |

The classes are imbalanced, so the data was split with stratification (80/20, `random_state=42`: 12,800 train / 3,200 test) and class weights were balanced during training.

_Dataset source: [link](https://www.kaggle.com/datasets/praveengovi/emotions-dataset-for-nlp)_

## 🧹 Preprocessing Pipeline

Applied in this order, both in the notebook and in `app.py`:

1. Lowercasing
2. Punctuation removal
3. Number removal
4. URL removal
5. HTML tag removal
6. Non-ASCII character and emoji removal
7. Tokenization (NLTK) and English stopword removal

## 🧠 Modelling

Two feature representations (Bag of Words with 1-2 grams, and TF-IDF) were tested with four algorithms, using a stratified train/test split:

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 |
|-------|---------:|----------------:|-------------:|---------:|------------:|
| Multinomial Naive Bayes, BoW | 82.62% | 86.80% | 67.86% | 72.53% | 81.31% |
| Multinomial Naive Bayes, TF-IDF | 66.22% | 86.72% | 40.73% | 40.34% | 58.80% |
| **Logistic Regression, BoW** | **90.53%** | 87.17% | 89.51% | **88.22%** | 90.59% |
| Logistic Regression, TF-IDF | 85.09% | 86.80% | 74.27% | 78.72% | 84.51% |
| SVM (linear), BoW | 88.94% | 85.76% | 86.09% | 85.88% | 88.93% |
| SVM (linear), TF-IDF | 88.19% | 83.91% | 87.27% | 85.37% | 88.31% |
| Linear SVC, BoW | 90.59% | 87.50% | 88.73% | 88.07% | 90.62% |
| Linear SVC, TF-IDF | 89.16% | 85.46% | 89.65% | 87.18% | 89.25% |

**Key finding:** Bag of Words beat TF-IDF for every model on this dataset.

**Final model:** Logistic Regression + Bag of Words, built as a single scikit-learn `Pipeline` and saved with `joblib`:

```python
Pipeline([
    ("vectorizer", CountVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.95)),
    ("classifier", LogisticRegression(C=0.8, max_iter=2000,
                                      class_weight="balanced", random_state=42)),
])
```

It is essentially tied with Linear SVC on accuracy, has the higher macro F1, and gives class probabilities, which the app uses for confidence scores.

### Final model performance (test set, 3,200 samples)

| Emotion  | Precision | Recall | F1-score | Support |
|----------|----------:|-------:|---------:|--------:|
| Sadness  | 0.95 | 0.93 | 0.94 | 933 |
| Anger    | 0.88 | 0.88 | 0.88 | 432 |
| Love     | 0.79 | 0.88 | 0.83 | 261 |
| Surprise | 0.78 | 0.90 | 0.84 | 115 |
| Fear     | 0.90 | 0.86 | 0.88 | 387 |
| Joy      | 0.93 | 0.92 | 0.92 | 1072 |
| **Overall accuracy** | | | **0.91** | 3200 |

The most common mistake is confusing **joy and love** (49 joy samples predicted as love, 29 love samples predicted as joy), which is expected because the two emotions are semantically close.

## 📂 Project Structure

```
emotion-detection-streamlit/
├── app.py                          # Streamlit app
├── emotion_classifier.joblib       # Trained pipeline (vectorizer + classifier)
├── requirements.txt                # Dependencies
├── README.md
├── .gitignore
├── models/
│   └── model_metadata.json         # Model settings and evaluation info (generated by the notebook)
├── notebooks/
│   └── emotion-detection-using-nlp.ipynb   # Full workflow: EDA, preprocessing, modelling, evaluation
├── data/
│   └── README.md                   # Dataset source (place train.txt here to re-run the notebook)
└── assets/
    └── screenshot.png              # App screenshot
```

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/emotion-detection-streamlit.git
cd emotion-detection-streamlit
```

### 2. (Optional) Create a virtual environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`. The required NLTK data (`punkt`, `punkt_tab`, `stopwords`) is downloaded automatically on first run.

### Re-running the notebook

Place `train.txt` in the `data/` folder, install `matplotlib` and `seaborn` in addition to the requirements, and run `notebooks/emotion-detection-using-nlp.ipynb`. The notebook reads the data from `../data/`.

## 🛠️ Tech Stack

- **Language:** Python
- **NLP:** NLTK (tokenization, stopwords)
- **ML:** scikit-learn (CountVectorizer, Logistic Regression, Pipeline)
- **Web app:** Streamlit, Altair
- **Data and analysis:** pandas, numpy, matplotlib, seaborn
- **Model persistence:** joblib

> **Note:** load the model with the same scikit-learn version it was trained with, otherwise `joblib` may raise compatibility warnings or errors. Pin that version in `requirements.txt`.

## ⚠️ Limitations

- **Negations are removed.** The standard NLTK stopword list contains words like "not", "no" and "never", so a sentence such as "I am not happy" can lose its negation. Keeping negation words is an easy improvement.
- **Class imbalance.** Surprise (3.6%) and love (8.2%) have far fewer examples than joy and sadness, which makes them harder to predict reliably.
- **Label noise.** Some identical texts carry different labels in the dataset, which puts a ceiling on achievable accuracy.
- **Bag of Words ignores word order** beyond bigrams, so sarcasm and long-range context are not captured.
- **English only.**

## 🔮 Future Improvements

- Keep negation words during stopword removal and compare results
- Try transformer models (BERT / DistilBERT) for higher accuracy
- Tune hyperparameters with the stratified 5-fold cross-validation already set up in the notebook
- Add batch prediction from CSV files
- Deploy with Docker

## 🤝 Contributing

Contributions, issues and feature requests are welcome. Feel free to open an issue or submit a pull request.

## 👤 Author

**Sagar**

- GitHub: [@sagarsingh0001](https://github.com/sagarsingh0001)
- LinkedIn: _https://www.linkedin.com/in/sagar-singh0308/_

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

⭐ If you found this project useful, consider giving it a star!
