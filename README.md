# 🔍 Language Detective — NLP Mini Project

An NLP-powered application that identifies which of **10 languages** a piece of text is written in. It acts like a detective — analyzing character patterns ("clues") to solve the case and reveal the language, along with a confidence score and the evidence it used.

---

## 📌 What This Project Does

You type (or paste) **any sentence**, and the Language Detective will:

1. **Predict the language** it's written in
2. **Show a confidence score** (how sure it is)
3. **Reveal the "clues"** — the character patterns (n-grams) that gave it away
4. **List runner-up suspects** — other languages it considered

Two different ML techniques are trained, compared head-to-head, and the better one is used for predictions.

---

## 🌍 Languages Supported

| # | Language   | Flag |
|---|------------|------|
| 1 | English    | 🇬🇧  |
| 2 | French     | 🇫🇷  |
| 3 | Spanish    | 🇪🇸  |
| 4 | German     | 🇩🇪  |
| 5 | Italian    | 🇮🇹  |
| 6 | Portuguese | 🇵🇹  |
| 7 | Dutch      | 🇳🇱  |
| 8 | Swedish    | 🇸🇪  |
| 9 | Turkish    | 🇹🇷  |
| 10| Indonesian | 🇮🇩  |

---

## 🧠 How It Works (Simple Explanation)

### The Core Idea

Every language has **unique letter patterns**. For example:
- German uses **"sch"**, **"ung"**, **"keit"** a lot
- French has **"tion"**, **"eux"**, **"ais"**
- Spanish favors **"ción"**, **"mente"**, **"ando"**

The detective learns these patterns from 1,000 example sentences and then uses them to identify new text.

### Two Approaches Are Compared

| | Approach A | Approach B |
|---|---|---|
| **Name** | Character N-gram + Naive Bayes | Word TF-IDF + Logistic Regression |
| **What it looks at** | Small letter sequences (1–3 characters) like "th", "sch", "ção" | Whole words and how important they are |
| **Classifier** | Naive Bayes (probabilistic) | Logistic Regression (linear) |
| **Strength** | Great at catching language "fingerprints" even in short text | Works well when languages have very different vocabularies |

The project trains both, measures their accuracy and F1 score, and **automatically picks the winner**.

### Step-by-Step Flow

```
User types a sentence
        ↓
Text is converted into character n-gram features
        ↓
The trained model calculates probability for each language
        ↓
Highest probability = predicted language
        ↓
Most distinctive n-grams are extracted as "clues"
        ↓
Results displayed: Language, Confidence, Clues, Runner-ups
```

---

## 📁 Project Files

```
NLP language detective/
│
├── generate_dataset.py          # Builds the dataset (run once)
├── language_dataset.csv         # 1,000 labeled sentences (100 per language)
├── language_detective.py        # Core script: trains models, compares, investigates
├── app.py                       # Interactive Streamlit web UI
├── confusion_matrix.png         # Generated confusion matrix plot
├── Language_Detective_Report.docx  # Project report (submittable)
├── Language_Detective_README.md    # Original short README
└── README.md                    # ← This file
```

### What Each File Does

| File | Purpose |
|------|---------|
| `generate_dataset.py` | Contains 100 hand-written sentences per language. Running it creates `language_dataset.csv`. |
| `language_dataset.csv` | The training data — two columns: `text` and `language`. |
| `language_detective.py` | The main script. Trains both ML models, prints accuracy comparison, generates the confusion matrix, and runs sample investigations in the terminal. |
| `app.py` | A Streamlit web app that wraps everything in a clean, interactive UI. |

---

## ⚙️ Setup & Installation

### Prerequisites

- Python 3.8 or higher

### Install Dependencies

```bash
pip install scikit-learn pandas matplotlib streamlit
```

---

## 🚀 How to Run

### Option 1 — Interactive Web UI (Recommended)

```bash
python -m streamlit run app.py
```

This opens a browser at `http://localhost:8501` with a clean interface where you can:
- Type any sentence and click **Investigate**
- Pick from pre-loaded sample sentences
- See the verdict, clues, and suspects visually
- View model performance and confusion matrix in the sidebar

### Option 2 — Terminal / Command Line

```bash
python generate_dataset.py      # Only needed once, to build the CSV
python language_detective.py     # Trains models, compares, and starts interactive prompt
```

This prints results to the terminal and enters an interactive loop where you can type sentences.

---

## 🖥️ Web UI Features

| Feature | Description |
|---------|-------------|
| 🔍 **Text input** | Type or paste any sentence |
| 🎯 **Verdict banner** | Shows predicted language with country flag and confidence bar |
| 🧩 **Top clues** | Character patterns displayed as visual chips |
| 🕵️ **Other suspects** | Runner-up languages with their probabilities |
| 📋 **Full breakdown** | Expandable table with all 10 language probabilities |
| 📊 **Sidebar metrics** | Accuracy & F1 for both approaches, confusion matrix heatmap |
| 🏆 **Winner badge** | Highlights which approach performed better |

---

## 📊 Key Concepts Used

| Concept | Where It's Used |
|---------|-----------------|
| **Text Vectorization** | `CountVectorizer` (char n-grams), `TfidfVectorizer` (word-level) |
| **Classification** | `MultinomialNB`, `LogisticRegression` |
| **Model Evaluation** | Accuracy, Macro F1, Confusion Matrix, Classification Report |
| **Feature Analysis** | Log-probability differences to find distinctive n-grams |
| **Train/Test Split** | 80/20 stratified split |

---

## 🔬 Try It Yourself

### In the Web UI
Just type any sentence and click **Investigate**.

### In Python Code
```python
from language_detective import investigate

investigate("Bonjour tout le monde!")
investigate("Ich liebe dich")
investigate("Me gusta mucho este lugar")
```

---

## 📈 Extending This Project

Here are ideas to take it further:

- **Add more languages** — including non-Latin scripts (Russian, Arabic, Hindi, Chinese, Japanese)
- **Use real-world data** — news articles, tweets, Wikipedia excerpts
- **Compare with production tools** — `langdetect`, `fasttext`, `lingua`
- **Add a "teach" mode** — let users correct wrong predictions and retrain
- **Deploy online** — host on Streamlit Cloud, Heroku, or Railway

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core language |
| scikit-learn | ML models and evaluation |
| pandas | Data handling |
| matplotlib | Confusion matrix visualization |
| Streamlit | Interactive web interface |

---

## 👤 Author

CSMIT Project — NLP Language Detective

---

## 📄 License

This project is for educational purposes as part of a college mini project.
