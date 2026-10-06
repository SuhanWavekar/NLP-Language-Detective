# Language Detective (NLP Mini Project)

Identifies which of 10 languages a piece of text is written in, and shows the
"clues" (character patterns) it used to reach its verdict. Two detection
techniques are built and compared head-to-head.

## Languages covered
English, French, Spanish, German, Italian, Portuguese, Dutch, Swedish, Turkish, Indonesian

## Files
- `generate_dataset.py` — builds `language_dataset.csv` (1,000 sentences, 100 per language)
- `language_dataset.csv` — the generated dataset
- `language_detective.py` — main script: trains both models, compares them, solves new cases
- `confusion_matrix.png` — generated after running the script
- `Language_Detective_Report.docx` — write-up you can submit

## Setup
```bash
pip install scikit-learn pandas matplotlib
```

## Run
```bash
python generate_dataset.py     # only needed once, to (re)build the dataset
python language_detective.py
```

## What it does
1. **Approach A — Character n-grams + Naive Bayes**: extracts 1-3 letter sequences
   (e.g. "sch", "tion", "ggi") as fingerprints for each language
2. **Approach B — Word TF-IDF + Logistic Regression**: classifies based on whole-word importance
3. Compares both on accuracy and F1 score, and plots a confusion matrix for the winner
4. Runs the `investigate()` function on new sentences, printing the predicted language,
   confidence score, top supporting "clues", and runner-up guesses

## Try your own text
```python
investigate("Your sentence here")
```

## Extending this project
- Add more languages, including non-Latin scripts (Russian, Arabic, Hindi, Chinese)
- Train on longer, real-world text (news, tweets) instead of short sentences
- Compare against production libraries like `langdetect` or `fasttext`'s language identifier
- Wrap `investigate()` in a small Flask/Streamlit app for a live "detective" demo
