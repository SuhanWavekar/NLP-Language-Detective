# -*- coding: utf-8 -*-
"""
LANGUAGE DETECTIVE
-------------------
An NLP mini project that identifies which language a piece of text is
written in, and shows the "clues" (evidence) it used to solve the case.

Two investigative techniques are built and compared:
  A) Character n-gram fingerprinting + Naive Bayes  (classic language-ID approach)
  B) Word-level TF-IDF + Logistic Regression         (general text-classification approach)

Dataset: 1,000 sentences, 100 each, across 10 languages
(English, French, Spanish, German, Italian, Portuguese, Dutch, Swedish, Turkish, Indonesian)
"""

import sys
# Ensure UTF-8 output on Windows terminals
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
    ConfusionMatrixDisplay, classification_report
)

# =====================================================================
# STEP 1: Load the case files (dataset)
# =====================================================================
df = pd.read_csv("language_dataset.csv", encoding="utf-8")
print(f"Loaded {len(df)} sentences across {df['language'].nunique()} languages")
print(df["language"].value_counts(), "\n")

X_text = df["text"]
y = df["language"]

X_train_text, X_test_text, y_train, y_test = train_test_split(
    X_text, y, test_size=0.20, random_state=42, stratify=y
)

# =====================================================================
# APPROACH A: Character n-gram fingerprinting + Naive Bayes
# ---------------------------------------------------------------------
# Every language has a distinctive "fingerprint" of common letter
# sequences (e.g. "sch" in German, "tion" in French/English, "ção" in
# Portuguese). We extract 1-3 character n-grams within word boundaries
# (char_wb) and let Naive Bayes with tuned Laplace smoothing (alpha=0.1)
# learn which fingerprints belong to which language.
# =====================================================================
char_vectorizer = CountVectorizer(analyzer="char_wb", ngram_range=(1, 3), min_df=2)
X_train_char = char_vectorizer.fit_transform(X_train_text)
X_test_char = char_vectorizer.transform(X_test_text)

char_model = MultinomialNB(alpha=0.1)
char_model.fit(X_train_char, y_train)
char_preds = char_model.predict(X_test_char)

char_accuracy = accuracy_score(y_test, char_preds)
char_f1 = f1_score(y_test, char_preds, average="macro")

# =====================================================================
# APPROACH B: Word-level TF-IDF + Logistic Regression
# ---------------------------------------------------------------------
# Instead of character patterns, this looks at whole words and their
# importance (TF-IDF with sublinear scaling). Works well when vocabulary
# barely overlaps between languages, with tuned regularization C=3.0.
# =====================================================================
word_vectorizer = TfidfVectorizer(analyzer="word", ngram_range=(1, 1), min_df=1, sublinear_tf=True)
X_train_word = word_vectorizer.fit_transform(X_train_text)
X_test_word = word_vectorizer.transform(X_test_text)

word_model = LogisticRegression(C=3.0, max_iter=1000)
word_model.fit(X_train_word, y_train)
word_preds = word_model.predict(X_test_word)

word_accuracy = accuracy_score(y_test, word_preds)
word_f1 = f1_score(y_test, word_preds, average="macro")

# =====================================================================
# STEP 2: Compare the two detectives
# =====================================================================
print("=" * 55)
print("CASE COMPARISON: Character N-grams vs Word TF-IDF")
print("=" * 55)
print(f"\n[A] Character n-gram + Naive Bayes")
print(f"    Accuracy: {char_accuracy:.3f}   Macro F1: {char_f1:.3f}")
print(f"\n[B] Word TF-IDF + Logistic Regression")
print(f"    Accuracy: {word_accuracy:.3f}   Macro F1: {word_f1:.3f}")

winner = "Character n-grams" if char_f1 >= word_f1 else "Word TF-IDF"
print(f"\nWinning approach on this test set: {winner}")

print("\n--- Detailed report: Character n-gram model ---")
print(classification_report(y_test, char_preds))

# =====================================================================
# STEP 3: Confusion matrix for the stronger model
# =====================================================================
best_preds = char_preds if char_f1 >= word_f1 else word_preds
labels = sorted(y.unique())
cm = confusion_matrix(y_test, best_preds, labels=labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
fig, ax = plt.subplots(figsize=(9, 8))
disp.plot(ax=ax, cmap="Purples", xticks_rotation=45)
plt.title(f"Confusion Matrix - {winner}")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
print("\nSaved confusion_matrix.png")

# =====================================================================
# STEP 4: The Detective's Report - solve a new case with evidence
# =====================================================================
def investigate(text: str, top_n_clues: int = 5):
    """
    Predicts the language of `text` using the character n-gram model,
    and reports the top n-gram "clues" that pointed to that language.

    A clue's strength is measured as how much MORE likely that n-gram
    is under the predicted language versus the average across all
    other languages (log-prob difference) -- this surfaces genuinely
    distinctive sequences (e.g. "sch", "tion") instead of just common
    single letters that appear in every language.
    """
    vec = char_vectorizer.transform([text])
    predicted = char_model.predict(vec)[0]
    proba = char_model.predict_proba(vec)[0]
    confidence = max(proba)

    class_index = list(char_model.classes_).index(predicted)
    feature_names = char_vectorizer.get_feature_names_out()
    log_probs = char_model.feature_log_prob_          # shape: [n_classes, n_features]
    predicted_log_probs = log_probs[class_index]
    mean_other_log_probs = (log_probs.sum(axis=0) - predicted_log_probs) / (log_probs.shape[0] - 1)
    distinctiveness = predicted_log_probs - mean_other_log_probs

    present_ngrams = vec.nonzero()[1]
    ngram_scores = [(feature_names[i], distinctiveness[i]) for i in present_ngrams]
    # Prefer multi-character n-grams (more interpretable clues than single letters)
    ngram_scores.sort(key=lambda x: (len(x[0].strip()) >= 2, x[1]), reverse=True)
    top_clues = [ng for ng, score in ngram_scores[:top_n_clues] if ng.strip()]

    print(f"\n{'='*55}")
    print(f'CASE FILE: "{text}"')
    print(f"{'='*55}")
    print(f"VERDICT: {predicted}  (confidence: {confidence*100:.1f}%)")
    print(f"TOP CLUES (n-grams): {top_clues}")

    # Show runner-up guesses too
    ranked = sorted(zip(char_model.classes_, proba), key=lambda x: x[1], reverse=True)
    print("Other suspects:")
    for lang, p in ranked[1:4]:
        print(f"  - {lang}: {p*100:.1f}%")

    return predicted, confidence, top_clues


# Run real test cases covering all 10 languages
sample_cases = [
    "I really love traveling to new countries and meeting people.",
    "Bonjour tout le monde, je m'appelle Marie.",
    "Estoy muy feliz de verte otra vez.",
    "Der Himmel ist heute wunderschön blau.",
    "Buongiorno, oggi voglio andare al mare.",
    "Olá, como vai você hoje? Tudo bem?",
    "Ik hou van reizen naar nieuwe landen.",
    "Det är så fantastiskt vackert väder idag.",
    "Bugün nasılsınız, iyi misiniz?",
    "Saya sangat senang bisa belajar bahasa baru.",
]

print("\n\n" + "#" * 55)
print("# SOLVING NEW CASES")
print("#" * 55)
for case in sample_cases:
    investigate(case)

print("\n\n" + "#" * 55)
print("# INTERACTIVE DETECTIVE")
print("#" * 55)
print("Type a sentence in any language to investigate it.")
print("Type 'exit' or 'quit' to end the investigation.")

while True:
    try:
        user_input = input("\nEnter text: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ["exit", "quit"]:
            print("Investigation ended. Case closed!")
            break
        investigate(user_input)
    except (KeyboardInterrupt, EOFError):
        print("\nInvestigation ended. Case closed!")
        break

