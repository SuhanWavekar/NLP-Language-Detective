# -*- coding: utf-8 -*-
"""
LANGUAGE DETECTIVE — Interactive Streamlit UI
A minimalist interface for the NLP language detection mini project.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
    classification_report
)

# ── Page config ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="Language Detective",
    page_icon="🔍",
    layout="centered",
)

# ── Custom minimalist CSS ────────────────────────────────────────────
st.markdown("""
<style>
    /* --- Base typography & palette --- */
    :root {
        --bg: #fafafa;
        --card: #ffffff;
        --accent: #4f46e5;
        --accent-light: #e0e7ff;
        --text: #1e1e2e;
        --muted: #6b7280;
        --border: #e5e7eb;
        --success: #10b981;
        --warning: #f59e0b;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Smooth body */
    .stApp {
        background-color: var(--bg);
    }

    /* Card styling */
    .card {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }

    /* Verdict banner */
    .verdict-banner {
        background: linear-gradient(135deg, #4f46e5, #7c3aed);
        color: white;
        border-radius: 12px;
        padding: 1.75rem 2rem;
        text-align: center;
        margin: 1rem 0;
    }
    .verdict-banner h1 {
        margin: 0;
        font-size: 2rem;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    .verdict-banner p {
        margin: 0.25rem 0 0 0;
        font-size: 1rem;
        opacity: 0.85;
    }

    /* Clue chips */
    .clue-chip {
        display: inline-block;
        background: var(--accent-light);
        color: var(--accent);
        font-weight: 600;
        padding: 0.3rem 0.75rem;
        border-radius: 999px;
        margin: 0.2rem;
        font-size: 0.85rem;
        font-family: 'Courier New', monospace;
    }

    /* Suspect row */
    .suspect-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.5rem 0;
        border-bottom: 1px solid var(--border);
    }
    .suspect-row:last-child { border-bottom: none; }
    .suspect-name { font-weight: 500; color: var(--text); }
    .suspect-pct { font-weight: 600; color: var(--muted); font-size: 0.9rem; }

    /* Metric cards */
    .metric-card {
        text-align: center;
        padding: 1rem;
    }
    .metric-card .value {
        font-size: 2rem;
        font-weight: 700;
        color: var(--accent);
    }
    .metric-card .label {
        font-size: 0.8rem;
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Flag emoji for languages */
    .lang-flag { font-size: 1.5rem; margin-right: 0.5rem; }

    /* Confidence bar */
    .conf-bar-bg {
        background: var(--border);
        border-radius: 999px;
        height: 8px;
        width: 100%;
        margin-top: 0.5rem;
    }
    .conf-bar-fill {
        height: 8px;
        border-radius: 999px;
        transition: width 0.4s ease;
    }
</style>
""", unsafe_allow_html=True)

# ── Language flags ───────────────────────────────────────────────────
LANG_FLAGS = {
    "English": "🇬🇧", "French": "🇫🇷", "Spanish": "🇪🇸",
    "German": "🇩🇪", "Italian": "🇮🇹", "Portuguese": "🇵🇹",
    "Dutch": "🇳🇱", "Swedish": "🇸🇪", "Turkish": "🇹🇷",
    "Indonesian": "🇮🇩",
}


# ── Model training (cached) ─────────────────────────────────────────
@st.cache_resource(show_spinner="Training the detective…")
def train_models():
    """Train both models once and cache them."""
    csv_path = os.path.join(os.path.dirname(__file__), "language_dataset.csv")
    df = pd.read_csv(csv_path, encoding="utf-8")

    X_text = df["text"]
    y = df["language"]

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        X_text, y, test_size=0.20, random_state=42, stratify=y
    )

    # Approach A — Character n-grams + Naive Bayes
    char_vec = CountVectorizer(analyzer="char_wb", ngram_range=(1, 3), min_df=2)
    X_train_char = char_vec.fit_transform(X_train_text)
    X_test_char = char_vec.transform(X_test_text)
    char_model = MultinomialNB(alpha=0.1)
    char_model.fit(X_train_char, y_train)
    char_preds = char_model.predict(X_test_char)
    char_acc = accuracy_score(y_test, char_preds)
    char_f1 = f1_score(y_test, char_preds, average="macro")

    # Approach B — Word TF-IDF + Logistic Regression
    word_vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 1), min_df=1, sublinear_tf=True)
    X_train_word = word_vec.fit_transform(X_train_text)
    X_test_word = word_vec.transform(X_test_text)
    word_model = LogisticRegression(C=3.0, max_iter=1000)
    word_model.fit(X_train_word, y_train)
    word_preds = word_model.predict(X_test_word)
    word_acc = accuracy_score(y_test, word_preds)
    word_f1 = f1_score(y_test, word_preds, average="macro")

    # Confusion matrix for best model
    best_preds = char_preds if char_f1 >= word_f1 else word_preds
    labels = sorted(y.unique())
    cm = confusion_matrix(y_test, best_preds, labels=labels)

    # Classification report
    report = classification_report(y_test, char_preds, output_dict=True)

    return {
        "char_vec": char_vec, "char_model": char_model,
        "word_vec": word_vec, "word_model": word_model,
        "char_acc": char_acc, "char_f1": char_f1,
        "word_acc": word_acc, "word_f1": word_f1,
        "cm": cm, "labels": labels,
        "report": report,
        "n_samples": len(df),
        "n_languages": df["language"].nunique(),
    }


def investigate(text: str, models: dict):
    """Predict language and extract evidence clues."""
    char_vec = models["char_vec"]
    char_model = models["char_model"]

    vec = char_vec.transform([text])
    predicted = char_model.predict(vec)[0]
    proba = char_model.predict_proba(vec)[0]
    confidence = max(proba)

    class_index = list(char_model.classes_).index(predicted)
    feature_names = char_vec.get_feature_names_out()
    log_probs = char_model.feature_log_prob_
    predicted_log_probs = log_probs[class_index]
    mean_other = (log_probs.sum(axis=0) - predicted_log_probs) / (log_probs.shape[0] - 1)
    distinctiveness = predicted_log_probs - mean_other

    present_ngrams = vec.nonzero()[1]
    ngram_scores = [(feature_names[i], distinctiveness[i]) for i in present_ngrams]
    ngram_scores.sort(key=lambda x: (len(x[0].strip()) >= 2, x[1]), reverse=True)
    top_clues = [ng for ng, score in ngram_scores[:8] if ng.strip()]

    ranked = sorted(zip(char_model.classes_, proba), key=lambda x: x[1], reverse=True)

    return {
        "predicted": predicted,
        "confidence": confidence,
        "clues": top_clues,
        "ranked": ranked,
    }


# ── Load models ──────────────────────────────────────────────────────
models = train_models()

# ── Header ───────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align: center; padding: 2rem 0 1rem 0;">
    <p style="font-size: 3rem; margin: 0;">🔍</p>
    <h1 style="margin: 0; font-size: 2rem; font-weight: 700; letter-spacing: -1px; color: #1e1e2e;">
        Language Detective
    </h1>
    <p style="color: #6b7280; margin-top: 0.25rem; font-size: 0.95rem;">
        Type any sentence — the detective will identify its language and show the clues.
    </p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar — Model Performance ─────────────────────────────────────
with st.sidebar:
    st.markdown("### 📊 Model Performance")

    st.markdown(f"""
    <div class="card">
        <div style="display: flex; gap: 1rem;">
            <div class="metric-card" style="flex:1;">
                <div class="value">{models['n_samples']}</div>
                <div class="label">Sentences</div>
            </div>
            <div class="metric-card" style="flex:1;">
                <div class="value">{models['n_languages']}</div>
                <div class="label">Languages</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### Approach A: Char N-grams + NB")
    col1, col2 = st.columns(2)
    col1.metric("Accuracy", f"{models['char_acc']:.1%}")
    col2.metric("F1 Score", f"{models['char_f1']:.1%}")

    st.markdown("#### Approach B: Word TF-IDF + LR")
    col1, col2 = st.columns(2)
    col1.metric("Accuracy", f"{models['word_acc']:.1%}")
    col2.metric("F1 Score", f"{models['word_f1']:.1%}")

    winner = "Char N-grams" if models["char_f1"] >= models["word_f1"] else "Word TF-IDF"
    st.success(f"🏆 Winner: **{winner}**")

    st.markdown("---")
    st.markdown("#### Confusion Matrix")
    # Display confusion matrix as a heatmap
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use("Agg")

    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(models["cm"], cmap="Purples", aspect="auto")
    ax.set_xticks(range(len(models["labels"])))
    ax.set_yticks(range(len(models["labels"])))
    ax.set_xticklabels([l[:3] for l in models["labels"]], fontsize=7, rotation=45)
    ax.set_yticklabels([l[:3] for l in models["labels"]], fontsize=7)
    # Add text annotations
    for i in range(len(models["labels"])):
        for j in range(len(models["labels"])):
            val = models["cm"][i, j]
            if val > 0:
                color = "white" if val > models["cm"].max() / 2 else "#4f46e5"
                ax.text(j, i, str(val), ha="center", va="center", fontsize=7, color=color)
    ax.set_xlabel("Predicted", fontsize=8)
    ax.set_ylabel("Actual", fontsize=8)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("---")
    st.caption("Built with scikit-learn & Streamlit")

# ── Main input ───────────────────────────────────────────────────────
text_input = st.text_area(
    "Enter text to investigate",
    placeholder="e.g. Bonjour tout le monde, je m'appelle Marie.",
    height=100,
    label_visibility="collapsed",
)

col_btn, col_samples = st.columns([1, 2])
with col_btn:
    investigate_btn = st.button("🔍  Investigate", use_container_width=True, type="primary")
with col_samples:
    sample = st.selectbox(
        "Or try a sample",
        [""] + [
            "I really love traveling to new countries.",
            "Bonjour tout le monde, je m'appelle Marie.",
            "Estoy muy feliz de verte otra vez.",
            "Der Himmel ist heute wunderschön blau.",
            "Buongiorno, oggi voglio andare al mare.",
            "Olá, como vai você hoje? Tudo bem?",
            "Ik hou van reizen naar nieuwe landen.",
            "Det är så fantastiskt vackert väder idag.",
            "Bugün nasılsınız, iyi misiniz?",
            "Saya sangat senang bisa belajar bahasa baru.",
        ],
        label_visibility="collapsed",
    )

# Determine which text to use
active_text = text_input.strip() if investigate_btn and text_input.strip() else sample

# ── Results ──────────────────────────────────────────────────────────
if active_text:
    result = investigate(active_text, models)
    lang = result["predicted"]
    conf = result["confidence"]
    flag = LANG_FLAGS.get(lang, "🌐")

    # Confidence colour
    if conf >= 0.8:
        conf_color = "#10b981"
    elif conf >= 0.5:
        conf_color = "#f59e0b"
    else:
        conf_color = "#ef4444"

    # Verdict banner
    st.markdown(f"""
    <div class="verdict-banner">
        <p style="font-size: 2.5rem; margin: 0;">{flag}</p>
        <h1>{lang}</h1>
        <p>Confidence: {conf*100:.1f}%</p>
        <div class="conf-bar-bg" style="max-width: 300px; margin: 0.75rem auto 0 auto;">
            <div class="conf-bar-fill" style="width: {conf*100:.0f}%; background: {conf_color};"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Two-column layout: Clues + Suspects
    col_clues, col_suspects = st.columns(2)

    with col_clues:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**🧩 Top Clues** (character patterns)")
        chips_html = " ".join(
            f'<span class="clue-chip">{c}</span>' for c in result["clues"]
        )
        st.markdown(chips_html, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_suspects:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**🕵️ Other Suspects**")
        for lang_name, prob in result["ranked"][1:5]:
            pct = prob * 100
            lflag = LANG_FLAGS.get(lang_name, "🌐")
            st.markdown(f"""
            <div class="suspect-row">
                <span class="suspect-name">{lflag} {lang_name}</span>
                <span class="suspect-pct">{pct:.1f}%</span>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Expandable: full probability breakdown
    with st.expander("📋 Full probability breakdown"):
        prob_df = pd.DataFrame(
            result["ranked"], columns=["Language", "Probability"]
        )
        prob_df["Probability"] = prob_df["Probability"].apply(lambda x: f"{x*100:.2f}%")
        prob_df.index = range(1, len(prob_df) + 1)
        st.dataframe(prob_df, use_container_width=True, hide_index=False)

# ── Footer ───────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align: center; padding: 2rem 0 1rem 0; color: #9ca3af; font-size: 0.8rem;">
    Language Detective · NLP Mini Project · 10 Languages · 2 ML Approaches
</div>
""", unsafe_allow_html=True)
