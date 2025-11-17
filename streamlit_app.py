from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import streamlit as st
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

import nltk

# Ensure required NLTK resources are available (safe to call multiple times)
for resource in ["punkt", "stopwords"]:
    try:
        nltk.data.find(f"tokenizers/{resource}")
    except LookupError:
        nltk.download(resource, quiet=True)


# ---------------------------------------------------------------------------
# Paths & Model Loading
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"


@st.cache_resource
def load_topic_assets() -> Tuple:
    lda_model = joblib.load(MODEL_DIR / "topic_model_lda.pkl")
    count_vectorizer = joblib.load(MODEL_DIR / "topic_vectorizer.pkl")
    with open(MODEL_DIR / "topic_labels.json", "r", encoding="utf-8") as f:
        topic_labels = json.load(f)
    topic_labels = {int(k): v for k, v in topic_labels.items()}
    return lda_model, count_vectorizer, topic_labels


@st.cache_resource
def load_sentiment_assets() -> Tuple:
    sentiment_model = joblib.load(MODEL_DIR / "sentiment_classifier.pkl")
    tfidf_vectorizer = joblib.load(MODEL_DIR / "topic_vectorizer_using_tfidf.pkl")
    return sentiment_model, tfidf_vectorizer


LDA_MODEL, COUNT_VECTORIZER, TOPIC_LABELS = load_topic_assets()
SENTIMENT_MODEL, TFIDF_VECTORIZER = load_sentiment_assets()


# ---------------------------------------------------------------------------
# Text Cleaning Utilities (mirrors notebook preprocessing)
# ---------------------------------------------------------------------------
STOP_WORDS = set(stopwords.words("english"))
STEMMER = PorterStemmer()


def clean_text_for_topics(text: str) -> str:
    text = re.sub(r"[^a-zA-Z\s]", "", str(text).lower())
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_text_for_sentiment(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    tokens = word_tokenize(text)
    filtered = [token for token in tokens if token not in STOP_WORDS]
    stemmed = [STEMMER.stem(token) for token in filtered]
    return " ".join(stemmed)


# ---------------------------------------------------------------------------
# Prediction Helpers
# ---------------------------------------------------------------------------
def infer_topic(text: str) -> Dict:
    cleaned = clean_text_for_topics(text)
    dtm = COUNT_VECTORIZER.transform([cleaned])
    topic_distribution = LDA_MODEL.transform(dtm)[0]
    topic_id = int(np.argmax(topic_distribution))
    topic_label = TOPIC_LABELS.get(topic_id, f"Topic {topic_id}")
    top_keywords = _topic_keywords(topic_id)

    return {
        "id": topic_id,
        "label": topic_label,
        "probability": float(topic_distribution[topic_id]),
        "distribution": topic_distribution,
        "keywords": top_keywords,
    }


def _topic_keywords(topic_id: int, n_words: int = 10) -> List[str]:
    vocab = COUNT_VECTORIZER.get_feature_names_out()
    weights = LDA_MODEL.components_[topic_id]
    top_indices = weights.argsort()[-n_words:][::-1]
    return [vocab[i] for i in top_indices]


def infer_sentiment(text: str) -> Dict:
    cleaned = clean_text_for_sentiment(text)
    features = TFIDF_VECTORIZER.transform([cleaned])
    prediction = SENTIMENT_MODEL.predict(features)[0]
    probabilities = SENTIMENT_MODEL.predict_proba(features)[0]
    class_probabilities = {
        SENTIMENT_MODEL.classes_[idx]: float(prob) for idx, prob in enumerate(probabilities)
    }

    return {
        "label": prediction,
        "probabilities": class_probabilities,
        "top_label_probability": class_probabilities[prediction],
    }


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Course Evaluation Topic & Sentiment Analyzer",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 Course Evaluation Topic & Sentiment Analyzer")
st.markdown(
    """
Use this app to classify new qualitative feedback from Business Intelligence students.
It predicts:

1. **Theme / Topic** using the LDA model trained on 44K hotel reviews (proxy for course reflections).
2. **Sentiment** using a TF-IDF + Logistic Regression classifier tuned on 44K labelled reviews.

_Tip: Paste one student's comment below and click **Analyze Feedback**._
"""
)

default_text = (
    "I appreciated the hands-on labs and supportive teaching team, but I wish "
    "there was more time for group discussions and real client projects."
)

user_text = st.text_area(
    "Student feedback",
    value=default_text,
    height=220,
    placeholder="Paste one student's end-term evaluation comment here...",
)

if st.button("Analyze Feedback", type="primary"):
    if not user_text.strip():
        st.warning("Please provide some text before running the analysis.")
    else:
        with st.spinner("Running topic modelling and sentiment analysis..."):
            topic_result = infer_topic(user_text)
            sentiment_result = infer_sentiment(user_text)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Predicted Topic")
            st.metric(
                label=topic_result["label"],
                value=f"Confidence: {topic_result['probability']:.2%}",
            )
            st.write("**Top keywords**")
            st.write(", ".join(topic_result["keywords"]))
            st.progress(topic_result["probability"])

        with col2:
            st.subheader("Sentiment Prediction")
            st.metric(
                label=sentiment_result["label"].title(),
                value=f"{sentiment_result['top_label_probability']:.2%}",
            )
            st.write("**Probability by class**")
            prob_df = (
                st.dataframe(
                    {
                        "Sentiment": list(sentiment_result["probabilities"].keys()),
                        "Probability": [
                            f"{p:.2%}" for p in sentiment_result["probabilities"].values()
                        ],
                    },
                    use_container_width=True,
                )
            )

        st.success(
            "Done! Use these insights to update the leading KPI (topic sentiment mix) and track "
            "whether course experience improvements are trending toward the 3.8/5 target."
        )

else:
    st.info("Paste a student's qualitative comment and click **Analyze Feedback** to begin.")


st.divider()
st.caption(
    "Models: 5-topic LDA (CountVectorizer) + Logistic Regression sentiment classifier (TF-IDF). "
    "Artifacts loaded from the local `model/` directory."
)

