from __future__ import annotations

import math
import pickle
import re
import string
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import current_app
try:
    from fuzzywuzzy import fuzz
except Exception:
    try:
        # rapidfuzz provides faster implementations; use it if fuzzywuzzy missing
        from rapidfuzz import fuzz
    except Exception:
        from difflib import SequenceMatcher

        class _FallbackFuzz:
            @staticmethod
            def QRatio(a, b):
                return int(round(100 * SequenceMatcher(None, a, b).ratio()))

            @staticmethod
            def partial_ratio(a, b):
                return int(round(100 * SequenceMatcher(None, a, b).ratio()))

            @staticmethod
            def token_sort_ratio(a, b):
                a_sorted = " ".join(sorted(a.split()))
                b_sorted = " ".join(sorted(b.split()))
                return int(round(100 * SequenceMatcher(None, a_sorted, b_sorted).ratio()))

            @staticmethod
            def token_set_ratio(a, b):
                a_set = " ".join(sorted(set(a.split())))
                b_set = " ".join(sorted(set(b.split())))
                return int(round(100 * SequenceMatcher(None, a_set, b_set).ratio()))

        fuzz = _FallbackFuzz()
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer
try:
    from keras.models import load_model
    from keras.layers import (
        Bidirectional,
        Concatenate,
        Dense,
        Dropout,
        Embedding,
        GlobalMaxPool1D,
        InputLayer,
        Lambda,
        LSTM,
    )
except Exception:
    from tensorflow.keras.models import load_model
    from tensorflow.keras.layers import (
        Bidirectional,
        Concatenate,
        Dense,
        Dropout,
        Embedding,
        GlobalMaxPool1D,
        InputLayer,
        Lambda,
        LSTM,
    )
from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel

# The models and data files are located in the parent directory (root of the workspace)
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "train2.csv"
BOW_MODEL_PATH = BASE_DIR / "model.pkl"
BOW_VECTORIZER_PATH = BASE_DIR / "bow_vectorizer.pkl"
LSTM_MODEL_PATH = BASE_DIR / "bilstm.h5"
LSTM_TOKENIZER_PATH = BASE_DIR / "bilstm_tokenizer.pkl"
HYBRID_MODEL_PATH = BASE_DIR / "Hybrid_model.pkl"
TINYLLAMA_BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
TINYLLAMA_ADAPTER_PATH = BASE_DIR / \
    "tinyllama-duplicate-detector" / "checkpoint-6250"
MAXLEN = 25

MODEL_CHOICES = {
    "bow": "Keyword Matcher (Traditional & Fast)",
    "lstm": "Sequence Analyzer (Deep Neural Network)",
    "hybrid": "Semantic Similarity (Sentence Transformer)",
    "tinyllama": "Generative AI Reasoning (TinyLlama LLM)",
}

CONTRACTIONS = {
    "'re": " are",
    "'s": " is",
    "'d": " would",
    "'ll": " will",
    "'t": " not",
    "'ve": " have",
    "'m": " am",
}


def preprocess_text(text: str, stem: bool = False) -> str:
    if text is None or (isinstance(text, float) and math.isnan(text)):
        return ""

    text = str(text).lower().strip()

    for old, new in CONTRACTIONS.items():
        text = text.replace(old, new)

    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"[@#]\w+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = text.split()

    if stem:
        try:
            from nltk.stem import PorterStemmer

            stemmer = PorterStemmer()
            tokens = [stemmer.stem(token) for token in tokens]
        except Exception:
            pass

    return " ".join(tokens)


def _common_word_count(q1: str, q2: str) -> int:
    q1_words = set(q1.split())
    q2_words = set(q2.split())
    return len(q1_words.intersection(q2_words))


def _longest_common_substring_ratio(q1: str, q2: str) -> float:
    if not q1 or not q2:
        return 0.0

    q1_len = len(q1)
    q2_len = len(q2)
    matrix = [[0] * (q2_len + 1) for _ in range(q1_len + 1)]
    longest = 0

    for i in range(1, q1_len + 1):
        for j in range(1, q2_len + 1):
            if q1[i - 1] == q2[j - 1]:
                matrix[i][j] = matrix[i - 1][j - 1] + 1
                longest = max(longest, matrix[i][j])

    # Notebook uses min(len(q1), len(q2)) + 1 as the denominator for length features
    return longest / (min(q1_len, q2_len) + 1)


NLTK_STOP_WORDS = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", 
    "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", 
    "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", 
    "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", 
    "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", 
    "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", 
    "with", "about", "against", "between", "into", "through", "during", "before", "after", 
    "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", 
    "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", 
    "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", 
    "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", 
    "should", "now", "d", "ll", "m", "o", "re", "ve", "y", "ain", "aren", "couldn", "didn", 
    "doesn", "hadn", "hasn", "haven", "isn", "ma", "mightn", "mustn", "needn", "shan", 
    "shouldn", "wasn", "weren", "won", "wouldn"
}


def _bow_feature_vector(question1: str, question2: str) -> np.ndarray:
    q1 = preprocess_text(question1)
    q2 = preprocess_text(question2)

    q1_len = len(q1)
    q2_len = len(q2)

    q1_tokens = q1.split()
    q2_tokens = q2.split()

    q1_num_words = len(q1_tokens)
    q2_num_words = len(q2_tokens)

    # 1. Basic features
    q1_set = set(q1_tokens)
    q2_set = set(q2_tokens)
    common_words = len(q1_set.intersection(q2_set))
    total_words = len(q1_set) + len(q2_set)
    word_share = round(common_words / total_words, 2) if total_words else 0.0

    # 2. Token features (exact implementation from the notebook)
    SAFE_DIV = 0.0001
    
    if q1_num_words == 0 or q2_num_words == 0:
        cwc_min = cwc_max = csc_min = csc_max = ctc_min = ctc_max = last_word_eq = first_word_eq = 0.0
    else:
        q1_words = set([word for word in q1_tokens if word not in NLTK_STOP_WORDS])
        q2_words = set([word for word in q2_tokens if word not in NLTK_STOP_WORDS])

        q1_stops = set([word for word in q1_tokens if word in NLTK_STOP_WORDS])
        q2_stops = set([word for word in q2_tokens if word in NLTK_STOP_WORDS])

        common_word_count = len(q1_words.intersection(q2_words))
        common_stop_count = len(q1_stops.intersection(q2_stops))
        common_token_count = len(set(q1_tokens).intersection(set(q2_tokens)))

        cwc_min = common_word_count / (min(len(q1_words), len(q2_words)) + SAFE_DIV)
        cwc_max = common_word_count / (max(len(q1_words), len(q2_words)) + SAFE_DIV)
        csc_min = common_stop_count / (min(len(q1_stops), len(q2_stops)) + SAFE_DIV)
        csc_max = common_stop_count / (max(len(q1_stops), len(q2_stops)) + SAFE_DIV)
        ctc_min = common_token_count / (min(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
        ctc_max = common_token_count / (max(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)

        last_word_eq = float(q1_tokens[-1] == q2_tokens[-1])
        first_word_eq = float(q1_tokens[0] == q2_tokens[0])

    # 3. Length features (notebook uses word count differences, not character length differences)
    abs_len_diff = float(abs(q1_num_words - q2_num_words))
    mean_len = (q1_num_words + q2_num_words) / 2.0
    longest_substr_ratio = _longest_common_substring_ratio(q1, q2)

    # 4. Fuzzy features
    fuzz_ratio = float(fuzz.QRatio(q1, q2))
    fuzz_partial_ratio = float(fuzz.partial_ratio(q1, q2))
    token_sort_ratio = float(fuzz.token_sort_ratio(q1, q2))
    token_set_ratio = float(fuzz.token_set_ratio(q1, q2))

    return np.array(
        [
            float(q1_len),
            float(q2_len),
            float(q1_num_words),
            float(q2_num_words),
            float(common_words),
            float(total_words),
            float(word_share),
            float(cwc_min),
            float(cwc_max),
            float(csc_min),
            float(csc_max),
            float(ctc_min),
            float(ctc_max),
            float(last_word_eq),
            float(first_word_eq),
            float(abs_len_diff),
            float(mean_len),
            float(longest_substr_ratio),
            float(fuzz_ratio),
            float(fuzz_partial_ratio),
            float(token_sort_ratio),
            float(token_set_ratio),
        ],
        dtype=np.float32,
    )


@lru_cache(maxsize=1)
def _load_bow_model():
    with open(BOW_MODEL_PATH, "rb") as file:
        return pickle.load(file)


@lru_cache(maxsize=1)
def _load_bow_vectorizer():
    # Load the CountVectorizer saved during retraining to ensure the vocabulary is perfectly aligned
    if not BOW_VECTORIZER_PATH.exists():
        raise FileNotFoundError(
            f"BOW CountVectorizer not found at {BOW_VECTORIZER_PATH}. "
            "Please run the retraining script to save the vectorizer."
        )
    with open(BOW_VECTORIZER_PATH, "rb") as file:
        return pickle.load(file)


@lru_cache(maxsize=1)
def _load_hybrid_model():
    with open(HYBRID_MODEL_PATH, "rb") as file:
        return pickle.load(file)


@lru_cache(maxsize=1)
def _load_lstm_bundle():
    # Prefer the explicit HDF5 path, fall back to alternate names.
    lstm_path = LSTM_MODEL_PATH
    alternate_paths = [BASE_DIR / "BiLSTM.h5", BASE_DIR / "BiLSTM.keras"]
    if not lstm_path.exists():
        for alternate in alternate_paths:
            if alternate.exists():
                lstm_path = alternate
                break

    if not lstm_path.exists():
        raise FileNotFoundError(
            f"LSTM model file not found at {LSTM_MODEL_PATH} or {BASE_DIR / 'BiLSTM.h5'}."
        )

    # The retrained model was saved by Keras 3 (tensorflow.keras in TF 2.21).
    # Load it with keras directly and register AbsDiff, the custom layer that
    # replaced the unportable Lambda layer from the original notebook.
    import tensorflow as tf
    from keras.models import load_model as keras_load_model
    from keras.layers import Layer

    class AbsDiff(Layer):
        """Element-wise absolute difference – replaces the old Lambda layer."""
        def call(self, inputs):
            return tf.abs(inputs[0] - inputs[1])
        def get_config(self):
            return super().get_config()

    try:
        model = keras_load_model(
            str(lstm_path), compile=False, custom_objects={"AbsDiff": AbsDiff}
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load BiLSTM model from {lstm_path}. "
            f"Original error: {exc}"
        ) from exc

    # Load the tokenizer that was saved during training so that word→index
    # mappings are identical to what the model was trained on.
    # Rebuilding the tokenizer from scratch produces different mappings and
    # causes completely wrong predictions.
    if not LSTM_TOKENIZER_PATH.exists():
        raise FileNotFoundError(
            f"BiLSTM tokenizer not found at {LSTM_TOKENIZER_PATH}. "
            "Please run the training notebook and save the tokenizer with: "
            "pickle.dump(tok, open('bilstm_tokenizer.pkl', 'wb'))"
        )

    with open(LSTM_TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)

    return model, tokenizer


@lru_cache(maxsize=1)
def _load_sentence_transformer():
    return SentenceTransformer("all-MiniLM-L6-v2")


@lru_cache(maxsize=1)
def _load_tinyllama_bundle():
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(TINYLLAMA_ADAPTER_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device_map = "auto" if torch.cuda.is_available() else None
    base_model = AutoModelForCausalLM.from_pretrained(
        TINYLLAMA_BASE_MODEL,
        torch_dtype=dtype,
        device_map=device_map,
    )
    model = PeftModel.from_pretrained(base_model, TINYLLAMA_ADAPTER_PATH)
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
    return generator, tokenizer


def _predict_bow(question1: str, question2: str) -> dict:
    model = _load_bow_model()
    features = _bow_feature_vector(question1, question2).reshape(1, -1)

    # The original training pipeline concatenated advanced engineered
    # features (22) with CountVectorizer features for question1 and
    # question2 (each of size 3000 -> 6000 total) resulting in 6022
    # input features. The saved `model.pkl` contains only the
    # fitted RandomForest; the vectorizer wasn't serialized. Rebuild
    # the CountVectorizer on the training corpus so we can produce the
    # same-size BOW vectors at inference time.
    try:
        cv = _load_bow_vectorizer()
        # transform the pair into two vectors and concatenate horizontally
        bow_arr = cv.transform(
            [preprocess_text(question1), preprocess_text(question2)]).toarray()
        q1_bow = bow_arr[0].reshape(1, -1)
        q2_bow = bow_arr[1].reshape(1, -1)
        bow_concat = np.hstack([q1_bow, q2_bow])
        features = np.hstack([features, bow_concat])
    except Exception:
        # If rebuilding vectorizer fails, fall back to engineered-only features
        pass

    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(features)[0][1])
    else:
        prediction = model.predict(features)[0]
        probability = float(prediction)

    label = "Duplicated" if probability >= 0.4 else "Not duplicated"
    return {"label": label, "probability": probability}


def _predict_lstm(question1: str, question2: str) -> dict:
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    model, tokenizer = _load_lstm_bundle()
    q1 = preprocess_text(question1)
    q2 = preprocess_text(question2)

    q1_seq = tokenizer.texts_to_sequences([q1])
    q2_seq = tokenizer.texts_to_sequences([q2])
    q1_pad = pad_sequences(q1_seq, maxlen=MAXLEN, padding="post")
    q2_pad = pad_sequences(q2_seq, maxlen=MAXLEN, padding="post")

    probability = float(model.predict([q1_pad, q2_pad], verbose=0)[0][0])
    label = "Duplicated" if probability >= 0.4 else "Not duplicated"
    return {"label": label, "probability": probability}


def _predict_hybrid(question1: str, question2: str) -> dict:
    model = _load_hybrid_model()
    encoder = _load_sentence_transformer()

    emb1 = encoder.encode([preprocess_text(question1)], convert_to_numpy=True)
    emb2 = encoder.encode([preprocess_text(question2)], convert_to_numpy=True)
    diff = np.abs(emb1 - emb2)
    mul = emb1 * emb2
    cosine = np.sum(emb1 * emb2, axis=1, dtype=np.float32) / (
        np.linalg.norm(emb1, axis=1) * np.linalg.norm(emb2, axis=1)
    )
    features = np.hstack([diff, mul, cosine.reshape(-1, 1)])

    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(features)[0][1])
    else:
        prediction = model.predict(features)[0]
        probability = float(prediction)

    label = "Duplicated" if probability >= 0.4 else "Not duplicated"
    return {"label": label, "probability": probability}


def _predict_tinyllama(question1: str, question2: str) -> dict:
    generator, tokenizer = _load_tinyllama_bundle()
    prompt = f"""### Instruction:
Determine if the two questions are semantically duplicate.

### Input:
Question 1: {question1}
Question 2: {question2}

### Response:
"""

    out = generator(
        prompt,
        max_new_tokens=3,
        do_sample=False,
        return_full_text=False,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    answer = out[0]["generated_text"].strip()
    first_token = re.sub(r"[^a-z]", "", answer.split()
                         [0].lower()) if answer else ""

    if first_token in {"yes", "duplicate"}:
        label = "Duplicated"
    elif first_token in {"no", "not"}:
        label = "Not duplicated"
    else:
        label = "Unclear"

    return {"label": label}


def predict_duplicate(model_key: str, question1: str, question2: str) -> dict:
    if model_key == "bow":
        result = _predict_bow(question1, question2)
    elif model_key == "lstm":
        result = _predict_lstm(question1, question2)
    elif model_key == "hybrid":
        result = _predict_hybrid(question1, question2)
    elif model_key == "tinyllama":
        result = _predict_tinyllama(question1, question2)
    else:
        raise ValueError(f"Unknown model key: {model_key}")

    result["model"] = MODEL_CHOICES[model_key]
    return result
