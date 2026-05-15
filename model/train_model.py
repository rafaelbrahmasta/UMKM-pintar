
"""
UMKM Pintar — Advanced NLP Chatbot Trainer
Algoritma:
- TF-IDF Vectorizer
- LinearSVC
- Confidence Threshold
- Fallback Intent
- Stemming Bahasa Indonesia
- GridSearchCV Tuning
"""

import json
import pickle
import re
import os
import random
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    GridSearchCV
)
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix
)

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# =====================================================
# PATH
# =====================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "Data", "chatbot_dataset.json")
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "chatbot_model.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)

# =====================================================
# LOAD DATASET
# =====================================================

with open(DATASET_PATH, "r", encoding="utf-8") as f:
    dataset = json.load(f)

print("=" * 60)
print("     UMKM PINTAR — ADVANCED NLP TRAINER")
print("=" * 60)
print(f"Total Intent : {len(dataset['intents'])}")

# =====================================================
# STEMMER
# =====================================================

factory = StemmerFactory()
stemmer = factory.create_stemmer()

# =====================================================
# STOPWORDS
# =====================================================

STOPWORDS_ID = {
    "yang", "dan", "di", "ke", "dari", "ini", "itu", "dengan",
    "untuk", "ada", "pada", "adalah", "juga", "akan", "atau",
    "jika", "maka", "ya", "ok", "oke", "deh", "dong", "sih",
    "nih", "lah", "kah", "nya", "pun", "agar", "supaya",
    "tetapi", "namun", "tapi", "karena", "sebab", "jadi",
    "secara", "hal", "lebih", "kepada", "dalam"
}

# =====================================================
# PREPROCESSING
# =====================================================

ABBREVIATIONS = [
    (r"\bumkm\b", "usaha mikro kecil menengah umkm"),
    (r"\bbep\b", "break even point bep titik impas"),
    (r"\bhpp\b", "harga pokok penjualan produksi hpp"),
    (r"\broi\b", "return on investment roi"),
    (r"\bkur\b", "kredit usaha rakyat kur"),
    (r"\bnib\b", "nomor induk berusaha nib"),
    (r"\bswot\b", "strengths weaknesses opportunities threats swot"),
    (r"\busp\b", "unique selling point usp"),
    (r"\bfifo\b", "first in first out fifo"),
]


def preprocess(text: str) -> str:
    text = text.lower().strip()

    # expand singkatan
    for pattern, replacement in ABBREVIATIONS:
        text = re.sub(pattern, replacement, text)

    # hapus simbol
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # tokenisasi
    tokens = text.split()

    # remove stopwords
    tokens = [t for t in tokens if t not in STOPWORDS_ID and len(t) > 1]

    # stemming
    text = " ".join(tokens)
    text = stemmer.stem(text)

    return text

# =====================================================
# SIAPKAN DATA
# =====================================================

X_raw = []
y_raw = []
responses_map = {}
tag_list = []

for intent in dataset["intents"]:
    tag = intent["tag"]

    responses_map[tag] = intent["responses"]
    tag_list.append(tag)

    for pattern in intent["patterns"]:
        X_raw.append(pattern)
        y_raw.append(tag)

print(f"Total Pattern Asli : {len(X_raw)}")

# =====================================================
# AUGMENTATION
# =====================================================

PREFIXES = [
    "tolong ",
    "bisa ",
    "mau tahu ",
    "jelaskan ",
    "gimana ",
    "bagaimana ",
    "saya ingin tahu ",
]


def augment(text, label):
    variants = [(text, label)]

    if len(text.split()) >= 2:
        for pre in PREFIXES:
            variants.append((pre + text, label))

        if not text.endswith("?"):
            variants.append((text + "?", label))

        if text.startswith("cara "):
            variants.append(("bagaimana " + text[5:], label))

        if "apa itu" in text:
            variants.append((text.replace("apa itu", "pengertian"), label))

    return variants


X_aug = []
y_aug = []

for text, label in zip(X_raw, y_raw):
    augmented = augment(text, label)

    for t, l in augmented:
        X_aug.append(t)
        y_aug.append(l)

# shuffle
combined = list(zip(X_aug, y_aug))
random.shuffle(combined)
X_aug, y_aug = zip(*combined)

# preprocess
X_proc = [preprocess(t) for t in X_aug]

print(f"Total Sampel Setelah Augmentasi : {len(X_proc)}")

# =====================================================
# TRAIN TEST SPLIT
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X_proc,
    y_aug,
    test_size=0.2,
    random_state=42,
    stratify=y_aug
)

print(f"Train : {len(X_train)}")
print(f"Test  : {len(X_test)}")

# =====================================================
# PIPELINE
# =====================================================

pipeline = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            ngram_range=(1, 3),
            max_features=10000,
            sublinear_tf=True,
            min_df=1,
            analyzer="word"
        )
    ),
    (
        "clf",
        LinearSVC(
            C=1.5,
            max_iter=5000,
            class_weight="balanced"
        )
    )
])

# =====================================================
# HYPERPARAMETER TUNING
# =====================================================

print("\nMelakukan hyperparameter tuning...")

param_grid = {
    "tfidf__ngram_range": [(1, 1), (1, 2), (1, 3)],
    "clf__C": [0.5, 1.0, 1.5, 2.0]
}

search = GridSearchCV(
    pipeline,
    param_grid,
    cv=3,
    scoring="accuracy",
    n_jobs=-1
)

search.fit(X_train, y_train)

best_model = search.best_estimator_

print("\nBest Parameter:")
print(search.best_params_)

# =====================================================
# TRAINING
# =====================================================

print("\nMelatih model terbaik...")
best_model.fit(X_train, y_train)

# =====================================================
# EVALUATION
# =====================================================

print("\nEvaluasi model...")

y_pred = best_model.predict(X_test)

acc = accuracy_score(y_test, y_pred)

cv_scores = cross_val_score(
    best_model,
    X_proc,
    y_aug,
    cv=5,
    scoring="accuracy"
)

print("\n" + "=" * 60)
print(f"Accuracy Test Set : {acc * 100:.2f}%")
print(
    f"Cross Validation  : {cv_scores.mean() * 100:.2f}% "
    f"+/- {cv_scores.std() * 100:.2f}%"
)
print("=" * 60)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, zero_division=0))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

# =====================================================
# CONFIDENCE FUNCTION
# =====================================================


def predict_with_confidence(model, text):
    processed = preprocess(text)

    scores = model.decision_function([processed])

    pred = model.predict([processed])[0]

    confidence = np.max(scores)

    return pred, float(confidence)

# =====================================================
# SAVE MODEL
# =====================================================

model_data = {
    "model": best_model,
    "responses_map": responses_map,
    "tag_list": tag_list,
    "accuracy": round(acc * 100, 2),
    "cv_mean": round(cv_scores.mean() * 100, 2),
    "n_intents": len(tag_list),
    "n_samples": len(X_proc),
    "best_params": search.best_params_,
    "preprocess_function": "custom_preprocess_v2",
    "confidence_threshold": 0.5,
}

with open(MODEL_PATH, "wb") as f:
    pickle.dump(model_data, f)

print(f"\nModel berhasil disimpan:")
print(MODEL_PATH)

# =====================================================
# MANUAL TESTING
# =====================================================

print("\nManual Testing")
print("-" * 60)

TEST_CASES = [
    ("cara hitung laba bersih", "laba_bersih"),
    ("apa itu break even point", "bep"),
    ("cara meningkatkan omzet", "strategi_pemasaran"),
    ("berapa pajak usaha saya", "pajak"),
    ("cara promosi instagram", "strategi_pemasaran"),
    ("saya mau buka usaha", "rekomendasi_usaha"),
    ("cara daftar tokopedia", "digitalisasi"),
    ("halo selamat pagi", "salam"),
    ("terima kasih", "perpisahan"),
    ("cuaca hari ini bagaimana", "fallback"),
]

correct = 0

for text, expected in TEST_CASES:
    pred, conf = predict_with_confidence(best_model, text)

    # fallback handling
    if conf < 0.5:
        pred = "fallback"

    status = "OK" if pred == expected else "SALAH"

    if pred == expected:
        correct += 1

    print(
        f"[{status}] "
        f"Input: '{text}' "
        f"-> Prediksi: {pred} "
        f"(confidence: {conf:.2f})"
    )

print("\n" + "=" * 60)
print(
    f"Manual Testing Result: {correct}/{len(TEST_CASES)} benar "
    f"({correct / len(TEST_CASES) * 100:.0f}%)"
)
print("=" * 60)

print("\nTraining selesai!")
print("Jalankan aplikasi dengan: python app.py")


