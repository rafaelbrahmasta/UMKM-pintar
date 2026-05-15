import json
import pickle
import re
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# =====================================================
# PATH
# =====================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_PATH = os.path.join(
    BASE_DIR,
    "Data",
    "chatbot_dataset.JSON"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "model"
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "chatbot_model.pkl"
)

os.makedirs(MODEL_DIR, exist_ok=True)

# =====================================================
# LOAD DATASET
# =====================================================

with open(DATASET_PATH, "r", encoding="utf-8") as f:

    dataset = json.load(f)

print("✅ Dataset berhasil dimuat")
print("📦 Total Intent:", len(dataset["intents"]))

# =====================================================
# STOPWORDS
# =====================================================

STOPWORDS_ID = {
    "yang", "dan", "di", "ke", "dari", "ini", "itu",
    "dengan", "untuk", "ada", "pada", "bisa",
    "saya", "kamu", "anda", "adalah", "juga",
    "sudah", "akan", "atau", "jika", "maka",
    "ya", "ok", "oke", "deh", "dong", "sih",
    "nih", "lah", "kah", "nya", "pun",
    "agar", "supaya", "tetapi", "namun",
    "tapi", "karena", "sebab", "jadi",
    "mau", "ingin", "minta", "mohon",
    "buat", "bikin", "dapat", "bagi",
    "oleh", "secara", "hal", "lebih"
}

# =====================================================
# PREPROCESS
# =====================================================

def preprocess(text):

    text = text.lower().strip()

    abbreviations = [
        ("umkm", "usaha mikro kecil menengah umkm"),
        ("bep", "break even point bep"),
        ("hpp", "harga pokok penjualan hpp"),
        ("roi", "return on investment roi"),
        ("kur", "kredit usaha rakyat kur"),
        ("nib", "nomor induk berusaha nib"),
        ("swot", "strength weakness opportunity threat swot")
    ]

    for pattern, replacement in abbreviations:

        text = re.sub(
            r"\b" + pattern + r"\b",
            replacement,
            text
        )

    text = re.sub(r"[^a-z0-9\s]", " ", text)

    tokens = text.split()

    tokens = [
        t for t in tokens
        if t not in STOPWORDS_ID and len(t) > 1
    ]

    text = " ".join(tokens)

    return text

# =====================================================
# PREPARE DATA
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

print("✅ Total Patterns:", len(X_raw))

# =====================================================
# DATA AUGMENTATION
# =====================================================

X_aug = []
y_aug = []

for text, label in zip(X_raw, y_raw):

    X_aug.append(text)
    y_aug.append(label)

    if len(text.split()) >= 2:

        prefixes = [
            "tolong ",
            "bisa ",
            "mau tahu ",
            "jelaskan ",
            "apa itu "
        ]

        for prefix in prefixes:

            X_aug.append(prefix + text)
            y_aug.append(label)

# =====================================================
# PREPROCESS DATA
# =====================================================

X_processed = [
    preprocess(text)
    for text in X_aug
]

# =====================================================
# SPLIT DATA
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X_processed,
    y_aug,
    test_size=0.2,
    random_state=42,
    stratify=y_aug
)

# =====================================================
# MACHINE LEARNING PIPELINE
# =====================================================

pipeline = Pipeline([

    (
        "tfidf",
        TfidfVectorizer(
            ngram_range=(1, 3),
            max_features=8000,
            sublinear_tf=True
        )
    ),

    (
        "clf",
        LinearSVC(
            C=1.5,
            max_iter=3000,
            class_weight="balanced"
        )
    )

])

# =====================================================
# TRAIN MODEL
# =====================================================

print("🚀 Training model...")

pipeline.fit(X_train, y_train)

# =====================================================
# EVALUATION
# =====================================================

y_pred = pipeline.predict(X_test)

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n📊 HASIL EVALUASI")
print("=" * 50)

print(
    "✅ Accuracy:",
    round(accuracy * 100, 2),
    "%"
)

print("\n")
print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)

# =====================================================
# SAVE MODEL
# =====================================================

artifacts = {

    "model": pipeline,

    "responses_map": responses_map,

    "tag_list": tag_list,

    "accuracy": round(
        accuracy * 100,
        2
    ),

    "confidence_threshold": 0.2
}

with open(MODEL_PATH, "wb") as f:

    pickle.dump(
        artifacts,
        f
    )

print("\n✅ Model berhasil disimpan!")
print("📁 Lokasi:", MODEL_PATH)