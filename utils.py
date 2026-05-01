import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher


# -----------------------------
# 🔹 Preprocess (SMART CLEANING)
# -----------------------------
def preprocess(text):
    text = text.lower()

    # remove symbols
    text = re.sub(r'[^a-z\s]', '', text)

    # fix common broken words
    fixes = {
        "data base": "database",
        "stru ctur ed": "structured",
        "collect ion": "collection",
        "inf orma tion": "information",
        "man age": "manage",
        "stor e": "store"
    }

    for wrong, right in fixes.items():
        text = text.replace(wrong, right)

    # remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text


# -----------------------------
# 🔹 Similarity (keep but less important)
# -----------------------------
def calculate_similarity(student_text, model_text):
    student_text = preprocess(student_text)
    model_text = preprocess(model_text)

    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(3, 5))
    vectors = vectorizer.fit_transform([student_text, model_text])

    return cosine_similarity(vectors[0], vectors[1])[0][0]


# -----------------------------
# 🔹 Fuzzy keyword match (VERY IMPORTANT)
# -----------------------------
def is_similar(a, b, threshold=0.75):
    return SequenceMatcher(None, a, b).ratio() > threshold


def keyword_score(student_text, keywords):
    student_text = preprocess(student_text)
    words = student_text.split()

    count = 0

    for keyword in keywords:
        keyword = keyword.lower()

        for w in words:
            if is_similar(w, keyword):
                count += 1
                break

    return count / len(keywords)


# -----------------------------
# 🔹 FINAL SCORE (EXAM STYLE)
# -----------------------------
def final_score(similarity, keyword_score_value):

    base = (0.3 * similarity) + (0.7 * keyword_score_value)

    # strong boost only if BOTH are decent
    if keyword_score_value > 0.6 and similarity > 0.3:
        base += 0.2

    return min(base, 1.0)