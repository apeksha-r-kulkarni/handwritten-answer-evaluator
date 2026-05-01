import easyocr
from utils import calculate_similarity, keyword_score, final_score

# -----------------------------
# 🔹 Initialize EasyOCR
# -----------------------------
reader = easyocr.Reader(['en'])

# -----------------------------
# 🔹 Read image
# -----------------------------
results = reader.readtext('ex2.jpeg', detail=0)

# Join detected text
student_text = " ".join(results)

print("Extracted Text:")
print(student_text)

# -----------------------------
# 🔹 Model Answer
# -----------------------------
model_answer = """An operating system is system software that manages hardware and software resources"""
keywords = [
    "operating",
    "system",
    "software",
    "manages",
    "hardware",
    "resources"
]

# -----------------------------
# 🔹 Scoring
# -----------------------------
sim = calculate_similarity(student_text, model_answer)
k_score = keyword_score(student_text, keywords)
final = final_score(sim, k_score)

print("\nSimilarity:", sim)
print("Keyword Score:", k_score)
print("Final Score (out of 10):", final * 10)