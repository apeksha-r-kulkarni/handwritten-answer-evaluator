import streamlit as st
import pytesseract
from PIL import Image
from utils import calculate_similarity, keyword_score, final_score

# Connect Tesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

st.title("🧠 Handwritten Answer Evaluation System")

# Upload image
uploaded_file = st.file_uploader("Upload handwritten answer", type=["jpg", "jpeg", "png"])

# Model answer input
model_answer = st.text_area("Enter model answer")

# Keywords
keywords_input = st.text_input("Enter keywords (comma separated)")
keywords = [k.strip() for k in keywords_input.split(",") if k]

if st.button("Evaluate"):
    if uploaded_file and model_answer:
        
        # Load image
        img = Image.open(uploaded_file)
        
        # OCR
        student_text = pytesseract.image_to_string(img)
        
        st.subheader("📄 Extracted Text")
        st.write(student_text)

        # Calculate scores
        sim = calculate_similarity(student_text, model_answer)
        k_score = keyword_score(student_text, keywords)
        final = final_score(sim, k_score)

        # Display results
        st.subheader("📊 Results")
        st.write(f"Similarity: {sim:.2f}")
        st.write(f"Keyword Score: {k_score:.2f}")
        st.write(f"Final Score (out of 10): {final*10:.2f}")

        # Missing keywords
        missing = [k for k in keywords if k.lower() not in student_text.lower()]
        if missing:
            st.subheader("⚠️ Missing Keywords")
            st.write(", ".join(missing))

    else:
        st.warning("Please upload image and enter model answer")