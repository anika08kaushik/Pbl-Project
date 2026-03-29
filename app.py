import streamlit as st
import re 
from PyPDF2 import PdfReader

# ------------------ TITLE ------------------
st.title("Smart Hiring Assistant")

# ------------------ FILE UPLOAD ------------------
st.subheader("Upload Resume")

st.subheader("Enter Job Description")

job_description = st.text_area("Paste Job Description Here")

uploaded_file = st.file_uploader("Upload a PDF Resume", type=["pdf"])

# ------------------ PDF TEXT EXTRACTION ------------------
def extract_text_from_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text()
    return text

# ------------------ RESUME DETECTION ------------------
def is_resume(text):
    resume_keywords = [
        "education", "experience", "skills",
        "projects", "internship", "work", "profile"
    ]
    
    text = text.lower()
    
    match_count = sum(1 for word in resume_keywords if word in text)
    
    return match_count >= 2  # minimum keywords required

# ------------------ TEXT CLEANING ------------------
def clean_text(text):
    text = text.lower()  # lowercase everything
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # remove symbols/numbers
    text = re.sub(r'\s+', ' ', text)  # remove extra spaces
    return text

# ------------------ MAIN LOGIC ------------------
if uploaded_file is not None:
    resume_text = extract_text_from_pdf(uploaded_file)
    
    if is_resume(resume_text):
        st.success("✅ Resume detected")
        
        st.subheader("Extracted Resume Text")
        st.write(resume_text)
    else:
        st.error("❌ This does not look like a resume. Please upload a valid resume.")

# ---------------- JOB DESCRIPTION CLEANING ----------------
if job_description:
    cleaned_jd = clean_text(job_description)

    st.subheader("Cleaned Job Description")
    st.write(cleaned_jd)