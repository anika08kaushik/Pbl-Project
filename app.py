import streamlit as st
from PyPDF2 import PdfReader
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="Smart Hiring Assistant", layout="wide")

# ------------------ TITLE ------------------
st.title("💼 Smart Hiring Assistant")

# ------------------ SESSION STATE FOR ROLE ------------------
if "role" not in st.session_state:
    st.session_state.role = None

# ------------------ ROLE SELECTION UI ------------------
st.markdown("## What do you want to do?")

col1, col2 = st.columns(2)

with col1:
    if st.button("👤 I am a Candidate"):
        st.session_state.role = "Candidate"

with col2:
    if st.button("🧑‍💼 I am a Recruiter"):
        st.session_state.role = "Recruiter"

# Reset button
if st.session_state.role:
    if st.button("🔄 Reset"):
        st.session_state.role = None

role = st.session_state.role

# ------------------ SKILLS LIST ------------------
skills_list = [
    "python", "java", "c++", "sql", "html", "css", "javascript",
    "machine learning", "deep learning", "data analysis",
    "pandas", "numpy", "tensorflow", "flask", "react",
    "mongodb", "excel", "power bi"
]

# ------------------ FUNCTIONS ------------------

def extract_text_from_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text

def is_resume(text):
    keywords = ["education", "experience", "skills", "projects", "internship", "work"]
    text = text.lower()
    return sum(1 for word in keywords if word in text) >= 2

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_skills(text):
    found = []
    for skill in skills_list:
        if re.search(r'\b' + re.escape(skill) + r'\b', text):
            found.append(skill)
    return found

def calculate_match_score(resume, jd):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([resume, jd])
    similarity = cosine_similarity(vectors[0:1], vectors[1:2])
    return similarity[0][0]

# ------------------ CANDIDATE FLOW ------------------

if role == "Candidate":

    st.subheader("📄 Upload Your Resume")
    uploaded_file = st.file_uploader("Upload your resume (PDF)", type=["pdf"])

    st.subheader("📝 Enter Job Description")
    job_description = st.text_area("Paste job description here")

    if uploaded_file and job_description:

        st.success(f"Uploaded: {uploaded_file.name}")

        resume_text = extract_text_from_pdf(uploaded_file)

        if is_resume(resume_text):

            cleaned_resume = clean_text(resume_text)
            cleaned_jd = clean_text(job_description)

            resume_skills = extract_skills(cleaned_resume)
            jd_skills = extract_skills(cleaned_jd)

            match_score = calculate_match_score(cleaned_resume, cleaned_jd)

            common_skills = set(resume_skills) & set(jd_skills)
            missing_skills = set(jd_skills) - set(resume_skills)

            skill_score = (len(common_skills) / len(jd_skills) * 100) if jd_skills else 0

            final_score = (0.7 * match_score * 100) + (0.3 * skill_score)

            st.subheader("📊 Your Result")
            st.success(f"🎯 Final ATS Score: {final_score:.2f}%")

            st.write(f"🔹 Overall Match: {match_score*100:.2f}%")
            st.write(f"🔹 Skill Match: {skill_score:.2f}%")

            st.subheader("✅ Matching Skills")
            st.write(list(common_skills))

            st.subheader("❌ Missing Skills")
            st.write(list(missing_skills))

        else:
            st.error("❌ This does not look like a valid resume")

# ------------------ RECRUITER FLOW ------------------

elif role == "Recruiter":

    st.subheader("📂 Upload Candidate Resumes")
    uploaded_files = st.file_uploader(
        "Upload multiple resumes (PDF)",
        type=["pdf"],
        accept_multiple_files=True
    )

    st.subheader("📝 Enter Job Description")
    job_description = st.text_area("Paste job description here")

    if uploaded_files:
        st.success(f"{len(uploaded_files)} resumes uploaded")

    if uploaded_files and job_description:

        cleaned_jd = clean_text(job_description)
        jd_skills = extract_skills(cleaned_jd)

        results = []

        for file in uploaded_files:

            resume_text = extract_text_from_pdf(file)

            if not is_resume(resume_text):
                continue

            cleaned_resume = clean_text(resume_text)
            resume_skills = extract_skills(cleaned_resume)

            match_score = calculate_match_score(cleaned_resume, cleaned_jd)

            common_skills = set(resume_skills) & set(jd_skills)
            missing_skills = set(jd_skills) - set(resume_skills)

            skill_score = (len(common_skills) / len(jd_skills) * 100) if jd_skills else 0

            final_score = (0.7 * match_score * 100) + (0.3 * skill_score)

            results.append({
                "Resume": file.name,
                "Final Score (%)": round(final_score, 2),
                "TF-IDF (%)": round(match_score * 100, 2),
                "Skill Match (%)": round(skill_score, 2),
                "Matched Skills": ", ".join(common_skills),
                "Missing Skills": ", ".join(missing_skills)
            })

        results = sorted(results, key=lambda x: x["Final Score (%)"], reverse=True)

        st.subheader("🏆 Candidate Ranking")

        if results:
            df = pd.DataFrame(results)
            st.dataframe(df)

            st.success(f"🥇 Top Candidate: {results[0]['Resume']} ({results[0]['Final Score (%)']}%)")

            st.subheader("📊 Score Visualization")
            st.bar_chart(df.set_index("Resume")["Final Score (%)"])

        else:
            st.error("❌ No valid resumes found")

# ------------------ DEFAULT ------------------

if role is None:
    st.info("👆 Please select your role to continue")