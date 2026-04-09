import io
import streamlit as st
from PyPDF2 import PdfReader
import re
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import fuzz
import matplotlib.pyplot as plt
import yake


# Caching helpers: cache the transformer model and embedding computations
@st.cache_resource
def load_embedding_model():
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer("all-MiniLM-L6-v2")
    except ModuleNotFoundError as e:
        return None

@st.cache_data
def encode_texts(texts_tuple):
    """Encode a tuple of texts using the cached model. Returns numpy array or None."""
    model = load_embedding_model()
    if model is None:
        return None
    return model.encode(list(texts_tuple), convert_to_numpy=True)

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


# ------------------ FUNCTIONS ------------------

def extract_text_from_pdf(file):
    reader = PdfReader(file)
    text = ""
    try:
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    except Exception:
        # fall back to best-effort empty string
        text = ""

    # If simple extraction returned very little, try OCR fallback if available
    if len(text.strip()) < 20:
        try:
            from pdf2image import convert_from_bytes
            import pytesseract
            images = convert_from_bytes(file.read())
            ocr_text = ""
            for img in images:
                ocr_text += pytesseract.image_to_string(img) + "\n"
            if ocr_text.strip():
                return ocr_text
        except Exception:
            # OCR dependencies missing or failed — return whatever we have
            pass

    return text

def is_resume(text):
    keywords = ["education", "experience", "skills", "projects", "internship", "work"]
    text = text.lower()
    return sum(1 for word in keywords if word in text) >= 2


def split_sections(text):
    """Split resume text into sections (skills, experience, projects, education, other)."""
    sections = {"skills": "", "experience": "", "projects": "", "education": "", "other": ""}
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    current = "other"
    for line in lines:
        low = line.lower()
        if re.search(r"^skills?:?$", low) or "skills" in low and len(line.split()) <= 4:
            current = "skills"
            continue
        if re.search(r"^experience:?$", low) or "experience" in low:
            current = "experience"
            continue
        if re.search(r"^projects?:?$", low) or "project" in low:
            current = "projects"
            continue
        if re.search(r"^education:?$", low) or "education" in low:
            current = "education"
            continue
        sections[current] += line + "\n"
    return sections

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# def is_valid_skill(phrase):
#     words = phrase.split()

#     blacklist = ["engineer", "developer", "experience", "work", "team", "system", "role", "skills"]

#     if any(b in phrase for b in blacklist):
#         return False

#     # ❌ reject weird phrases like "skills python"
#     if len(words) == 2 and words[0] in ["skills", "experience"]:
#         return False

#     if len(words) > 2:
#         return False

#     return True

def extract_experience_years(text):
    text = text.lower()

    patterns = [
        r'(\d+)\+?\s*years',
        r'(\d+)\s*yrs',
        r'(\d+)\s*year'
    ]

    years = []

    for pattern in patterns:
        matches = re.findall(pattern, text)
        for m in matches:
            try:
                years.append(int(m))
            except:
                pass

    return max(years) if years else 0

def parse_job_description(jd_text):
    skills = extract_skills(jd_text)

    return {
        "required": skills,   # 🔥 take all skills
        "optional": []
    }

def extract_skills(text):
    text = text.lower()

    skill_section = ""

    match = re.search(r"skills[:\n](.*?)(experience|projects|education|$)", text, re.S)
    
    if match:
        skill_section = match.group(1)
    else:
        skill_section = text

    # 🔥 YAKE keyword extraction
    kw_extractor = yake.KeywordExtractor(
        lan="en",
        n=2,        # 1–2 word phrases
        top=25      # number of keywords
    )

    keywords = kw_extractor.extract_keywords(skill_section)

    skills = [kw[0].lower() for kw in keywords]

    return list(set(skills))


def extract_jd_skills(jd_text):
    jd_text = jd_text.lower()

    skills = []

    match = re.search(r"(required skills:.*?)(responsibilities:|$)", jd_text, re.S)
    if match:
        skills_text = match.group(1)
        skills += re.findall(r'\b[a-zA-Z\+\#]{2,}\b', skills_text)

    match2 = re.search(r"(preferred skills:.*?)(responsibilities:|$)", jd_text, re.S)
    if match2:
        skills_text = match2.group(1)
        skills += re.findall(r'\b[a-zA-Z\+\#]{2,}\b', skills_text)

    return list(set(skills))

def clean_skills(skills):
    return [
        s for s in skills
        if len(s) > 2
        and len(s.split()) <= 3
        and not s.isdigit()
    ]

def enrich_skills(text, skills):
    model = load_embedding_model()
    if model is None:
        return skills

    words = list(set(re.findall(r'\b[A-Za-z\+\#]{2,}\b', text)))

    # encode JD/resume context
    text_vec = model.encode([text])[0]
    word_vecs = model.encode(words)

    refined = []

    for i, w in enumerate(words):
        sim = cosine_similarity([word_vecs[i]], [text_vec])[0][0]

        # 🔥 keep only context-relevant words
        if sim > 0.4 and 3 < len(w) <= 12:
            refined.append(w.lower())

    return list(set(skills + refined))

def calculate_match_score(resume, jd):
    # overall semantic similarity between resume and job description
    model = load_embedding_model()
    if model is None:
        return 0.0
    try:
        vecs = encode_texts((resume, jd))
        if vecs is None:
            return 0.0
        similarity = cosine_similarity([vecs[0]], [vecs[1]])
        return float(similarity[0][0])
    except Exception:
        return 0.0
    
def semantic_match(skill1, skill2):
    model = load_embedding_model()
    if model is None:
        return False

    vecs = model.encode([skill1, skill2])
    sim = cosine_similarity([vecs[0]], [vecs[1]])[0][0]

    return sim > 0.6


def section_embeddings(section_texts):
    """Encode a dict of section_name->text and return name->embedding dict."""
    names = []
    texts = []
    for name, t in section_texts.items():
        names.append(name)
        texts.append(t if t else "")
    res = encode_texts(tuple(texts))
    if res is None:
        return {n: None for n in names}
    return {n: res[i] for i, n in enumerate(names)}

def match_skills(resume_skills, jd_skills, model=None):
    if model is None:
        model = load_embedding_model()
    if model is None:
        return set()

    if not resume_skills or not jd_skills:
        return set()

    resume_vecs = model.encode(resume_skills)
    jd_vecs = model.encode(jd_skills)

    sim_matrix = cosine_similarity(resume_vecs, jd_vecs)

    matched = set()

    for i in range(len(resume_skills)):
        for j in range(len(jd_skills)):
            if sim_matrix[i][j] > 0.5 or fuzz.partial_ratio(resume_skills[i], jd_skills[j]) > 85:
                matched.add(resume_skills[i])
                break

    return matched

def extract_skill_sections(jd_text):
    jd_text = jd_text.lower()

    required = ""
    preferred = ""

    req_match = re.search(r'required skills:(.*?)(preferred skills:|responsibilities:)', jd_text, re.S)
    if req_match:
        required = req_match.group(1)

    pref_match = re.search(r'preferred skills:(.*?)(responsibilities:|qualifications:)', jd_text, re.S)
    if pref_match:
        preferred = pref_match.group(1)

    return required + " " + preferred
# ------------------ CANDIDATE FLOW ------------------

if role == "Candidate":

    st.subheader("📄 Upload Your Resume")
    uploaded_file = st.file_uploader("Upload your resume (PDF)", type=["pdf"])

    st.subheader("📝 Enter Job Description")
    job_description = st.text_area("Paste job description here")

    if uploaded_file and job_description:

        model = load_embedding_model()

        st.success(f"Uploaded: {uploaded_file.name}")

        resume_text = extract_text_from_pdf(uploaded_file)

        if is_resume(resume_text):

            cleaned_resume = resume_text  # keep original for sections
            cleaned_jd = job_description

            # parse sections
            resume_sections = split_sections(cleaned_resume)
            resume_skills = extract_skills(cleaned_resume)
            resume_skills = enrich_skills(cleaned_resume, resume_skills)
            resume_skills = clean_skills(resume_skills)
            jd_required = extract_jd_skills(cleaned_jd)
            jd_optional = []

            # 🔥 FIX: fallback if no skills extracted
            if len(jd_required) == 0:
                jd_required = extract_skills(cleaned_jd[:500])
                jd_required = enrich_skills(cleaned_jd, jd_required)
                jd_required = clean_skills(jd_required)

            # semantic similarities: overall and per-section
            overall_sim = calculate_match_score(cleaned_resume, cleaned_jd)

            # embed sections for fine-grained similarity
            resume_embeds = section_embeddings(resume_sections)
            jd_sections = {"skills": ' '.join(jd_required), "experience": cleaned_jd, "projects": cleaned_jd}
            jd_embeds = section_embeddings(jd_sections)

            # compute section similarities (experience/projects use jd as proxy)
            experience_sim = 0.0
            projects_sim = 0.0
            skills_sim = 0.0

            resume_exp_years = extract_experience_years(cleaned_resume)
            jd_exp_years = extract_experience_years(cleaned_jd)

            try:
                if resume_embeds.get("experience") is not None and jd_embeds.get("experience") is not None:
                    experience_sim = float(cosine_similarity([resume_embeds["experience"]], [jd_embeds["experience"]])[0][0])
                if resume_embeds.get("projects") is not None and jd_embeds.get("projects") is not None:
                    projects_sim = float(cosine_similarity([resume_embeds["projects"]], [jd_embeds["projects"]])[0][0])
                if resume_embeds.get("skills") is not None and jd_embeds.get("skills") is not None:
                    skills_sim = float(cosine_similarity([resume_embeds["skills"]], [jd_embeds["skills"]])[0][0])
            except Exception:
                experience_sim = projects_sim = skills_sim = 0.0

            match_score = overall_sim

            common_skills = match_skills(resume_skills, jd_required + jd_optional, model)
            missing_required = set(jd_required) - common_skills
            missing_optional = set(jd_optional) - set(resume_skills)

            # skill_score: required and optional weighted (required more important)
            matched_required = match_skills(resume_skills, jd_required, model)
            req_score = (len(matched_required) / len(jd_required) * 100) if jd_required else 0
            matched_optional = match_skills(resume_skills, jd_optional, model)
            opt_score = (len(matched_optional) / len(jd_optional) * 100) if jd_optional else 0
            skill_score = (0.75 * req_score) + (0.25 * opt_score)

            # experience relevance: use experience_sim normalized
            if jd_exp_years > 0:
                exp_ratio = min(resume_exp_years / jd_exp_years, 1)
                experience_score = exp_ratio * 100
            else:
                experience_score = experience_sim * 100

            # final score: 0.5 semantic + 0.3 skill + 0.2 experience
            final_score = (0.4 * match_score * 100) + (0.4 * skill_score) + (0.2 * experience_score)

            st.subheader("📊 Your Result")
            st.success(f"🎯 Final ATS Score: {final_score:.2f}%")

            st.write(f"🔹 Overall Match: {match_score*100:.2f}%")
            st.write(f"🔹 Skill Match: {skill_score:.2f}%")
            st.write(f"🧠 Detected Experience: {resume_exp_years} years")

            st.subheader("✅ Matching Skills")
            st.write(sorted(list(common_skills)))

            st.subheader("❌ Missing Required Skills")
            st.write(sorted(list(missing_required)))

            st.subheader("ℹ️ Suggestions")
            suggestions = []
            if jd_exp_years > 0 and resume_exp_years < jd_exp_years:
                suggestions.append(
                f"You have {resume_exp_years} years of experience, but the job requires {jd_exp_years}+ years."
            )
            if experience_score < 40:
                suggestions.append("Consider adding more detailed experience descriptions relevant to the job.")
            if len(missing_required) > 0:
                for m in missing_required:
                    suggestions.append(f"Consider adding or highlighting your experience with: {m}")
            if match_score * 100 < 40:
                suggestions.append("Include more relevant projects and keywords from the job description.")

            if suggestions:
                for s in suggestions:
                    st.write("- ", s)
            else:
                st.write("Your resume looks well-aligned with the job description.")

            # Highlight matched skills in resume text
            highlighted = cleaned_resume
            for sk in sorted(list(common_skills)):
                highlighted = re.sub(r'(?i)(' + re.escape(sk) + r')', r"**\1**", highlighted)
            st.subheader("📄 Resume Preview (matched skills highlighted)")
            st.markdown(highlighted)

            # Export results
            result_row = {
                "Resume": uploaded_file.name,
                "Final Score (%)": round(final_score, 2),
                "Overall (%)": round(match_score * 100, 2),
                "Skill Match (%)": round(skill_score, 2),
                "Experience (%)": round(experience_score, 2),
                "Matched Skills": ", ".join(sorted(list(common_skills)))
            }
            csv_buf = io.StringIO()
            pd.DataFrame([result_row]).to_csv(csv_buf, index=False)
            st.download_button("Download result as CSV", data=csv_buf.getvalue(), file_name=f"{uploaded_file.name}_result.csv")

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

        model = load_embedding_model()

        cleaned_jd = job_description

        # 🔥 NEW: extract from FULL JD
        jd_required = extract_jd_skills(cleaned_jd)
        jd_optional = []

        results = []

        for file in uploaded_files:

            resume_text = extract_text_from_pdf(file)

            if not is_resume(resume_text):
                continue

            resume_sections = split_sections(resume_text)

            # 🔥 NEW: clean skill extraction
            resume_skills = clean_skills(extract_skills(resume_text))

            # overall similarity
            overall_sim = calculate_match_score(resume_text, cleaned_jd)

            # section embeddings
            resume_embeds = section_embeddings(resume_sections)
            jd_sections = {
                "skills": ' '.join(jd_required),
                "experience": cleaned_jd,
                "projects": cleaned_jd
            }
            jd_embeds = section_embeddings(jd_sections)

            # experience similarity
            experience_sim = 0.0
            resume_exp_years = extract_experience_years(resume_text)
            jd_exp_years = extract_experience_years(cleaned_jd)

            try:
                if resume_embeds.get("experience") is not None and jd_embeds.get("experience") is not None:
                    experience_sim = float(
                        cosine_similarity(
                            [resume_embeds["experience"]],
                            [jd_embeds["experience"]]
                        )[0][0]
                    )
            except:
                experience_sim = 0.0

            # 🔥 SKILL MATCHING
            common_skills = match_skills(resume_skills, jd_required, model)
            missing_required = set(jd_required) - common_skills

            matched_required = match_skills(resume_skills, jd_required, model)
            req_score = (len(matched_required) / len(jd_required) * 100) if jd_required else 0

            skill_score = req_score  # no optional now

            # experience score
            if jd_exp_years > 0:
                exp_ratio = min(resume_exp_years / jd_exp_years, 1)
                experience_score = exp_ratio * 100
            else:
                experience_score = experience_sim * 100

            # final score
            final_score = (
                (0.4 * overall_sim * 100) +
                (0.4 * skill_score) +
                (0.2 * experience_score)
            )

            results.append({
                "Resume": file.name,
                "Final Score (%)": round(final_score, 2),
                "Overall (%)": round(overall_sim * 100, 2),
                "Skill Match (%)": round(skill_score, 2),
                "Experience (%)": round(experience_score, 2),
                "Matched Skills": ", ".join(sorted(list(common_skills))),
                "Missing Required": ", ".join(sorted(list(missing_required)))
            })

        results = sorted(results, key=lambda x: x["Final Score (%)"], reverse=True)

        st.subheader("🏆 Candidate Ranking")

        if results:
            df = pd.DataFrame(results)
            st.dataframe(df)

            st.success(f"🥇 Top Candidate: {results[0]['Resume']} ({results[0]['Final Score (%)']}%)")

            st.subheader("📊 Score Visualization")
            st.bar_chart(df.set_index("Resume")["Final Score (%)"])

            # pie chart
            top = results[0]
            matched = len(top["Matched Skills"].split(", ")) if top["Matched Skills"] else 0
            missing = len(top["Missing Required"].split(", ")) if top["Missing Required"] else 0

            fig, ax = plt.subplots()
            ax.pie(
                [matched, missing],
                labels=["Matched", "Missing"],
                autopct="%1.1f%%",
                colors=["#4CAF50", "#F44336"]
            )
            ax.set_title("Top candidate: Skill match vs Missing required")
            st.pyplot(fig)

            # export
            csv_buf = io.StringIO()
            df.to_csv(csv_buf, index=False)
            st.download_button(
                "Download all results as CSV",
                data=csv_buf.getvalue(),
                file_name="candidate_ranking.csv"
            )

        else:
            st.error("❌ No valid resumes found")

# ------------------ DEFAULT ------------------

if role is None:
    st.info("👆 Please select your role to continue")