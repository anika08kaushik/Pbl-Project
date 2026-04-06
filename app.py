import io
import streamlit as st
from PyPDF2 import PdfReader
import re
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import fuzz
import matplotlib.pyplot as plt

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

# ------------------ SKILLS LIST ------------------
skills_list = [
    "python", "java", "c++", "sql", "html", "css", "javascript",
    "machine learning", "deep learning", "data analysis",
    "pandas", "numpy", "tensorflow", "flask", "react",
    "mongodb", "excel", "power bi"
]

# Skill aliases to improve matching (alias -> canonical skill)
skill_aliases = {
    "ml": "machine learning",
    "machine-learning": "machine learning",
    "deep-learning": "deep learning",
    "js": "javascript",
    "py": "python",
    "c": "c++",
    "powerbi": "power bi",
    "nlp": "natural language processing",
}

# Skill ontology: parent -> related terms
skill_ontology = {
    "machine learning": ["ml", "supervised learning", "unsupervised learning", "deep learning"],
    "data analysis": ["pandas", "numpy", "data visualization", "excel"],
    "web development": ["html", "css", "javascript", "react", "flask"],
    "databases": ["sql", "mongodb"],
    "devops": ["docker", "kubernetes"],
    "nlp": ["nlp", "natural language processing"],
}



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


def parse_job_description(jd_text):
    """Extract required and optional skills from the job description using simple heuristics."""
    jd = jd_text or ""
    lines = [l.strip() for l in jd.splitlines() if l.strip()]
    required = set()
    optional = set()

    # Look for explicit sections
    current = None
    for line in lines:
        low = line.lower()
        if low.startswith("required") or "must have" in low:
            current = "required"
            continue
        if low.startswith("optional") or "nice to have" in low:
            current = "optional"
            continue
        # if line contains skill terms, extract
        found = extract_skills(line)
        if current == "required":
            required.update(found)
        elif current == "optional":
            optional.update(found)
        else:
            # distribute into required if words like 'required' in the line
            if "required" in low or "must" in low:
                required.update(found)
            else:
                # default to required
                required.update(found)

    return {"required": sorted(required), "optional": sorted(optional)}

def extract_skills(text):
    """Extract skills from text using exact, alias, and fuzzy matching."""
    found = set()
    # normalize text for matching
    text = text.lower()
    # sentences for localized fuzzy matching
    sentences = [s.strip() for s in re.split(r'[\n\r\.]+', text) if s.strip()]

    for skill in skills_list:
        canonical = skill.lower()

        # 1) exact match
        if re.search(r'\b' + re.escape(canonical) + r'\b', text):
            found.add(skill)
            continue

        # 2) alias match
        for alias, target in skill_aliases.items():
            if target == canonical:
                if re.search(r'\b' + re.escape(alias) + r'\b', text):
                    found.add(skill)
                    break
        if skill in found:
            continue

        # 3) fuzzy match against sentences (handles minor typos/plurals)
        for s in sentences:
            score = fuzz.partial_ratio(canonical, s)
            if score >= 80:
                found.add(skill)
                break

    # Map found related terms to ontology parents too
    parents_found = set()
    for parent, terms in skill_ontology.items():
        parent_lower = parent.lower()
        # if parent already matched directly
        if any(parent_lower == f.lower() for f in found):
            parents_found.add(parent)
            continue
        # check related terms
        for t in terms:
            t_lower = t.lower()
            if re.search(r'\b' + re.escape(t_lower) + r'\b', text):
                parents_found.add(parent)
                break
            # fuzzy on related terms
            for s in sentences:
                if fuzz.partial_ratio(t_lower, s) >= 85:
                    parents_found.add(parent)
                    break
    # Combine canonical found skills (avoid duplicates)
    final_skills = set(found)
    # replace related term matches by parent when parent not already in final_skills
    for p in parents_found:
        # keep parent as canonical
        final_skills.add(p)

    return sorted(final_skills)

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

            cleaned_resume = resume_text  # keep original for sections
            cleaned_jd = job_description

            # parse sections
            resume_sections = split_sections(cleaned_resume)
            jd_parsed = parse_job_description(cleaned_jd)

            # skills
            resume_skills = extract_skills(resume_sections.get("skills", "") + "\n" + cleaned_resume)
            jd_required = jd_parsed.get("required", [])
            jd_optional = jd_parsed.get("optional", [])

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

            common_skills = set(resume_skills) & set(jd_required + jd_optional)
            missing_required = set(jd_required) - set(resume_skills)
            missing_optional = set(jd_optional) - set(resume_skills)

            # skill_score: required and optional weighted (required more important)
            req_score = (len(set(resume_skills) & set(jd_required)) / len(jd_required) * 100) if jd_required else 0
            opt_score = (len(set(resume_skills) & set(jd_optional)) / len(jd_optional) * 100) if jd_optional else 0
            skill_score = (0.75 * req_score) + (0.25 * opt_score)

            # experience relevance: use experience_sim normalized
            experience_score = experience_sim * 100

            # final score: 0.5 semantic + 0.3 skill + 0.2 experience
            final_score = (0.5 * match_score * 100) + (0.3 * skill_score) + (0.2 * experience_score)

            st.subheader("📊 Your Result")
            st.success(f"🎯 Final ATS Score: {final_score:.2f}%")

            st.write(f"🔹 Overall Match: {match_score*100:.2f}%")
            st.write(f"🔹 Skill Match: {skill_score:.2f}%")
            st.write(f"🔹 Experience Relevance: {experience_score:.2f}%")

            st.subheader("✅ Matching Skills")
            st.write(sorted(list(common_skills)))

            st.subheader("❌ Missing Required Skills")
            st.write(sorted(list(missing_required)))

            st.subheader("ℹ️ Suggestions")
            suggestions = []
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

        cleaned_jd = job_description
        jd_parsed = parse_job_description(cleaned_jd)
        jd_skills = jd_parsed.get("required", []) + jd_parsed.get("optional", [])

        results = []

        for file in uploaded_files:

            resume_text = extract_text_from_pdf(file)

            if not is_resume(resume_text):
                continue

            resume_sections = split_sections(resume_text)
            resume_skills = extract_skills(resume_sections.get("skills", "") + "\n" + resume_text)

            overall_sim = calculate_match_score(resume_text, cleaned_jd)

            # section embeddings
            resume_embeds = section_embeddings(resume_sections)
            jd_sections = {"skills": ' '.join(jd_parsed.get("required", [])), "experience": cleaned_jd, "projects": cleaned_jd}
            jd_embeds = section_embeddings(jd_sections)

            experience_sim = 0.0
            try:
                if resume_embeds.get("experience") is not None and jd_embeds.get("experience") is not None:
                    experience_sim = float(cosine_similarity([resume_embeds["experience"]], [jd_embeds["experience"]])[0][0])
            except Exception:
                experience_sim = 0.0

            jd_required = jd_parsed.get("required", [])
            jd_optional = jd_parsed.get("optional", [])

            common_skills = set(resume_skills) & set(jd_required + jd_optional)
            missing_required = set(jd_required) - set(resume_skills)

            req_score = (len(set(resume_skills) & set(jd_required)) / len(jd_required) * 100) if jd_required else 0
            opt_score = (len(set(resume_skills) & set(jd_optional)) / len(jd_optional) * 100) if jd_optional else 0
            skill_score = (0.75 * req_score) + (0.25 * opt_score)

            experience_score = experience_sim * 100

            final_score = (0.5 * overall_sim * 100) + (0.3 * skill_score) + (0.2 * experience_score)

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

            # Pie chart for top candidate skill match vs missing
            top = results[0]
            matched = len(top["Matched Skills"].split(", ")) if top["Matched Skills"] else 0
            missing = len(top["Missing Required"].split(", ")) if top["Missing Required"] else 0
            fig, ax = plt.subplots()
            ax.pie([matched, missing], labels=["Matched", "Missing"], autopct="%1.1f%%", colors=["#4CAF50", "#F44336"])
            ax.set_title("Top candidate: Skill match vs Missing required")
            st.pyplot(fig)

            # allow exporting full results
            csv_buf = io.StringIO()
            df.to_csv(csv_buf, index=False)
            st.download_button("Download all results as CSV", data=csv_buf.getvalue(), file_name="candidate_ranking.csv")

        else:
            st.error("❌ No valid resumes found")

# ------------------ DEFAULT ------------------

if role is None:
    st.info("👆 Please select your role to continue")