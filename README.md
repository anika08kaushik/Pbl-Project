#  Smart Hiring Assistant

Smart Hiring Assistant is an AI-powered web application that helps **candidates evaluate their resumes** and allows **recruiters to efficiently shortlist candidates** using intelligent matching techniques.

---

##  Features
  
###  Candidate Mode

* Upload resume (PDF)
* Enter job description
* Get ATS score with:

  * Overall match %
  * Skill match %
  * Matching & missing skills

###  Recruiter Mode

* Upload multiple resumes
* Rank candidates based on score
* View top candidate
* Visualize results with charts

---

##  How It Works

* Extracts text from resumes (PDF)
* Cleans and processes text
* Identifies relevant skills
* Uses **TF-IDF + Cosine Similarity**
* Calculates final ATS score:

  **Final Score = 70% similarity + 30% skill match**

---

##  Tech Stack

* Python
* Streamlit
* scikit-learn
* PyPDF2
* pandas

---

##  Setup

```bash id="lq6qjr"
git clone https://github.com/your-username/Smart-Hiring-Assistant.git
cd Smart-Hiring-Assistant
pip install -r requirements.txt
streamlit run app.py
```

---

##  Use Cases

* Resume screening
* ATS score checking
* Candidate ranking
* Academic / hackathon project

---


