
import os
import spacy
import nltk
import pdfplumber
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load NLP models
nlp = spacy.load("en_core_web_sm")
nltk.download("punkt")

# Extract text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

# Extract key details from resume
def extract_resume_details(text):
    doc = nlp(text)
    skills = set()
    experience = []
    education = []
    
    for ent in doc.ents:
        if ent.label_ == "ORG":
            education.append(ent.text)
        elif ent.label_ in ["DATE", "TIME", "ORDINAL"]:
            experience.append(ent.text)
    
    for token in doc:
        if token.pos_ in ["NOUN", "PROPN"] and token.text.lower() not in skills:
            skills.add(token.text.lower())

    return {
        "skills": list(skills),
        "experience": experience,
        "education": education
    }

# Compute similarity between resume and job description
def compute_similarity(resume_texts, job_desc):
    vectorizer = TfidfVectorizer(stop_words="english")
    all_texts = resume_texts + [job_desc]
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    
    similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
    return similarities.flatten()

# Rank candidates based on job description
def rank_candidates(resume_folder, job_description):
    resumes = []
    resume_texts = []
    
    for filename in os.listdir(resume_folder):
        if filename.endswith(".pdf"):
            resume_text = extract_text_from_pdf(os.path.join(resume_folder, filename))
            resumes.append(filename)
            resume_texts.append(resume_text)
    
    similarity_scores = compute_similarity(resume_texts, job_description)
    ranked_candidates = sorted(zip(resumes, similarity_scores), key=lambda x: x[1], reverse=True)
    
    return ranked_candidates

if __name__ == "__main__":
    resume_folder = "resumes/"  # Folder containing PDF resumes
    job_description = "Looking for a Python developer with experience in AI, NLP, and data science."
    
    print("Ranking candidates...")
    ranked_candidates = rank_candidates(resume_folder, job_description)
    
    for rank, (resume, score) in enumerate(ranked_candidates, start=1):
        print(f"{rank}. {resume} - Similarity Score: {score:.4f}")
