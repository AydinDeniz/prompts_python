
import spacy
import openai
import os
import re
from docx import Document
from collections import defaultdict

# Load spaCy NLP model
nlp = spacy.load("en_core_web_sm")

# OpenAI API key
API_KEY = "your_openai_api_key"

# Extract text from contract document
def extract_text_from_docx(doc_path):
    doc = Document(doc_path)
    return "\n".join([para.text for para in doc.paragraphs])

# Identify contract clauses
def identify_clauses(text):
    clauses = defaultdict(str)
    
    patterns = {
        "termination": r"termination|end of contract|cancellation",
        "liability": r"liability|indemnification|responsibility",
        "payment": r"payment terms|fees|compensation",
        "confidentiality": r"confidentiality|non-disclosure|privacy",
        "jurisdiction": r"jurisdiction|governing law|dispute resolution"
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            clauses[key] = match.group()
    
    return clauses

# AI-Powered Risk Analysis
def ai_risk_analysis(text):
    prompt = f"Analyze the following legal contract text for potential risks and compliance issues:\n{text}"
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "system", "content": "You are a legal expert analyzing contract documents."},
                  {"role": "user", "content": prompt}],
        max_tokens=500
    )
    return response["choices"][0]["message"]["content"]

# Main function
if __name__ == "__main__":
    contract_file = "contract.docx"  # Replace with actual file path

    if not os.path.exists(contract_file):
        print("Error: Contract document not found!")
        exit()

    print("Extracting text from contract document...")
    contract_text = extract_text_from_docx(contract_file)

    print("Identifying contract clauses...")
    clauses = identify_clauses(contract_text)
    print("Extracted Clauses:", clauses)

    print("Performing AI-powered risk analysis...")
    risk_analysis = ai_risk_analysis(contract_text)
    print("Risk Analysis Report:\n", risk_analysis)

    # Save analysis report
    with open("contract_analysis.txt", "w", encoding="utf-8") as f:
        f.write("Extracted Clauses:\n")
        f.write(str(clauses))
        f.write("\n\nRisk Analysis Report:\n")
        f.write(risk_analysis)

    print("Contract analysis saved to 'contract_analysis.txt'.")
