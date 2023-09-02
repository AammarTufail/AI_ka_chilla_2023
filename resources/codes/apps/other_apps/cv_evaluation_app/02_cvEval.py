import streamlit as st
import openai
import io
import PyPDF2
from docx import Document
import config
import spacy


# Set your OpenAI API key here
openai.api_key = config.OPENAI_API_KEY

def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def extract_text_from_docx(docx_file):
    doc = Document(docx_file)
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text

def generate_summary(text, max_length=100):
    response = openai.Completion.create(
        engine="davinci",
        prompt=text,
        max_tokens=max_length
    )
    return response.choices[0].text.strip()

def evaluate_cv(cv_text, job_description):
    summarized_cv = generate_summary(cv_text, max_length=500)
    summarized_description = generate_summary(job_description, max_length=100)

    # prompt = f"CV: {summarized_cv}\n\nJob Description: {summarized_description}\n\nEvaluate:"
    # prompt = f"CV Summary: {summarized_cv}\n\nJob Description: {summarized_description}\n\nEvaluate the candidate profile and provide expert HR feedback:"
    # prompt = f"As an expert HR manager, evaluate the summarized CV based on the provided job description: \n\nJob Description:{summarized_description}\n\n Summarized CV:{summarized_cv}"
    # prompt = f"Job Description: {summarized_description}\n\n CV Summary: {summarized_cv}"
    # prompt = f"CV Summary: {summarized_cv}\n\nJob Description: {job_description}\n\nEvaluate the candidate's fit for the position and provide feedback:"
    prompt = f"CV: {cv_text}\n\nJob Description: {job_description}\n\nEvaluate:"

    response = openai.Completion.create(
        engine="davinci",
        prompt=prompt,
        max_tokens=150
    )

    evaluation = response.choices[0].text.strip()

    # Calculate the fit percentage based on similarity between CV and job description
    fit_percentage = calculate_similarity(summarized_cv, summarized_description) * 100

    return evaluation, fit_percentage

# Load the pre-trained model
nlp = spacy.load("en_core_web_md")

def calculate_similarity(text1, text2):
    doc1 = nlp(text1)
    doc2 = nlp(text2)
    
    # Calculate cosine similarity between the document vectors
    similarity = doc1.similarity(doc2)
    
    return similarity

def main():
    st.title("CV Evaluation App")

    cv_file = st.file_uploader("Upload CV (txt, pdf, docx)", type=["txt", "pdf", "docx"])
    job_description = st.text_area("Job Description", "")

    if cv_file and job_description:
        cv_text = ""

        if cv_file.type == "text/plain":
            cv_text = cv_file.read().decode("utf-8")
        elif cv_file.type == "application/pdf":
            cv_text = extract_text_from_pdf(cv_file)
        elif cv_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            cv_text = extract_text_from_docx(cv_file)

        if st.button("Evaluate CV"):
            evaluation, fit_percentage = evaluate_cv(cv_text, job_description)
            st.subheader("Evaluation Result")
            st.write(f"Candidate Fit in Job Description: {fit_percentage:.2f}%")
            st.write("Evaluation Description:")
            st.write(evaluation)

if __name__ == "__main__":
    main()
