import streamlit as st
from PyPDF2 import PdfReader
from typing import List
import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")


# Define Gemini LLM (Vertex AI)
class GeminiLLM:
    def __init__(self, api_key, model_name="gemini-1.5-flash"):
        self.genai = genai.configure(api_key=api_key)
        self.model_name = model_name
        self.model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            system_instruction="You are an excellent teacher for children. You easily make students understands students math problem. While Ensuring AI safety compliance against your response",
        )

    def predict(self, prompt):
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            return f"Error using Gemini: {e}"


llm = GeminiLLM(api_key=gemini_api_key)


# Function to extract text from PDFs
def extract_text_from_pdfs(files: List[bytes]) -> List[str]:
    files = [files]
    texts = []
    for file in files:
        pdf_reader = PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        texts.append(text)
    return texts


# Function to query Gemini API
def query_gemini(context: str, question: str) -> str:
    answer = llm.predict(f"context: {context} \nquestion:{question}")

    return answer
    # headers = {"Authorization": f"Bearer {GEMINI_API_KEY}"}
    # payload = {"context": context, "query": question}
    # response = requests.post(GEMINI_API_URL, headers=headers, json=payload)
    # if response.status_code == 200:
    #     return response.json().get("answer", "No response from the model.")
    # else:
    #     return f"Error: {response.status_code}, {response.text}"


# Streamlit App
st.title("AI Research Assistant")
st.subheader("Upload up to 5 Research Papers (PDFs) and Ask Questions")

# File upload section
uploaded_files = st.file_uploader(
    "Upload Research Papers (PDF format)", type=["pdf"], accept_multiple_files=False
)

# Check if files are uploaded
if uploaded_files:
    st.write("Processing uploaded files...")
    pdf_texts = extract_text_from_pdfs(uploaded_files)
    combined_context = " ".join(pdf_texts)  # Combine all PDFs' content

    print("combined_context:", combined_context)

    # Display file names
    st.write("Uploaded files:")
    uploaded_files = [uploaded_files]
    for file in uploaded_files:
        st.write(f"- {file.name}")

    # Question and Answer Section
    st.subheader("Ask Questions")
    question = st.text_input("Type your question here")

    if st.button("Ask"):
        if question.strip():
            with st.spinner("Getting the answer from Gemini..."):
                answer = query_gemini(combined_context, question)
            st.success("Answer:")
            st.write(answer)
        else:
            st.error("Please type a question.")

    # Follow-up Question Section
    st.subheader("Follow-up Question")
    follow_up_question = st.text_input("Type your follow-up question here")

    if st.button("Ask Follow-up"):
        if follow_up_question.strip():
            with st.spinner("Getting the follow-up answer from Gemini..."):
                answer = query_gemini(combined_context, follow_up_question)
            st.success("Answer:")
            st.write(answer)
        else:
            st.error("Please type a follow-up question.")
