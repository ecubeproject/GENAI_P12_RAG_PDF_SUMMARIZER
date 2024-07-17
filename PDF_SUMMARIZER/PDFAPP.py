# PDF_Summarizer.py
import os
import streamlit as st
import PyPDF2
from PyPDF2 import PdfReader
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.summarize import load_summarize_chain
from langchain import OpenAI
import tempfile

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Streamlit setup
st.set_page_config(
    page_title="AI BASED PDF SUMMARIZER",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown("<h1 style='text-align: center; color: black;'>AI BASED PDF SUMMARIZER</h1>", unsafe_allow_html=True)
st.sidebar.markdown("<h3 style='text-align: center; color: black;'>Assistant Console</h3>", unsafe_allow_html=True)

# Initialize pdf_vectors globally
pdf_vectors = None

# ---- PDF Loading & Embedding ----
uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type=['pdf'])
if uploaded_file:
    pdf_reader = PdfReader(uploaded_file)
    pdf_text = ""
    for page in pdf_reader.pages:
        pdf_text += page.extract_text()
    text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", "."], chunk_size=500)
    pdf_docs = text_splitter.split_text(pdf_text)
    if pdf_docs:
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        pdf_vectors = FAISS.from_texts(pdf_docs, embeddings)
        st.sidebar.success("PDF processed successfully!")

# ---- Query Interface ----
llm = OpenAI(temperature=0.9, max_tokens=500, openai_api_key=OPENAI_API_KEY)
query_pdf = st.text_input('Ask your question about PDFs:')
if query_pdf:
    if st.button("Summarize PDF", key="query_pdf_button"):
        if pdf_vectors is not None:
            docs = pdf_vectors.similarity_search(query_pdf)
            chain = load_qa_chain(llm, chain_type="stuff")
            response = chain.run(input_documents=docs, question=query_pdf)
            st.write(response)
        else:
            st.error("Please upload a PDF and ensure it is processed successfully before querying.")

# ---- Summarize Entire PDF ----
if uploaded_file and st.sidebar.button("Summarize Entire PDF", key="summarize_pdf_button"):
    def summarize_pdfs_from_folder(pdfs_folder):
        summaries = []
        for pdf_file in pdfs_folder:
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_path = temp_file.name
                temp_file.write(pdf_file.getvalue())
            loader = PyPDFLoader(temp_path)
            docs = loader.load_and_split()
            chain = load_summarize_chain(llm, chain_type="map_reduce")
            summary = chain.run(docs)
            summaries.append(summary)
            os.remove(temp_path)
        return summaries

    summaries = summarize_pdfs_from_folder([uploaded_file])
    for summary in summaries:
        st.write(summary)
