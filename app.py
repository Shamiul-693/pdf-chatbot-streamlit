
import streamlit as st
from utils.pdf_reader import extract_text_from_pdf
from utils.embedder import get_embeddings, save_vector_store
from utils.chatbot import query_bot

from openai.embeddings_utils import OpenAIEmbeddings  # or load your own

st.title("PDF Chatbot")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
query = st.text_input("Ask a question")

if uploaded_file:
    with open(f"pdfs/{uploaded_file.name}", "wb") as f:
        f.write(uploaded_file.getbuffer())

    text = extract_text_from_pdf(f"pdfs/{uploaded_file.name}")
    chunks = text.split("\n\n")  # simple chunking
    model = OpenAIEmbeddings()
    embeddings = get_embeddings(chunks, model)
    index = save_vector_store(chunks, embeddings)

    if query:
        answer = query_bot(query, index, chunks, model)
        st.write(answer)
