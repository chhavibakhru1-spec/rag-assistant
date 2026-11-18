
import streamlit as st
import fitz
import re
import unicodedata
import os
import faiss
import numpy as np

from sentence_transformers import SentenceTransformer
from google import genai

# ---------------------------
# PAGE CONFIGURATION
# ---------------------------

st.set_page_config(
    page_title="AI Document Assistant",
    page_icon="📚",
    layout="wide"
)

st.title("📚 AI Document Assistant")
st.write("Upload your PDF and ask questions about it.")

# ---------------------------
# GEMINI API CONFIGURATION
# ---------------------------

api_key = os.environ.get("GEMINI_API_KEY")

if not api_key:
    st.error("Gemini API key not configured.")
    st.stop()

client = genai.Client(api_key=api_key)

# ---------------------------
# LOAD EMBEDDING MODEL
# ---------------------------

@st.cache_resource
def load_model():

    return SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

model = load_model()

# ---------------------------
# TEXT CLEANING
# ---------------------------

def clean_text(text):

    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\x00", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n+", "\n", text)

    return text.strip()

# ---------------------------
# PDF EXTRACTION AND CHUNKING
# ---------------------------

def process_pdf(pdf_file, chunk_size=500, overlap=100):

    chunks = []

    doc = fitz.open(
        stream=pdf_file.getvalue(),
        filetype="pdf"
    )

    for page_number, page in enumerate(doc):

        text = clean_text(page.get_text("text"))

        start = 0

        while start < len(text):

            end = start + chunk_size

            chunk_text = text[start:end]

            if chunk_text.strip():

                chunks.append({
                    "text": chunk_text,
                    "page": page_number + 1,
                    "source": pdf_file.name
                })

            start += chunk_size - overlap

    doc.close()

    return chunks

# ---------------------------
# CREATE VECTOR DATABASE
# ---------------------------

def create_vector_store(chunks):

    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        batch_size=16,
        show_progress_bar=False,
        normalize_embeddings=True
    )

    embeddings = embeddings.astype("float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])

    index.add(embeddings)

    return index

# ---------------------------
# DOCUMENT RETRIEVAL
# ---------------------------

def search_documents(question, index, chunks, top_k=5):

    query_embedding = model.encode(
        [question],
        normalize_embeddings=True
    )

    query_embedding = query_embedding.astype("float32")

    k = min(top_k, index.ntotal)

    scores, indices = index.search(query_embedding, k)

    results = []

    for score, idx in zip(scores[0], indices[0]):

        if idx == -1:
            continue

        results.append({
            **chunks[idx],
            "similarity_score": float(score)
        })

    return results

# ---------------------------
# GEMINI ANSWER GENERATION
# ---------------------------

def generate_answer(question, retrieved_docs):

    context = ""

    for i, doc in enumerate(retrieved_docs):

        context += f"""
        Document {i+1}
        Source: {doc['source']}
        Page: {doc['page']}
        Content: {doc['text']}
        """

    prompt = f"""
    You are an AI document assistant.

    Answer using ONLY the provided context.

    Instructions:
    1. Do not invent facts.
    2. If the answer is unavailable, say so.
    3. Give simple and clear answers.
    4. Mention source filename and page number.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    response = client.models.generate_content(
        model="gemini-3.8-flash",
        contents=prompt
    )

    return response.text

# ---------------------------
# PDF UPLOAD
# ---------------------------

uploaded_file = st.file_uploader(
    "Upload your PDF",
    type=["pdf"]
)

if uploaded_file:

    if st.button("Process Document"):

        with st.spinner("Processing your document..."):

            chunks = process_pdf(uploaded_file)

            if not chunks:

                st.error("No readable text found in this PDF.")

            else:

                index = create_vector_store(chunks)

                st.session_state["chunks"] = chunks
                st.session_state["index"] = index
                st.session_state["filename"] = uploaded_file.name

                st.session_state.messages = []

                st.success(
                    f"Successfully processed {len(chunks)} chunks!"
                )

# ---------------------------
# CHAT INTERFACE
# ---------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if (
    "index" in st.session_state
    and "chunks" in st.session_state
):

    question = st.chat_input(
        "Ask a question about your PDF..."
    )

    if question:

        st.session_state.messages.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):

            with st.spinner("Finding relevant information..."):

                retrieved_docs = search_documents(
                    question,
                    st.session_state["index"],
                    st.session_state["chunks"]
                )

                answer = generate_answer(
                    question,
                    retrieved_docs
                )

                st.markdown(answer)

                with st.expander("View source documents"):

                    for doc in retrieved_docs:

                        st.write("Source:", doc["source"])
                        st.write("Page:", doc["page"])
                        st.write(
                            "Similarity:",
                            round(doc["similarity_score"], 4)
                        )
                        st.write(doc["text"])
                        st.divider()

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })

else:

    st.info("Please upload and process a PDF to start chatting.")
