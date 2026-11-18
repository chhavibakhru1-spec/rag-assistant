
import os
import re
import time
import unicodedata

import streamlit as st
import fitz
import faiss
import numpy as np

from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import errors


# --------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------

st.set_page_config(
    page_title="AI Document Assistant",
    page_icon="📚",
    layout="wide"
)

st.title("📚 AI Document Assistant")
st.write("Upload your PDF and ask questions about it.")


# --------------------------------
# 2. GEMINI API CONFIGURATION
# --------------------------------

try:
    api_key = st.secrets["GEMINI_API_KEY"]
except Exception:
    api_key = os.environ.get("GEMINI_API_KEY")

if not api_key:
    st.error(
        "Gemini API key is missing. "
        "Please add GEMINI_API_KEY in Streamlit Secrets."
    )
    st.stop()

client = genai.Client(api_key=api_key)

MODEL_NAME = "gemini-3.8-flash"


# --------------------------------
# 3. LOAD EMBEDDING MODEL
# --------------------------------

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )


with st.spinner("Loading embedding model..."):
    embedding_model = load_embedding_model()


# --------------------------------
# 4. TEXT CLEANING
# --------------------------------

def clean_text(text):

    text = unicodedata.normalize("NFKC", text)

    text = text.replace("\x00", "")

    text = re.sub(r"[ \t]+", " ", text)

    text = re.sub(r"\n+", "\n", text)

    return text.strip()


# --------------------------------
# 5. PDF EXTRACTION AND CHUNKING
# --------------------------------

def process_pdf(pdf_file, chunk_size=500, overlap=100):

    chunks = []

    pdf_bytes = pdf_file.getvalue()

    with fitz.open(
        stream=pdf_bytes,
        filetype="pdf"
    ) as pdf:

        for page_number, page in enumerate(pdf):

            text = page.get_text("text")

            text = clean_text(text)

            if not text:
                continue

            words = text.split()

            start = 0

            while start < len(words):

                end = start + chunk_size

                chunk_text = " ".join(words[start:end])

                if chunk_text.strip():

                    chunks.append({
                        "text": chunk_text,
                        "page": page_number + 1,
                        "source": pdf_file.name
                    })

                if end >= len(words):
                    break

                start += chunk_size - overlap

    return chunks


# --------------------------------
# 6. CREATE FAISS VECTOR STORE
# --------------------------------

def create_vector_store(chunks):

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = embedding_model.encode(
        texts,
        batch_size=16,
        show_progress_bar=False,
        normalize_embeddings=True
    )

    embeddings = np.asarray(
        embeddings,
        dtype="float32"
    )

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    return index


# --------------------------------
# 7. RETRIEVE RELEVANT DOCUMENTS
# --------------------------------

def search_documents(question, index, chunks, top_k=5):

    query_embedding = embedding_model.encode(
        [question],
        normalize_embeddings=True
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype="float32"
    )

    k = min(top_k, len(chunks))

    scores, indices = index.search(
        query_embedding,
        k
    )

    retrieved_docs = []

    for score, idx in zip(scores[0], indices[0]):

        if idx < 0:
            continue

        doc = chunks[idx].copy()

        doc["similarity_score"] = float(score)

        retrieved_docs.append(doc)

    return retrieved_docs


# --------------------------------
# 8. GENERATE ANSWER USING GEMINI
# --------------------------------

def generate_answer(question, retrieved_docs):

    context = "\n\n".join([

        f"Source: {doc['source']}\n"
        f"Page: {doc['page']}\n"
        f"Content: {doc['text']}"

        for doc in retrieved_docs
    ])

    prompt = f"""
You are an AI Document Assistant.

Answer the user's question using the provided document context.

Instructions:
1. Use only the information in the provided context.
2. Give clear and accurate answers.
3. Explain difficult concepts in simple language.
4. Do not make up information.
5. If the answer is not available in the context,
   clearly say that the document does not provide the answer.
6. Mention relevant page numbers whenever possible.

DOCUMENT CONTEXT:

{context}

USER QUESTION:

{question}

ANSWER:
"""

    # Retry temporary Gemini API failures

    for attempt in range(4):

        try:

            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt
            )

            if response.text:
                return response.text

            return "Gemini returned an empty response. Please try again."

        except errors.APIError as e:

            status_code = getattr(e, "code", None)

            if status_code in [429, 500, 502, 503, 504]:

                if attempt < 3:

                    wait_time = 2 ** (attempt + 1)

                    time.sleep(wait_time)

                    continue

                return (
                    "Gemini is temporarily unavailable or "
                    "the API quota has been reached. "
                    "Please try again later."
                )

            return (
                f"Gemini API error ({status_code}). "
                "Please check your API configuration and quota."
            )

        except Exception as e:

            return (
                "An unexpected error occurred while generating "
                f"the answer: {str(e)}"
            )


# --------------------------------
# 9. SESSION STATE
# --------------------------------

if "chunks" not in st.session_state:
    st.session_state.chunks = None

if "index" not in st.session_state:
    st.session_state.index = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "processed_filename" not in st.session_state:
    st.session_state.processed_filename = None


# --------------------------------
# 10. PDF UPLOAD
# --------------------------------

uploaded_file = st.file_uploader(
    "Upload your PDF",
    type=["pdf"]
)


if uploaded_file is not None:

    if st.button("Process Document"):

        with st.spinner(
            "Extracting text and creating embeddings..."
        ):

            try:

                chunks = process_pdf(uploaded_file)

                if not chunks:

                    st.error(
                        "No readable text was found in this PDF."
                    )

                else:

                    index = create_vector_store(chunks)

                    st.session_state.chunks = chunks

                    st.session_state.index = index

                    st.session_state.messages = []

                    st.session_state.processed_filename = (
                        uploaded_file.name
                    )

                    st.success(
                        f"Successfully processed {uploaded_file.name}"
                    )

                    st.info(
                        f"Created {len(chunks)} text chunks."
                    )

            except Exception as e:

                st.error(
                    f"Document processing failed: {str(e)}"
                )


# --------------------------------
# 11. CHAT INTERFACE
# --------------------------------

if (
    st.session_state.index is not None
    and st.session_state.chunks is not None
):

    st.subheader("Ask Questions About Your PDF")

    # Display previous conversation

    for message in st.session_state.messages:

        with st.chat_message(message["role"]):

            st.markdown(message["content"])


    question = st.chat_input(
        "Ask a question about your document..."
    )


    if question:

        # Display user question

        st.session_state.messages.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user"):

            st.markdown(question)


        # Retrieve relevant documents

        with st.spinner("Searching relevant document sections..."):

            retrieved_docs = search_documents(
                question,
                st.session_state.index,
                st.session_state.chunks,
                top_k=5
            )


        # Generate answer

        with st.chat_message("assistant"):

            with st.spinner("Generating answer..."):

                answer = generate_answer(
                    question,
                    retrieved_docs
                )

            st.markdown(answer)

            # Display source documents

            with st.expander("View source documents"):

                for doc in retrieved_docs:

                    st.write(
                        "Source:",
                        doc["source"]
                    )

                    st.write(
                        "Page:",
                        doc["page"]
                    )

                    st.write(
                        "Similarity:",
                        round(doc["similarity_score"], 4)
                    )

                    st.write(doc["text"])

                    st.divider()


        # Save assistant response

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })


else:

    st.info(
        "Please upload and process a PDF to start chatting."
    )

