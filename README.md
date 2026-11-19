# RAG-Based AI Document Assistant

An AI-powered document question-answering application that allows users to interact with PDF documents and receive context-aware answers using Retrieval-Augmented Generation (RAG).

## Live Demo

[Try the Live Application](https://rag-assistant-5safvzxkhmxhfo7tha8og6.streamlit.app/)

## Overview

The RAG-Based AI Document Assistant uses document retrieval and Google's Gemini API to answer questions based on the content of uploaded or indexed documents.

Instead of relying only on the language model's general knowledge, the application retrieves relevant document chunks and uses them as context to generate answers.

## Features

- PDF document processing
- Text extraction and chunking
- Vector similarity search using FAISS
- Retrieval-Augmented Generation (RAG)
- Context-aware question answering
- Google Gemini API integration
- Interactive Streamlit interface

## Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming |
| Streamlit | User interface |
| Google Gemini API | Answer generation |
| FAISS | Vector similarity search |
| Pickle | Storing processed chunks |
| RAG | Document-based question answering |

## Project Structure

```text
RAG-Based-AI-Document-Assistant/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── document_index.faiss
└── chunks.pkl
```

## How It Works

1. **Document Processing:** Extracts text from PDF documents.
2. **Text Chunking:** Divides the extracted text into smaller chunks.
3. **Vector Indexing:** Stores document representations for similarity search using FAISS.
4. **Retrieval:** Retrieves relevant chunks based on the user's question.
5. **Answer Generation:** Sends the retrieved context to Google Gemini to generate a relevant response.
6. **User Interface:** Displays the answers through a Streamlit application.

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/chhavibakhru1-spec/rag-assistant.git
cd rag-assistant
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it:

Windows:
```bash
venv\Scripts\activate
```

Linux/macOS:
```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Gemini API Key

Set your Gemini API key as an environment variable.

Linux/macOS:
```bash
export GEMINI_API_KEY="your_api_key"
```

Windows PowerShell:
```powershell
$env:GEMINI_API_KEY="your_api_key"
```

Never commit API keys or secrets to GitHub.

### 5. Run the application

```bash
streamlit run app.py
```

## Environment Variables

| Variable | Description |
|---|---|
| GEMINI_API_KEY | Google Gemini API key |

## Future Improvements

- Support for multiple document uploads
- Conversation history
- Improved document retrieval and ranking
- Source citations for generated answers
- Support for additional document formats

## Author

**Chhavi Bakhru**

GitHub: https://github.com/chhavibakhru1-spec

---

*Built as an AI/ML project exploring Retrieval-Augmented Generation, vector databases, and LLM-powered document question answering.*
