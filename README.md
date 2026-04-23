# Hugging Face Chatbot with Agentic RAG

This project is a starter chatbot with a Streamlit UI that uses:

- a Hugging Face chat model for generation
- a sentence-transformer model for embeddings
- FAISS for vector search
- a lightweight agent loop that decides whether it should retrieve context before answering

## Features

- Ask direct questions without retrieval when the request is generic
- Retrieve supporting chunks when the question depends on your local documents
- Rewrite vague questions into better search queries
- Show the reasoning path taken by the agent
- Ingest `.txt`, `.md`, and `.pdf` files from a local `data/` folder

## Project Structure

```text
.
|-- app.py
|-- data/
|-- requirements.txt
|-- .env.example
`-- src/
    |-- chatbot.py
    |-- config.py
    |-- ingestion.py
    |-- llm.py
    |-- rag.py
    `-- vector_store.py
```

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Copy `.env.example` to `.env` and set:

- `HF_TOKEN`: your Hugging Face access token
- `HF_CHAT_MODEL`: any instruct/chat model available to you on Hugging Face
- `HF_EMBEDDING_MODEL`: embedding model for retrieval

4. Add your knowledge files to `data/`.

## Run

```bash
streamlit run app.py
```

Then open the local Streamlit URL shown in the terminal.

## How Agentic RAG Works Here

The chatbot follows a small agent loop:

1. Classify the user question as `direct`, `retrieve`, or `clarify`
2. If needed, rewrite the question into a stronger retrieval query
3. Search the vector store
4. Ask the model to answer using retrieved context
5. Return both the answer and the path the agent chose

This is intentionally simple and practical. If you want, we can extend it next with:

- web search tools
- SQL/document tools
- memory across sessions
- citation highlighting
- LangGraph or Haystack orchestration

## Example Questions

- `Summarize the architecture described in my PDFs`
- `What does the onboarding guide say about deployment?`
- `Explain agentic RAG in simple terms`
- `Compare the policy in document A with document B`
