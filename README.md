# 🤖 RAG Document Chatbot

**Production-grade Retrieval-Augmented Generation chatbot** — Upload any PDF, ask questions, get answers grounded in your documents.

Built by [Dr. Rajan Tripathi](https://www.upwork.com/freelancers/~01736da43bcfb3e720) | Director, AI² Innovation Lab | NVIDIA DLI University Ambassador

## Features

- **PDF Ingestion** — Extract text, split into overlapping chunks with metadata
- **Similarity Retrieval** — TF-IDF keyword + bigram matching (swap in FAISS/Qdrant for production)
- **LLM Generation** — Optional OpenAI GPT-4o-mini for AI-generated answers
- **Source Attribution** — Every answer cites chunk IDs, page numbers, and relevance scores
- **Zero-config mode** — Works without any API keys (extractive retrieval)

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Cloud (Free)

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set `app.py` as the main file
5. Deploy — live in ~2 minutes

## Architecture

```
PDF Upload → Text Extraction → Chunking (500 words, 100 overlap)
                                    ↓
User Query → Similarity Search → Top-K Retrieval → LLM Generation → Cited Answer
```

## Production Upgrades Available

This demo showcases the core RAG pipeline. Production versions I deploy for clients include:

- **Vector Embeddings** — OpenAI/Cohere embeddings + FAISS/Qdrant/Pinecone
- **Agentic Retrieval** — Multi-step reasoning with LangChain/LangGraph
- **Multi-document Support** — Ingest hundreds of documents with metadata filtering
- **Multilingual** — English, Russian, Uzbek, Hindi support
- **Enterprise Auth** — SSO, role-based access, audit logging
- **Conversation Memory** — Multi-turn context with summarization

## Hire Me

Looking for a production RAG chatbot or AI agent for your business?

- 📋 [Upwork](https://www.upwork.com/freelancers/~01736da43bcfb3e720)
- 🔗 [LinkedIn](https://linkedin.com/in/rajan-tripathi-phd-14135243)

## License

MIT — Free to use and modify.
