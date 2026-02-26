# ⚡ RAG Document Chatbot — Production Demo

**Upload any PDF → Ask questions → Get cited answers grounded in your documents.**

Built by [Dr. Rajan Tripathi](https://www.upwork.com/freelancers/~01736da43bcfb3e720) | Director, AI² Innovation Lab | NVIDIA DLI University Ambassador

## ✨ Features

- **Real Semantic Search** — sentence-transformers (MiniLM-L6) embeddings with cosine similarity
- **Pre-loaded Sample** — Works instantly on visit (AI in Healthcare document)
- **PDF Upload** — Upload any PDF and chat with it
- **Source Attribution** — Every answer shows page numbers and confidence scores
- **Multi-provider LLM** — Optional OpenAI GPT-4o-mini or Claude Sonnet for AI-generated answers
- **Smart Fallback** — Extractive retrieval works without any API key

## 🚀 Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 🏗️ Architecture

```
PDF Upload → Text Extraction → Smart Chunking (paragraph-aware, 300 words, 75 overlap)
                                         ↓
                              Sentence-Transformer Embeddings (MiniLM-L6-v2)
                                         ↓
User Query → Query Embedding → Cosine Similarity → Top-K Retrieval → LLM Generation → Cited Answer
```

## 🔧 Production Upgrades

This demo shows the core RAG architecture. My production deployments for clients add:

| Feature | Demo | Production |
|---------|------|------------|
| Embeddings | MiniLM-L6 | OpenAI/Cohere + FAISS/Qdrant |
| Retrieval | Single-step | Agentic multi-hop (LangGraph) |
| Documents | Single PDF | 1000s of docs with metadata filtering |
| Languages | English | EN/RU/UZ/HI multilingual |
| Auth | None | SSO + RBAC + audit logging |
| Hosting | Streamlit Cloud | Docker + FastAPI + Cloud Run |

## 📋 Hire Me

- [Upwork](https://www.upwork.com/freelancers/~01736da43bcfb3e720)
- [LinkedIn](https://linkedin.com/in/rajan-tripathi-phd-14135243)
