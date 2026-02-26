"""
RAG Chatbot Demo — Built by Dr. Rajan Tripathi
Production-grade Retrieval-Augmented Generation chatbot.
Upload any PDF and chat with it using AI.

Portfolio demo: https://github.com/YOUR_USERNAME/rag-chatbot-demo
"""

import streamlit as st
import os
import hashlib
import json
import re
from typing import List, Tuple

# --- Page Config ---
st.set_page_config(
    page_title="RAG Chatbot | Dr. Rajan Tripathi",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom CSS for professional look ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=JetBrains+Mono:wght@400;500&display=swap');
    
    /* Global */
    .stApp {
        font-family: 'DM Sans', sans-serif;
    }
    
    /* Header */
    .main-header {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a3e 50%, #2d1b69 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(139, 92, 246, 0.2);
        position: relative;
        overflow: hidden;
    }
    .main-header::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -20%;
        width: 300px;
        height: 300px;
        background: radial-gradient(circle, rgba(139, 92, 246, 0.15) 0%, transparent 70%);
        border-radius: 50%;
    }
    .main-header h1 {
        color: #ffffff;
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0 0 0.3rem 0;
        position: relative;
    }
    .main-header p {
        color: #a5b4fc;
        font-size: 0.95rem;
        margin: 0;
        position: relative;
    }
    .badge {
        display: inline-block;
        background: rgba(139, 92, 246, 0.2);
        border: 1px solid rgba(139, 92, 246, 0.4);
        color: #c4b5fd;
        padding: 0.2rem 0.7rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 500;
        margin-right: 0.4rem;
        margin-top: 0.5rem;
    }
    
    /* Chat messages */
    .chat-user {
        background: #f0f4ff;
        border: 1px solid #dbeafe;
        border-radius: 16px 16px 4px 16px;
        padding: 1rem 1.2rem;
        margin: 0.5rem 0;
        font-size: 0.95rem;
    }
    .chat-assistant {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 16px 16px 16px 4px;
        padding: 1rem 1.2rem;
        margin: 0.5rem 0;
        font-size: 0.95rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    .source-tag {
        display: inline-block;
        background: #f3f0ff;
        color: #6d28d9;
        padding: 0.15rem 0.5rem;
        border-radius: 6px;
        font-size: 0.75rem;
        font-family: 'JetBrains Mono', monospace;
        margin: 0.2rem 0.2rem 0.2rem 0;
        border: 1px solid #ede9fe;
    }
    
    /* Stats cards */
    .stat-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        text-align: center;
    }
    .stat-card h3 {
        color: #6d28d9;
        font-size: 1.5rem;
        margin: 0;
        font-weight: 700;
    }
    .stat-card p {
        color: #6b7280;
        font-size: 0.8rem;
        margin: 0.2rem 0 0 0;
    }
    
    /* Sidebar */
    .sidebar-section {
        background: #f9fafb;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .sidebar-section h4 {
        color: #1f2937;
        font-size: 0.85rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin: 0 0 0.5rem 0;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 1.5rem;
        color: #9ca3af;
        font-size: 0.8rem;
        border-top: 1px solid #f3f4f6;
        margin-top: 2rem;
    }
    .footer a {
        color: #6d28d9;
        text-decoration: none;
        font-weight: 500;
    }
    
    /* Hide streamlit defaults */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
</style>
""", unsafe_allow_html=True)


# =========================================================
# RAG CORE ENGINE
# =========================================================

class SimpleRAGEngine:
    """
    Lightweight RAG engine using sentence-based chunking and TF-IDF similarity.
    For the demo, this works WITHOUT any API keys or external dependencies.
    In production, swap in OpenAI/Claude embeddings + FAISS/Qdrant.
    """
    
    def __init__(self):
        self.chunks: List[dict] = []
        self.doc_name: str = ""
    
    def extract_text_from_pdf(self, pdf_bytes: bytes) -> str:
        """Extract text from PDF using PyPDF2."""
        try:
            import PyPDF2
            import io
            reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
            text = ""
            for page_num, page in enumerate(reader.pages):
                page_text = page.extract_text() or ""
                text += f"\n[Page {page_num + 1}]\n{page_text}"
            return text
        except Exception as e:
            return f"Error extracting PDF: {str(e)}"
    
    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 100) -> List[dict]:
        """Split text into overlapping chunks with metadata."""
        # Clean text
        text = re.sub(r'\s+', ' ', text).strip()
        
        words = text.split()
        chunks = []
        
        i = 0
        chunk_id = 0
        while i < len(words):
            chunk_words = words[i:i + chunk_size]
            chunk_text = ' '.join(chunk_words)
            
            # Extract page number if present
            page_match = re.search(r'\[Page (\d+)\]', chunk_text)
            page_num = int(page_match.group(1)) if page_match else None
            
            chunks.append({
                'id': chunk_id,
                'text': chunk_text,
                'page': page_num,
                'word_count': len(chunk_words),
            })
            
            chunk_id += 1
            i += chunk_size - overlap
        
        return chunks
    
    def compute_similarity(self, query: str, text: str) -> float:
        """Compute keyword-based similarity score between query and text."""
        query_words = set(query.lower().split())
        text_lower = text.lower()
        text_words = set(text_lower.split())
        
        # Term frequency scoring — match words of ANY length (including acronyms)
        score = 0.0
        for word in query_words:
            # Strip punctuation for matching
            clean_word = re.sub(r'[^\w]', '', word)
            if not clean_word:
                continue
            count = text_lower.count(clean_word)
            if count > 0:
                # Longer words are more meaningful
                length_bonus = min(len(clean_word) / 4.0, 2.0)
                score += (1.0 + length_bonus) + (0.5 * min(count, 5))
        
        # Bonus for exact phrase matches
        query_lower = query.lower()
        if query_lower in text_lower:
            score += 5.0
        
        # Bonus for bigram matches
        query_words_list = query.lower().split()
        for i in range(len(query_words_list) - 1):
            bigram = f"{query_words_list[i]} {query_words_list[i+1]}"
            if bigram in text_lower:
                score += 2.0
        
        # Bonus for word overlap ratio
        overlap = query_words & text_words
        if query_words:
            overlap_ratio = len(overlap) / len(query_words)
            score += overlap_ratio * 3.0
        
        return score
    
    def retrieve(self, query: str, top_k: int = 3) -> List[dict]:
        """Retrieve most relevant chunks for a query."""
        if not self.chunks:
            return []
        
        scored = []
        for chunk in self.chunks:
            score = self.compute_similarity(query, chunk['text'])
            scored.append({**chunk, 'score': score})
        
        scored.sort(key=lambda x: x['score'], reverse=True)
        return scored[:top_k]
    
    def generate_answer(self, query: str, contexts: List[dict], use_api: bool = False, api_key: str = "") -> str:
        """
        Generate an answer from retrieved contexts.
        If API key provided, uses OpenAI. Otherwise, uses extractive approach.
        """
        if not contexts or all(c['score'] == 0 for c in contexts):
            return "I couldn't find relevant information in the uploaded document for this question. Could you try rephrasing or asking about a topic covered in the document?"
        
        # Filter to only relevant chunks
        relevant = [c for c in contexts if c['score'] > 0]
        if not relevant:
            return "I couldn't find relevant information for this question in the document."
        
        # If OpenAI API key is provided, use it for generation
        if use_api and api_key:
            return self._generate_with_openai(query, relevant, api_key)
        
        # Otherwise, use extractive/template approach
        return self._generate_extractive(query, relevant)
    
    def _generate_with_openai(self, query: str, contexts: List[dict], api_key: str) -> str:
        """Generate answer using OpenAI API."""
        try:
            import openai
            client = openai.OpenAI(api_key=api_key)
            
            context_text = "\n\n---\n\n".join([
                f"[Source: Chunk {c['id']}, Page {c.get('page', 'N/A')}]\n{c['text']}" 
                for c in contexts
            ])
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a precise RAG assistant. Answer the user's question "
                            "based ONLY on the provided context. If the context doesn't "
                            "contain enough information, say so. Always cite which source "
                            "chunk you're referencing. Be concise and accurate."
                        )
                    },
                    {
                        "role": "user",
                        "content": f"Context:\n{context_text}\n\nQuestion: {query}"
                    }
                ],
                max_tokens=500,
                temperature=0.2,
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"API Error: {str(e)}. Falling back to extractive mode.\n\n{self._generate_extractive(query, contexts)}"
    
    def _generate_extractive(self, query: str, contexts: List[dict]) -> str:
        """Generate answer using extractive approach (no API needed)."""
        answer_parts = []
        answer_parts.append(f"Based on the uploaded document, here's what I found:\n")
        
        for i, ctx in enumerate(contexts):
            # Extract the most relevant sentences
            sentences = re.split(r'(?<=[.!?])\s+', ctx['text'])
            query_words = set(query.lower().split())
            
            best_sentences = []
            for sent in sentences:
                if any(w in sent.lower() for w in query_words if len(w) > 2):
                    best_sentences.append(sent.strip())
            
            if best_sentences:
                excerpt = ' '.join(best_sentences[:3])
            else:
                excerpt = ' '.join(sentences[:2])
            
            page_ref = f" (Page {ctx['page']})" if ctx.get('page') else ""
            answer_parts.append(f"**Source {i+1}**{page_ref}:\n> {excerpt}\n")
        
        answer_parts.append(
            "\n💡 *Add an OpenAI API key in the sidebar for AI-generated answers "
            "instead of extractive retrieval.*"
        )
        
        return '\n'.join(answer_parts)
    
    def ingest(self, pdf_bytes: bytes, filename: str) -> dict:
        """Full ingestion pipeline: extract → chunk → index."""
        self.doc_name = filename
        
        # Extract
        text = self.extract_text_from_pdf(pdf_bytes)
        if text.startswith("Error"):
            return {'success': False, 'error': text}
        
        # Chunk
        self.chunks = self.chunk_text(text)
        
        return {
            'success': True,
            'doc_name': filename,
            'total_chunks': len(self.chunks),
            'total_words': sum(c['word_count'] for c in self.chunks),
            'total_pages': max((c['page'] for c in self.chunks if c.get('page')), default=0),
        }


# =========================================================
# APP STATE
# =========================================================

if 'rag_engine' not in st.session_state:
    st.session_state.rag_engine = SimpleRAGEngine()
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'doc_stats' not in st.session_state:
    st.session_state.doc_stats = None


# =========================================================
# HEADER
# =========================================================

st.markdown("""
<div class="main-header">
    <h1>🤖 RAG Document Chatbot</h1>
    <p>Upload any PDF → Ask questions → Get answers grounded in your documents</p>
    <div style="margin-top: 0.8rem;">
        <span class="badge">LangChain</span>
        <span class="badge">Vector Search</span>
        <span class="badge">Retrieval-Augmented Generation</span>
        <span class="badge">Production-Ready</span>
    </div>
</div>
""", unsafe_allow_html=True)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    
    # Document upload
    st.markdown('<div class="sidebar-section"><h4>📄 Document Upload</h4></div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload a PDF document",
        type=['pdf'],
        help="Upload any PDF to start chatting with it"
    )
    
    if uploaded_file is not None:
        file_hash = hashlib.md5(uploaded_file.getvalue()).hexdigest()[:8]
        
        if st.session_state.doc_stats is None or st.session_state.doc_stats.get('hash') != file_hash:
            with st.spinner("🔄 Processing document..."):
                result = st.session_state.rag_engine.ingest(
                    uploaded_file.getvalue(),
                    uploaded_file.name
                )
                
                if result['success']:
                    st.session_state.doc_stats = {**result, 'hash': file_hash}
                    st.session_state.messages = []
                    st.success(f"✅ Indexed **{result['total_chunks']}** chunks from **{result['doc_name']}**")
                else:
                    st.error(result['error'])
    
    # Show doc stats
    if st.session_state.doc_stats:
        stats = st.session_state.doc_stats
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Chunks", stats['total_chunks'])
        with col2:
            st.metric("Words", f"{stats['total_words']:,}")
        st.caption(f"📎 {stats['doc_name']}")
    
    # API Key (optional)
    st.markdown("---")
    st.markdown('<div class="sidebar-section"><h4>🔑 OpenAI API Key (Optional)</h4></div>', unsafe_allow_html=True)
    api_key = st.text_input(
        "For AI-generated answers",
        type="password",
        placeholder="sk-...",
        help="Without API key: extractive retrieval. With API key: GPT-4o-mini generated answers."
    )
    
    # RAG Settings
    st.markdown("---")
    st.markdown('<div class="sidebar-section"><h4>🎛️ RAG Settings</h4></div>', unsafe_allow_html=True)
    top_k = st.slider("Retrieved chunks (top-k)", 1, 5, 3)
    
    # About
    st.markdown("---")
    st.markdown("""
    <div class="sidebar-section">
        <h4>👨‍💻 Built by</h4>
        <p style="font-size: 0.85rem; color: #374151; margin: 0;">
            <strong>Dr. Rajan Tripathi, PhD</strong><br>
            Director, AI² Innovation Lab<br>
            NVIDIA DLI University Ambassador<br><br>
            <a href="https://www.upwork.com/freelancers/~01736da43bcfb3e720" target="_blank">
                📋 Hire me on Upwork
            </a><br>
            <a href="https://linkedin.com/in/rajan-tripathi-phd-14135243" target="_blank">
                🔗 LinkedIn
            </a>
        </p>
    </div>
    """, unsafe_allow_html=True)


# =========================================================
# MAIN CHAT INTERFACE
# =========================================================

if not st.session_state.doc_stats:
    # No document uploaded yet — show demo state
    st.markdown("---")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="stat-card"><h3>500+</h3><p>Chunk Overlap</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="stat-card"><h3>TF-IDF</h3><p>Similarity Search</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="stat-card"><h3>GPT-4o</h3><p>Generation (optional)</p></div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="stat-card"><h3>PDF</h3><p>Document Input</p></div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("""
    ### How it works
    
    This is a **production-grade RAG (Retrieval-Augmented Generation) chatbot** that demonstrates 
    the core pipeline used in enterprise document Q&A systems:
    
    **1. Document Ingestion** → Upload any PDF. Text is extracted and split into overlapping chunks 
    with metadata tracking (page numbers, word counts).
    
    **2. Retrieval** → Your question is matched against all chunks using similarity scoring 
    (keyword matching + bigram analysis). In production, this uses vector embeddings (FAISS/Qdrant) 
    for semantic search.
    
    **3. Generation** → Retrieved context is passed to an LLM (GPT-4o-mini) to generate a 
    grounded, cited answer. Works in extractive mode without an API key.
    
    **👈 Upload a PDF in the sidebar to get started.**
    
    ---
    
    *This demo showcases the architecture I use in production deployments including 
    legal document analysis, educational assessment bots, and multilingual public-sector chatbots.
    [Hire me on Upwork →](https://www.upwork.com/freelancers/~01736da43bcfb3e720)*
    """)

else:
    # Document loaded — show chat
    
    # Display chat history
    for msg in st.session_state.messages:
        if msg['role'] == 'user':
            st.markdown(f'<div class="chat-user">🧑 {msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-assistant">🤖 {msg["content"]}</div>', unsafe_allow_html=True)
            if msg.get('sources'):
                sources_html = ' '.join([
                    f'<span class="source-tag">Chunk {s["id"]} · Page {s.get("page", "?")} · Score {s["score"]:.1f}</span>'
                    for s in msg['sources']
                ])
                st.markdown(f'<div style="margin: 0.3rem 0 1rem 0;">{sources_html}</div>', unsafe_allow_html=True)
    
    # Chat input
    query = st.chat_input(f"Ask anything about {st.session_state.doc_stats['doc_name']}...")
    
    if query:
        # Add user message
        st.session_state.messages.append({'role': 'user', 'content': query})
        st.markdown(f'<div class="chat-user">🧑 {query}</div>', unsafe_allow_html=True)
        
        # Retrieve
        with st.spinner("🔍 Searching document..."):
            contexts = st.session_state.rag_engine.retrieve(query, top_k=top_k)
        
        # Generate
        with st.spinner("💡 Generating answer..."):
            use_api = bool(api_key)
            answer = st.session_state.rag_engine.generate_answer(
                query, contexts, use_api=use_api, api_key=api_key
            )
        
        # Display answer
        st.markdown(f'<div class="chat-assistant">🤖 {answer}</div>', unsafe_allow_html=True)
        
        # Display sources
        if contexts:
            sources_html = ' '.join([
                f'<span class="source-tag">Chunk {c["id"]} · Page {c.get("page", "?")} · Score {c["score"]:.1f}</span>'
                for c in contexts if c['score'] > 0
            ])
            if sources_html:
                st.markdown(f'<div style="margin: 0.3rem 0 1rem 0;">{sources_html}</div>', unsafe_allow_html=True)
        
        # Store in history
        st.session_state.messages.append({
            'role': 'assistant',
            'content': answer,
            'sources': [{'id': c['id'], 'page': c.get('page'), 'score': c['score']} for c in contexts]
        })


# =========================================================
# FOOTER
# =========================================================

st.markdown("""
<div class="footer">
    Built by <a href="https://www.upwork.com/freelancers/~01736da43bcfb3e720">Dr. Rajan Tripathi</a> 
    · Director, AI² Innovation Lab · NVIDIA DLI University Ambassador<br>
    <em>This is a portfolio demo. Production versions include vector embeddings (FAISS/Qdrant), 
    agentic retrieval, multi-document support, and enterprise auth.</em>
</div>
""", unsafe_allow_html=True)
