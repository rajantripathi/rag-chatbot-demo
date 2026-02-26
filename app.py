"""
RAG Document Chatbot — Production Demo
Built by Dr. Rajan Tripathi | AI² Innovation Lab | NVIDIA DLI Ambassador

Features:
- Real semantic search using sentence-transformers embeddings
- Pre-loaded sample document (no friction for visitors)
- Beautiful, professional UI
- Optional OpenAI/Claude API integration
- Source attribution with confidence scores
"""

import streamlit as st
import os
import hashlib
import json
import re
import numpy as np
from typing import List, Tuple, Optional
import time

# --- Page Config ---
st.set_page_config(
    page_title="RAG Chatbot | Dr. Rajan Tripathi",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Professional CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
    
    /* Reset & Global */
    .stApp {
        font-family: 'Plus Jakarta Sans', sans-serif;
        background: #fafbfc;
    }
    
    /* Hero Header */
    .hero {
        background: linear-gradient(145deg, #0c0a1a 0%, #1a103a 40%, #2d1a5e 70%, #4c1d95 100%);
        padding: 2.5rem 2.5rem 2rem;
        border-radius: 20px;
        margin-bottom: 1.5rem;
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(139, 92, 246, 0.15);
    }
    .hero::before {
        content: '';
        position: absolute;
        top: -40%;
        right: -15%;
        width: 400px;
        height: 400px;
        background: radial-gradient(circle, rgba(167, 139, 250, 0.12) 0%, transparent 70%);
        border-radius: 50%;
    }
    .hero::after {
        content: '';
        position: absolute;
        bottom: -30%;
        left: 10%;
        width: 250px;
        height: 250px;
        background: radial-gradient(circle, rgba(79, 70, 229, 0.1) 0%, transparent 70%);
        border-radius: 50%;
    }
    .hero-title {
        color: #ffffff;
        font-size: 2rem;
        font-weight: 800;
        margin: 0 0 0.4rem 0;
        position: relative;
        letter-spacing: -0.02em;
    }
    .hero-subtitle {
        color: #c4b5fd;
        font-size: 1rem;
        margin: 0 0 1rem 0;
        position: relative;
        font-weight: 400;
        line-height: 1.5;
    }
    .hero-badges {
        position: relative;
        display: flex;
        flex-wrap: wrap;
        gap: 0.4rem;
    }
    .badge {
        display: inline-flex;
        align-items: center;
        gap: 0.3rem;
        background: rgba(139, 92, 246, 0.15);
        border: 1px solid rgba(139, 92, 246, 0.3);
        color: #ddd6fe;
        padding: 0.3rem 0.75rem;
        border-radius: 100px;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.02em;
        text-transform: uppercase;
    }
    .badge-green {
        background: rgba(34, 197, 94, 0.15);
        border-color: rgba(34, 197, 94, 0.3);
        color: #86efac;
    }
    
    /* Pipeline visualization */
    .pipeline {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 0.5rem;
    }
    .pipeline-step {
        text-align: center;
        flex: 1;
    }
    .pipeline-icon {
        width: 48px;
        height: 48px;
        background: linear-gradient(135deg, #f5f3ff, #ede9fe);
        border-radius: 14px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 1.3rem;
        margin-bottom: 0.4rem;
        border: 1px solid #e9e5f5;
    }
    .pipeline-label {
        font-size: 0.72rem;
        font-weight: 700;
        color: #4c1d95;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .pipeline-desc {
        font-size: 0.7rem;
        color: #6b7280;
        margin-top: 0.15rem;
    }
    .pipeline-arrow {
        color: #c4b5fd;
        font-size: 1.2rem;
        flex-shrink: 0;
    }
    
    /* Chat messages */
    .msg-user {
        background: linear-gradient(135deg, #4c1d95, #6d28d9);
        color: #ffffff;
        border-radius: 20px 20px 6px 20px;
        padding: 1rem 1.3rem;
        margin: 0.6rem 0;
        margin-left: 15%;
        font-size: 0.92rem;
        line-height: 1.5;
        box-shadow: 0 2px 8px rgba(109, 40, 217, 0.2);
    }
    .msg-bot {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 20px 20px 20px 6px;
        padding: 1.2rem 1.5rem;
        margin: 0.6rem 0;
        margin-right: 10%;
        font-size: 0.92rem;
        line-height: 1.65;
        box-shadow: 0 1px 4px rgba(0,0,0,0.04);
    }
    .msg-bot strong {
        color: #4c1d95;
    }
    
    /* Source chips */
    .sources-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.35rem;
        margin: 0.5rem 0 0.8rem 0;
    }
    .source-chip {
        display: inline-flex;
        align-items: center;
        gap: 0.25rem;
        background: #f5f3ff;
        color: #5b21b6;
        padding: 0.25rem 0.65rem;
        border-radius: 8px;
        font-size: 0.7rem;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 500;
        border: 1px solid #ede9fe;
    }
    .source-chip .score {
        background: #4c1d95;
        color: white;
        padding: 0.1rem 0.35rem;
        border-radius: 4px;
        font-size: 0.62rem;
    }
    
    /* Stats */
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 0.8rem;
        margin: 1rem 0;
    }
    .stat-box {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 1.1rem;
        text-align: center;
        transition: all 0.2s;
    }
    .stat-box:hover {
        border-color: #c4b5fd;
        box-shadow: 0 4px 12px rgba(109, 40, 217, 0.08);
    }
    .stat-number {
        font-size: 1.6rem;
        font-weight: 800;
        color: #4c1d95;
        line-height: 1;
    }
    .stat-label {
        font-size: 0.7rem;
        color: #6b7280;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-top: 0.3rem;
    }
    
    /* Sidebar */
    .sidebar-card {
        background: #f9fafb;
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 1.1rem;
        margin-bottom: 0.8rem;
    }
    .sidebar-card h4 {
        font-size: 0.7rem;
        font-weight: 800;
        color: #4c1d95;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin: 0 0 0.6rem 0;
    }
    
    /* CTA button */
    .cta-box {
        background: linear-gradient(135deg, #4c1d95, #7c3aed);
        border-radius: 14px;
        padding: 1.2rem;
        text-align: center;
        margin-top: 1rem;
    }
    .cta-box a {
        color: white;
        text-decoration: none;
        font-weight: 700;
        font-size: 0.9rem;
    }
    .cta-box p {
        color: #ddd6fe;
        font-size: 0.75rem;
        margin: 0.3rem 0 0 0;
    }
    
    /* Sample Q buttons */
    .sample-q {
        display: inline-block;
        background: #f5f3ff;
        border: 1px solid #ede9fe;
        color: #5b21b6;
        padding: 0.5rem 1rem;
        border-radius: 100px;
        font-size: 0.82rem;
        font-weight: 500;
        margin: 0.25rem;
        cursor: pointer;
        transition: all 0.2s;
    }
    .sample-q:hover {
        background: #ede9fe;
        border-color: #c4b5fd;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem 1rem;
        color: #9ca3af;
        font-size: 0.78rem;
        border-top: 1px solid #f3f4f6;
        margin-top: 2rem;
    }
    .footer a { color: #6d28d9; text-decoration: none; font-weight: 600; }
    
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(8px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .msg-bot, .msg-user {
        animation: fadeIn 0.3s ease-out;
    }
    
    /* Hide defaults */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
    header[data-testid="stHeader"] {background: transparent;}
</style>
""", unsafe_allow_html=True)


# =========================================================
# SAMPLE DOCUMENT (Pre-loaded so visitors see it working)
# =========================================================

SAMPLE_DOCUMENT = """
[Page 1]
ARTIFICIAL INTELLIGENCE IN HEALTHCARE: A COMPREHENSIVE OVERVIEW

1. INTRODUCTION

Artificial Intelligence (AI) is revolutionizing healthcare by enabling faster diagnosis, personalized treatment plans, and improved patient outcomes. The global AI in healthcare market was valued at $20.9 billion in 2024 and is projected to reach $148.4 billion by 2029, growing at a CAGR of 48.1%. This rapid growth reflects the increasing adoption of AI technologies across clinical, administrative, and research applications in hospitals, clinics, and pharmaceutical companies worldwide.

Healthcare AI encompasses a broad range of technologies including machine learning, natural language processing, computer vision, robotic process automation, and predictive analytics. These technologies are being applied to medical imaging analysis, drug discovery, clinical decision support, patient monitoring, and administrative workflow optimization.

2. MEDICAL IMAGING AND DIAGNOSTICS

AI-powered diagnostic tools have shown remarkable accuracy in detecting diseases from medical images. In radiology, deep learning algorithms can identify lung nodules, fractures, and other abnormalities in X-rays and CT scans with accuracy rates exceeding 95% in controlled studies. Google Health's AI system demonstrated the ability to detect breast cancer from mammograms with greater accuracy than human radiologists, reducing false positives by 5.7% and false negatives by 9.4%.

Pathology has also benefited significantly from AI. Digital pathology platforms using convolutional neural networks (CNNs) can analyze tissue samples and identify cancerous cells with high precision. Companies like PathAI and Paige have developed FDA-approved AI tools that assist pathologists in diagnosing various cancers including prostate, breast, and gastric cancers.

[Page 2]
Ophthalmology represents another area where AI has made significant strides. AI systems can detect diabetic retinopathy, age-related macular degeneration, and glaucoma from retinal images. The IDx-DR system became the first FDA-authorized autonomous AI diagnostic system, capable of detecting diabetic retinopathy without the need for a clinician to interpret the results.

3. DRUG DISCOVERY AND DEVELOPMENT

The traditional drug discovery process takes an average of 12-15 years and costs approximately $2.6 billion. AI is dramatically accelerating this process by predicting molecular interactions, identifying potential drug candidates, and optimizing clinical trial designs. Insilico Medicine used AI to discover a novel drug candidate for idiopathic pulmonary fibrosis in just 18 months, compared to the typical 4-5 years for the discovery phase alone.

AI models can screen millions of chemical compounds virtually, predicting their efficacy, toxicity, and pharmacokinetic properties before any laboratory testing. This approach has the potential to reduce drug development costs by 50-70% and timelines by 30-50%. Major pharmaceutical companies including Pfizer, Novartis, and AstraZeneca have established AI research divisions and partnerships with AI startups to leverage these capabilities.

4. CLINICAL DECISION SUPPORT SYSTEMS

Clinical Decision Support Systems (CDSS) powered by AI analyze patient data including electronic health records, lab results, imaging data, and genomic information to provide evidence-based treatment recommendations. These systems help physicians make more informed decisions, reduce diagnostic errors, and improve patient outcomes.

[Page 3]
Natural Language Processing (NLP) plays a crucial role in CDSS by extracting structured information from unstructured clinical notes. NLP algorithms can identify relevant symptoms, diagnoses, medications, and procedures from physician notes, enabling comprehensive patient profiling and risk assessment. Studies have shown that AI-powered CDSS can reduce diagnostic errors by up to 30% and improve treatment adherence by 25%.

5. CHALLENGES AND ETHICAL CONSIDERATIONS

Despite the tremendous potential, AI in healthcare faces significant challenges. Data privacy and security remain primary concerns, as AI systems require access to sensitive patient information. Regulatory frameworks are still evolving, with the FDA developing new pathways for AI-based medical devices and software. The lack of diverse training datasets poses risks of algorithmic bias, potentially leading to healthcare disparities among underrepresented populations.

Explainability is another critical challenge. Many AI models, particularly deep learning systems, operate as "black boxes," making it difficult for clinicians to understand the reasoning behind AI recommendations. This lack of transparency can erode trust and hinder clinical adoption. Researchers are actively developing explainable AI (XAI) methods to address this issue.

6. FUTURE OUTLOOK

The future of AI in healthcare is promising, with emerging applications in precision medicine, robotic surgery, mental health monitoring, and pandemic preparedness. Multimodal AI systems that integrate data from multiple sources including imaging, genomics, wearables, and electronic health records will enable more comprehensive and accurate clinical assessments. The integration of large language models like GPT-4 and Claude into clinical workflows is expected to transform medical documentation, patient communication, and clinical research.

As the technology matures and regulatory frameworks evolve, AI is expected to become an integral part of healthcare delivery, improving access, reducing costs, and ultimately saving lives across the globe.
"""

SAMPLE_DOC_NAME = "AI_in_Healthcare_Overview.pdf"


# =========================================================
# RAG ENGINE WITH EMBEDDINGS
# =========================================================

class RAGEngine:
    """
    Production-grade RAG engine with real semantic embeddings.
    Uses sentence-transformers for embeddings + cosine similarity.
    Falls back to enhanced TF-IDF if sentence-transformers unavailable.
    """
    
    def __init__(self):
        self.chunks: List[dict] = []
        self.embeddings: Optional[np.ndarray] = None
        self.model = None
        self.doc_name: str = ""
        self.use_semantic = False
        self._load_model()
    
    def _load_model(self):
        """Try to load sentence-transformers model."""
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            self.use_semantic = True
        except ImportError:
            self.use_semantic = False
    
    def chunk_text(self, text: str, chunk_size: int = 300, overlap: int = 75) -> List[dict]:
        """Smart chunking with overlap and metadata."""
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Split by paragraphs first, then by size
        paragraphs = re.split(r'\n\n+', text)
        chunks = []
        current_chunk = ""
        current_page = 1
        chunk_id = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # Track page numbers
            page_match = re.search(r'\[Page (\d+)\]', para)
            if page_match:
                current_page = int(page_match.group(1))
                para = re.sub(r'\[Page \d+\]', '', para).strip()
                if not para:
                    continue
            
            # If adding this paragraph exceeds chunk size, save current and start new
            words_in_current = len(current_chunk.split())
            words_in_para = len(para.split())
            
            if words_in_current + words_in_para > chunk_size and current_chunk:
                chunks.append({
                    'id': chunk_id,
                    'text': current_chunk.strip(),
                    'page': current_page,
                    'word_count': len(current_chunk.split()),
                })
                chunk_id += 1
                
                # Keep overlap
                overlap_words = current_chunk.split()[-overlap:]
                current_chunk = ' '.join(overlap_words) + ' ' + para
            else:
                current_chunk = (current_chunk + ' ' + para).strip()
        
        # Don't forget the last chunk
        if current_chunk.strip():
            chunks.append({
                'id': chunk_id,
                'text': current_chunk.strip(),
                'page': current_page,
                'word_count': len(current_chunk.split()),
            })
        
        return chunks
    
    def _embed(self, texts: List[str]) -> np.ndarray:
        """Compute embeddings."""
        if self.use_semantic and self.model:
            return self.model.encode(texts, normalize_embeddings=True)
        else:
            # TF-IDF fallback with better scoring
            return self._tfidf_vectors(texts)
    
    def _build_vocab(self, texts: List[str]):
        """Build vocabulary from corpus texts."""
        self._vocab = {}
        self._doc_freq = {}
        for text in texts:
            seen = set()
            for word in re.findall(r'\b\w+\b', text.lower()):
                if word not in self._vocab:
                    self._vocab[word] = len(self._vocab)
                if word not in seen:
                    self._doc_freq[word] = self._doc_freq.get(word, 0) + 1
                    seen.add(word)
        self._num_docs = len(texts)
    
    def _tfidf_vector(self, text: str) -> np.ndarray:
        """Vectorize a single text using stored vocabulary."""
        vec = np.zeros(len(self._vocab))
        words = re.findall(r'\b\w+\b', text.lower())
        word_counts = {}
        for w in words:
            word_counts[w] = word_counts.get(w, 0) + 1
        for w, count in word_counts.items():
            if w in self._vocab:
                tf = 1 + np.log(count)
                idf = np.log(self._num_docs / (1 + self._doc_freq.get(w, 0)))
                vec[self._vocab[w]] = tf * idf
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec
    
    def _tfidf_vectors(self, texts: List[str]) -> np.ndarray:
        """TF-IDF vectorization as fallback."""
        self._build_vocab(texts)
        return np.array([self._tfidf_vector(t) for t in texts])
    
    def retrieve(self, query: str, top_k: int = 3) -> List[dict]:
        """Semantic retrieval using cosine similarity."""
        if not self.chunks:
            return []
        
        # Embed query using same method as corpus
        if self.use_semantic and self.model:
            query_vec = self.model.encode([query], normalize_embeddings=True)
        else:
            query_vec = np.array([self._tfidf_vector(query)])
        
        # Cosine similarity
        if self.embeddings is not None:
            similarities = np.dot(self.embeddings, query_vec.T).flatten()
        else:
            return []
        
        # Rank and return top-k
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            chunk = self.chunks[idx].copy()
            chunk['score'] = float(similarities[idx])
            chunk['rank'] = len(results) + 1
            results.append(chunk)
        
        return results
    
    def generate_answer(self, query: str, contexts: List[dict], api_key: str = "", provider: str = "openai") -> str:
        """Generate answer from retrieved contexts."""
        if not contexts:
            return "I couldn't find relevant information in the document for this question."
        
        # Filter low-relevance chunks
        relevant = [c for c in contexts if c['score'] > 0.05]
        if not relevant:
            return "The document doesn't seem to contain information closely related to this question. Try rephrasing or asking about a topic covered in the document."
        
        # Try API generation
        if api_key:
            try:
                return self._generate_with_api(query, relevant, api_key, provider)
            except Exception as e:
                pass
        
        # Smart extractive fallback
        return self._smart_extractive(query, relevant)
    
    def _generate_with_api(self, query: str, contexts: List[dict], api_key: str, provider: str) -> str:
        """Generate using OpenAI or Anthropic API."""
        context_text = "\n\n---\n\n".join([
            f"[Source {c['rank']} | Page {c.get('page', '?')} | Relevance: {c['score']:.0%}]\n{c['text']}" 
            for c in contexts
        ])
        
        system_prompt = (
            "You are a precise document Q&A assistant. Answer based ONLY on the provided context. "
            "Be concise, accurate, and cite which source you're using (e.g., 'According to Source 1...'). "
            "If the context doesn't fully answer the question, say what you can answer and what's missing."
        )
        
        if provider == "openai":
            import openai
            client = openai.OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {query}"}
                ],
                max_tokens=600,
                temperature=0.15,
            )
            return response.choices[0].message.content
        else:
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=600,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {query}"}
                ],
            )
            return response.content[0].text
    
    def _smart_extractive(self, query: str, contexts: List[dict]) -> str:
        """Intelligent extractive answer with highlighting."""
        query_words = set(re.findall(r'\b\w{3,}\b', query.lower()))
        
        parts = []
        for ctx in contexts:
            sentences = re.split(r'(?<=[.!?])\s+', ctx['text'])
            
            # Score sentences by query relevance
            scored_sents = []
            for sent in sentences:
                sent_words = set(re.findall(r'\b\w{3,}\b', sent.lower()))
                overlap = len(query_words & sent_words)
                if overlap > 0:
                    scored_sents.append((overlap, sent.strip()))
            
            scored_sents.sort(key=lambda x: x[0], reverse=True)
            best = scored_sents[:3] if scored_sents else [(0, sentences[0].strip())]
            
            excerpt = ' '.join([s[1] for s in best])
            page = ctx.get('page', '?')
            score_pct = f"{ctx['score']:.0%}"
            
            parts.append(f"**Source {ctx['rank']}** (Page {page}, {score_pct} match):\n> {excerpt}")
        
        answer = '\n\n'.join(parts)
        answer += "\n\n💡 *Tip: Add an API key in the sidebar for AI-generated natural language answers.*"
        return answer
    
    def ingest_text(self, text: str, filename: str) -> dict:
        """Ingest raw text."""
        self.doc_name = filename
        self.chunks = self.chunk_text(text)
        
        # Compute embeddings
        chunk_texts = [c['text'] for c in self.chunks]
        self.embeddings = self._embed(chunk_texts)
        
        return {
            'success': True,
            'doc_name': filename,
            'total_chunks': len(self.chunks),
            'total_words': sum(c['word_count'] for c in self.chunks),
            'total_pages': max((c.get('page', 0) for c in self.chunks), default=0),
            'embedding_type': 'Semantic (MiniLM-L6)' if self.use_semantic else 'TF-IDF Vector',
        }
    
    def ingest_pdf(self, pdf_bytes: bytes, filename: str) -> dict:
        """Ingest PDF file."""
        try:
            import PyPDF2
            import io
            reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
            text = ""
            for page_num, page in enumerate(reader.pages):
                page_text = page.extract_text() or ""
                text += f"\n[Page {page_num + 1}]\n{page_text}"
            return self.ingest_text(text, filename)
        except Exception as e:
            return {'success': False, 'error': str(e)}


# =========================================================
# APP STATE
# =========================================================

if 'engine' not in st.session_state:
    st.session_state.engine = RAGEngine()
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'doc_stats' not in st.session_state:
    st.session_state.doc_stats = None
if 'sample_loaded' not in st.session_state:
    st.session_state.sample_loaded = False


# =========================================================
# AUTO-LOAD SAMPLE DOCUMENT
# =========================================================

def load_sample():
    """Load the sample document automatically."""
    if not st.session_state.sample_loaded:
        result = st.session_state.engine.ingest_text(SAMPLE_DOCUMENT, SAMPLE_DOC_NAME)
        if result['success']:
            st.session_state.doc_stats = result
            st.session_state.sample_loaded = True
            st.session_state.messages = []

load_sample()


# =========================================================
# HEADER
# =========================================================

st.markdown("""
<div class="hero">
    <div class="hero-title">⚡ RAG Document Chatbot</div>
    <div class="hero-subtitle">
        Upload any PDF and ask questions — answers are retrieved and cited from your document, 
        not hallucinated. Try it now with the pre-loaded sample, or upload your own.
    </div>
    <div class="hero-badges">
        <span class="badge">🔍 Semantic Search</span>
        <span class="badge">📄 PDF Ingestion</span>
        <span class="badge">🎯 Source Attribution</span>
        <span class="badge">🤖 LLM Generation</span>
        <span class="badge badge-green">✓ Live Demo</span>
    </div>
</div>
""", unsafe_allow_html=True)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:
    # Document upload
    st.markdown('<div class="sidebar-card"><h4>📄 Document</h4></div>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Upload your own PDF",
        type=['pdf'],
        help="Replace the sample document with your own"
    )
    
    if uploaded_file is not None:
        file_hash = hashlib.md5(uploaded_file.getvalue()).hexdigest()[:8]
        if st.session_state.doc_stats is None or st.session_state.doc_stats.get('hash') != file_hash:
            with st.spinner("⚡ Processing document..."):
                result = st.session_state.engine.ingest_pdf(uploaded_file.getvalue(), uploaded_file.name)
                if result['success']:
                    st.session_state.doc_stats = {**result, 'hash': file_hash}
                    st.session_state.messages = []
                    st.session_state.sample_loaded = False
                    st.success(f"✅ Loaded **{uploaded_file.name}**")
                else:
                    st.error(f"Error: {result.get('error')}")
    
    if st.session_state.sample_loaded:
        st.info("📎 Using pre-loaded sample: **AI in Healthcare**")
    
    # Doc stats
    if st.session_state.doc_stats:
        stats = st.session_state.doc_stats
        c1, c2 = st.columns(2)
        c1.metric("Chunks", stats['total_chunks'])
        c2.metric("Pages", stats['total_pages'])
        st.caption(f"Embeddings: {stats.get('embedding_type', 'TF-IDF')}")
    
    # API Key
    st.markdown("---")
    st.markdown('<div class="sidebar-card"><h4>🔑 API Key (Optional)</h4></div>', unsafe_allow_html=True)
    
    provider = st.radio("Provider", ["OpenAI", "Anthropic"], horizontal=True, label_visibility="collapsed")
    api_key = st.text_input(
        "API Key",
        type="password",
        placeholder="sk-... or sk-ant-...",
        help="For AI-generated answers. Without it, you get extractive retrieval."
    )
    
    # RAG settings
    st.markdown("---")
    st.markdown('<div class="sidebar-card"><h4>🎛️ Retrieval Settings</h4></div>', unsafe_allow_html=True)
    top_k = st.slider("Chunks to retrieve", 1, 5, 3)
    
    # About / CTA
    st.markdown("---")
    st.markdown("""
    <div class="sidebar-card">
        <h4>👨‍💻 Built by</h4>
        <p style="font-size: 0.85rem; color: #374151; margin: 0; line-height: 1.6;">
            <strong>Dr. Rajan Tripathi, PhD</strong><br>
            Director, AI² Innovation Lab<br>
            NVIDIA DLI University Ambassador
        </p>
    </div>
    <div class="cta-box">
        <a href="https://www.upwork.com/freelancers/~01736da43bcfb3e720" target="_blank">
            📋 Hire Me on Upwork
        </a>
        <p>RAG chatbots · AI Agents · Consulting</p>
    </div>
    """, unsafe_allow_html=True)


# =========================================================
# MAIN AREA
# =========================================================

# Pipeline visualization
st.markdown("""
<div class="pipeline">
    <div class="pipeline-step">
        <div class="pipeline-icon">📄</div><br>
        <span class="pipeline-label">Ingest</span>
        <div class="pipeline-desc">PDF → Text → Chunks</div>
    </div>
    <div class="pipeline-arrow">→</div>
    <div class="pipeline-step">
        <div class="pipeline-icon">🧬</div><br>
        <span class="pipeline-label">Embed</span>
        <div class="pipeline-desc">Chunks → Vectors</div>
    </div>
    <div class="pipeline-arrow">→</div>
    <div class="pipeline-step">
        <div class="pipeline-icon">🔍</div><br>
        <span class="pipeline-label">Retrieve</span>
        <div class="pipeline-desc">Query → Top-K</div>
    </div>
    <div class="pipeline-arrow">→</div>
    <div class="pipeline-step">
        <div class="pipeline-icon">🤖</div><br>
        <span class="pipeline-label">Generate</span>
        <div class="pipeline-desc">Context → Answer</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Stats
if st.session_state.doc_stats:
    stats = st.session_state.doc_stats
    st.markdown(f"""
    <div class="stats-grid">
        <div class="stat-box">
            <div class="stat-number">{stats['total_chunks']}</div>
            <div class="stat-label">Chunks Indexed</div>
        </div>
        <div class="stat-box">
            <div class="stat-number">{stats['total_words']:,}</div>
            <div class="stat-label">Words Processed</div>
        </div>
        <div class="stat-box">
            <div class="stat-number">{stats['total_pages']}</div>
            <div class="stat-label">Pages</div>
        </div>
        <div class="stat-box">
            <div class="stat-number">{'🧬' if st.session_state.engine.use_semantic else '📊'}</div>
            <div class="stat-label">{'Semantic' if st.session_state.engine.use_semantic else 'TF-IDF'} Search</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Suggested questions (only show if no messages yet)
if not st.session_state.messages and st.session_state.sample_loaded:
    st.markdown("#### 💬 Try asking:")
    
    sample_cols = st.columns(2)
    sample_questions = [
        "What is the market size for AI in healthcare?",
        "How does AI help in drug discovery?",
        "What are the ethical challenges of AI in healthcare?",
        "How accurate is AI in detecting breast cancer?",
    ]
    
    for i, q in enumerate(sample_questions):
        with sample_cols[i % 2]:
            if st.button(q, key=f"sample_{i}", use_container_width=True):
                st.session_state.messages.append({'role': 'user', 'content': q})
                contexts = st.session_state.engine.retrieve(q, top_k=top_k)
                answer = st.session_state.engine.generate_answer(
                    q, contexts, api_key=api_key,
                    provider=provider.lower()
                )
                st.session_state.messages.append({
                    'role': 'assistant', 'content': answer,
                    'sources': [{'id': c['id'], 'page': c.get('page'), 'score': c['score'], 'rank': c['rank']} for c in contexts]
                })
                st.rerun()

st.markdown("---")

# Chat history
for msg in st.session_state.messages:
    if msg['role'] == 'user':
        st.markdown(f'<div class="msg-user">{msg["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="msg-bot">{msg["content"]}</div>', unsafe_allow_html=True)
        if msg.get('sources'):
            chips = ''.join([
                f'<span class="source-chip">📄 Page {s.get("page", "?")} <span class="score">{s["score"]:.0%}</span></span>'
                for s in msg['sources'] if s['score'] > 0.05
            ])
            if chips:
                st.markdown(f'<div class="sources-row">{chips}</div>', unsafe_allow_html=True)

# Chat input
query = st.chat_input(f"Ask about {st.session_state.engine.doc_name or 'the document'}...")

if query:
    st.session_state.messages.append({'role': 'user', 'content': query})
    st.markdown(f'<div class="msg-user">{query}</div>', unsafe_allow_html=True)
    
    with st.spinner("🔍 Retrieving & generating..."):
        contexts = st.session_state.engine.retrieve(query, top_k=top_k)
        answer = st.session_state.engine.generate_answer(
            query, contexts, api_key=api_key,
            provider=provider.lower()
        )
    
    st.markdown(f'<div class="msg-bot">{answer}</div>', unsafe_allow_html=True)
    
    if contexts:
        chips = ''.join([
            f'<span class="source-chip">📄 Page {c.get("page", "?")} <span class="score">{c["score"]:.0%}</span></span>'
            for c in contexts if c['score'] > 0.05
        ])
        if chips:
            st.markdown(f'<div class="sources-row">{chips}</div>', unsafe_allow_html=True)
    
    st.session_state.messages.append({
        'role': 'assistant', 'content': answer,
        'sources': [{'id': c['id'], 'page': c.get('page'), 'score': c['score'], 'rank': c['rank']} for c in contexts]
    })


# =========================================================
# FOOTER
# =========================================================

st.markdown("""
<div class="footer">
    Built by <a href="https://www.upwork.com/freelancers/~01736da43bcfb3e720">Dr. Rajan Tripathi</a> 
    · Director, AI² Innovation Lab · NVIDIA DLI University Ambassador<br>
    <em>Production deployments include FAISS/Qdrant vector stores, LangChain agentic retrieval, 
    multi-document support, multilingual NLP, and enterprise authentication.</em><br><br>
    <a href="https://www.upwork.com/freelancers/~01736da43bcfb3e720">📋 Hire on Upwork</a> · 
    <a href="https://linkedin.com/in/rajan-tripathi-phd-14135243">🔗 LinkedIn</a>
</div>
""", unsafe_allow_html=True)
