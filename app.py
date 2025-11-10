# app.py
"""
Warhammer Rules Assistant — lightweight FAISS + OpenAI embeddings + Streamlit
Stable, minimal dependencies for Streamlit Cloud
"""

import os
import time
import tempfile
import shutil
from typing import List, Tuple, Dict
from urllib.parse import urljoin

import streamlit as st
import requests
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader
import numpy as np
import faiss
import openai

# -------------------------
# Page config + secrets
# -------------------------
st.set_page_config(page_title="Warhammer Rules Assistant", layout="wide")
st.title(st.secrets.get("ASSISTANT_NAME", "Warhammer Rules Assistant"))

DEFAULT_MODEL = st.secrets.get("DEFAULT_MODEL", "gpt-3.5-turbo")
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.warning("OpenAI API key not found. Add OPENAI_API_KEY to Streamlit Secrets or environment to enable LLM.")
openai.api_key = OPENAI_API_KEY

WARHAMMER_URLS = {
    "40K": "https://www.warhammer-community.com/en-gb/downloads/warhammer-40000/",
    "Age of Sigmar": "https://www.warhammer-community.com/en-gb/downloads/warhammer-age-of-sigmar/",
    "Kill Team": "https://www.warhammer-community.com/en-gb/downloads/kill-team/"
}

# -------------------------
# Helpers: PDF fetch/parse
# -------------------------
def fetch_pdf_links(url: str) -> List[str]:
    resp = requests.get(url, timeout=20)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.lower().endswith(".pdf"):
            full = urljoin(url, href)
            if full not in links:
                links.append(full)
    return links

def download_pdfs(links: List[str], target_dir: str, max_files: int = None, progress_callback=None) -> List[str]:
    os.makedirs(target_dir, exist_ok=True)
    downloaded = []
    count = 0
    total = len(links) if (max_files is None) else min(len(links), max_files)
    for link in links:
        if max_files is not None and count >= max_files:
            break
        filename = os.path.join(target_dir, os.path.basename(link.split("?")[0]))
        if os.path.exists(filename):
            downloaded.append(filename)
            count += 1
            if progress_callback:
                progress_callback(count, total, f"Cached: {os.path.basename(filename)}")
            continue
        try:
            r = requests.get(link, timeout=30)
            r.raise_for_status()
            with open(filename, "wb") as f:
                f.write(r.content)
            downloaded.append(filename)
            count += 1
            if progress_callback:
                progress_callback(count, total, os.path.basename(filename))
        except Exception as e:
            st.warning(f"Failed to download {link}: {e}")
    return downloaded

def extract_text_from_pdf(path: str) -> str:
    text_parts = []
    try:
        reader = PdfReader(path)
        for p in reader.pages:
            try:
                text = p.extract_text() or ""
            except Exception:
                text = ""
            if text:
                text_parts.append(text)
    except Exception as e:
        st.warning(f"Failed to read PDF {path}: {e}")
    return "\n\n".join(text_parts)

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    if not text:
        return []
    tokens = text.split()
    chunks = []
    i = 0
    while i < len(tokens):
        chunk = tokens[i:i+chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap
    return chunks

# -------------------------
# Embedding & FAISS helpers
# -------------------------
def embed_texts(texts: List[str], model: str = "text-embedding-3-small", batch_size: int = 16) -> List[List[float]]:
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        resp = openai.Embedding.create(input=batch, model=model)
        batch_emb = [item["embedding"] for item in resp["data"]]
        embeddings.extend(batch_emb)
        time.sleep(0.1)
    return embeddings

@st.cache_resource(show_spinner=False)
def create_faiss_resource(dim: int = 1536):
    index = faiss.IndexFlatIP(dim)
    metadata = {"texts": [], "sources": []}
    return {"index": index, "metadata": metadata}

def add_to_faiss(index, metadata, embeddings: List[List[float]], texts: List[str], sources: List[str]):
    arr = np.array(embeddings).astype("float32")
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    arr = arr / norms
    index.add(arr)
    metadata["texts"].extend(texts)
    metadata["sources"].extend(sources)

def search_faiss(index, metadata, query_embedding, top_k=4):
    q = np.array(query_embedding).astype("float32").reshape(1, -1)
    q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)
    D, I = index.search(q, top_k)
    results = []
    for score, idx in zip(D[0], I[0]):
        if idx < 0:
            continue
        results.append({"score": float(score), "text": metadata["texts"][idx], "source": metadata["sources"][idx]})
    return results

@st.cache_data(show_spinner=False)
def build_index_from_pdfs(pdf_paths: Tuple[str], openai_api_key: str, chunk_size: int = 1000, overlap: int = 200, progress_hook=None):
    os.environ["OPENAI_API_KEY"] = openai_api_key
    res = create_faiss_resource()
    index = res["index"]
    metadata = res["metadata"]

    local_texts = []
    local_sources = []
    for p in pdf_paths:
        text = extract_text_from_pdf(p)
        chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        for c in chunks:
            local_texts.append(c)
            local_sources.append(os.path.basename(p))
        if progress_hook:
            progress_hook(len(local_texts))
    if not local_texts:
        raise ValueError("No text extracted from PDFs.")
    embeddings = embed_texts(local_texts)
    add_to_faiss(index, metadata, embeddings, local_texts, local_sources)
    return {"count": len(local_texts), "docs": len(pdf_paths)}

# -------------------------
# UI
# -------------------------
st.sidebar.header("Configuration")
system = st.sidebar.selectbox("Select ruleset", list(WARHAMMER_URLS.keys()))
max_pdfs = st.sidebar.slider("Max PDFs to fetch (smaller = faster)", 1, 20, 5)
chunk_size = st.sidebar.number_input("Chunk token size", min_value=200, max_value=2000, value=1000, step=100)
overlap = st.sidebar.number_input("Chunk overlap", min_value=0, max_value=500, value=200, step=50)

if "last_indexed" not in st.session_state:
    st.session_state["last_indexed"] = None

st.sidebar.markdown("---")
st.sidebar.markdown("Note: Index is in-memory. For larger corpora use persistent storage.")

if st.sidebar.button("Fetch & Index Official PDFs"):
    with st.spinner("Fetching PDF links..."):
        try:
            links = fetch_pdf_links(WARHAMMER_URLS[system])
        except Exception as e:
            st.error(f"Failed to fetch links: {e}")
            links = []
    if not links:
        st.info("No PDF links found on the page.")
    else:
        tempdir = tempfile.mkdtemp(prefix="rules_")
        progress = st.progress(0)
        status = st.empty()
        def prog_cb(count, total, name=""):
            progress.progress(int((count/total)*100))
            status.text(f"Downloaded {count}/{total}: {name}")

        with st.spinner("Downloading PDFs..."):
            downloaded = download_pdfs(links, tempdir, max_files=max_pdfs, progress_callback=prog_cb)

        if not downloaded:
            st.error("No PDFs downloaded.")
        else:
            status.text("Extracting text and building index...")
            def build_progress_hook(n):
                status.text(f"Prepared {n} chunks (approx)")
            try:
                res = build_index_from_pdfs(tuple(downloaded), OPENAI_API_KEY, chunk_size=chunk_size, overlap=overlap, progress_hook=build_progress_hook)
                st.session_state["last_indexed"] = f"{res['docs']} PDFs, {res['count']} chunks"
                st.success(f"Indexed {res['docs']} PDFs into {res['count']} chunks.")
            except Exception as e:
                st.error(f"Index build failed: {e}")
        try:
            shutil.rmtree(tempdir)
        except Exception:
            pass

st.header("Ask the Rules Assistant")
st.markdown(st.secrets.get("WELCOME_MESSAGE", "Ask a rules question about 40K, Age of Sigmar, or Kill Team."))

question = st.text_input("Enter your question here (e.g., 'Can a unit that Advanced charge?')")

col1, col2 = st.columns([3,1])
with col2:
    st.markdown("**Index status**")
    st.write(st.session_state.get("last_indexed", "No index built yet"))
    if st.button("Clear index"):
        try:
            create_faiss_resource.clear()
            build_index_from_pdfs.clear()
            st.session_state["last_indexed"] = None
            st.success("Index cleared.")
        except Exception:
            st.error("Failed to clear cache. Try reloading the app.")

if st.button("Ask") and question:
    res = create_faiss_resource()
    index = res["index"]
    metadata = res["metadata"]
    if index.ntotal == 0:
        st.error("Index is empty. Please Fetch & Index PDFs first.")
    else:
        with st.spinner("Searching and generating answer..."):
            try:
                q_emb = embed_texts([question])[0]
                results = search_faiss(index, metadata, q_emb, top_k=6)
                context = "\n\n---\n\n".join([r["text"] for r in results])
                system_prompt = st.secrets.get("SYSTEM_PROMPT", "You are a rules assistant. Answer concisely and cite sources when possible.")
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Use only the context below to answer the question.\n\nContext:\n{context}\n\nQuestion: {question}"}
                ]
                chat_resp = openai.ChatCompletion.create(model=DEFAULT_MODEL, messages=messages, max_tokens=512, temperature=0.0)
                answer = chat_resp["choices"][0]["message"]["content"].strip()
                st.markdown("### Answer")
                st.write(answer)
                if results:
                    st.markdown("### Retrieved snippets (sources)")
                    for r in results:
                        st.write(f"**Source:** {r['source']} — score: {r['score']:.3f}")
                        st.write(r['text'][:1000] + ("..." if len(r['text']) > 1000 else ""))
            except Exception as e:
                st.error(f"Failed to generate answer: {e}")
