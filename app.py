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
    """
    Fetches and filters official Warhammer download PDFs.
    Focuses only on core rules, FAQs, and balance updates.
    Supports redirect links and embedded JSON data.
    """
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    links = set()

    # Keywords to prioritize (core rules, FAQs, balance dataslates)
    PRIORITY_KEYWORDS = [
        "core rules", "core-rules",
        "rules commentary", "faq",
        "balance dataslate", "balance-dataslate",
        "errata", "update"
    ]

    # --- 1️⃣ Direct .pdf links or redirects ---
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        text = a.get_text(strip=True).lower()
        target = f"{href} {text}"

        # Filter for relevant content
        if any(k in target.lower() for k in PRIORITY_KEYWORDS):
            full_url = urljoin(url, href)
            if full_url.lower().endswith(".pdf"):
                links.add(full_url)
            elif "redirect" in full_url and "link" in full_url:
                try:
                    r2 = requests.get(full_url, allow_redirects=True, timeout=20)
                    if r2.url.lower().endswith(".pdf"):
                        links.add(r2.url)
                except Exception as e:
                    st.warning(f"Redirect failed: {href} ({e})")

    # --- 2️⃣ Check for JSON-style embedded data blocks ---
    try:
        for script in soup.find_all("script"):
            if "downloads" in script.text and ".pdf" in script.text:
                for line in script.text.split('"'):
                    if line.lower().endswith(".pdf"):
                        if any(k in line.lower() for k in PRIORITY_KEYWORDS):
                            links.add(line)
    except Exception:
        pass

    # --- 3️⃣ Sanitize duplicates and return sorted list ---
    clean_links = sorted(set(links))
    if not clean_links:
        st.warning("⚠️ No relevant PDFs found — the page format may have changed.")
    else:
        st.info(f"✅ Found {len(clean_links)} relevant files.")
    return clean_links


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
