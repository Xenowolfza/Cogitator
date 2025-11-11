# app.py
"""
Warhammer Rules Assistant — lightweight FAISS + OpenAI embeddings + Streamlit
Stable, minimal dependencies for Streamlit Cloud
"""

import os
import time
import tempfile
import shutil
import io
import base64
from typing import List, Tuple, Dict

import streamlit as st
import requests
from PyPDF2 import PdfReader
import numpy as np
import faiss
from openai import OpenAI
from openai import APIError
from PIL import Image

# -------------------------
# Page config + secrets
# -------------------------
st.set_page_config(page_title="Warhammer Rules Assistant", layout="wide")
st.title(st.secrets.get("ASSISTANT_NAME", "Warhammer Rules Assistant"))
st.caption("⚙️ The Omnissiah’s Cogitator — AI-assisted rules analyst for Warhammer 40,000, Age of Sigmar, and Kill Team.")

DEFAULT_MODEL = st.secrets.get("DEFAULT_MODEL", "gpt-3.5-turbo")
VISION_MODEL = st.secrets.get("VISION_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = st.secrets.get("EMBEDDING_MODEL", "text-embedding-ada-002")
<<<<<<< HEAD
EMBEDDING_DIM = 3072 if "3-large" in EMBEDDING_MODEL else 1536
=======
>>>>>>> 6ac279395fa616c1bd912e81e61228dacf3a1e84
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.warning("OpenAI API key not found. Add OPENAI_API_KEY to Streamlit Secrets or environment to enable LLM.")

client = OpenAI(api_key=OPENAI_API_KEY)

WARHAMMER_PDFS = {
    "40K": [
        {
            "title": "Core Rules",
            "url": "https://assets.warhammer-community.com/warhammer40000_core&key_corerules_eng_24.09-5xfayxjekm.pdf",
            "description": "Essential rules for Warhammer 40,000 battles."
        },
        {
            "title": "Core Rules Updates and Commentary",
            "url": "https://assets.warhammer-community.com/eng_17-09_warhammer40000_core_rules_updates_and_commentary-htinngebrw-te32nyhkht.pdf",
            "description": "Amendments and player feedback responses (September 2025)."
        },
        {
            "title": "Balance Dataslate",
            "url": "https://assets.warhammer-community.com/eng_08-10_warhammer40000_core_rules_balance_dataslate-f47uib0gs9-9kju9nznun.pdf",
            "description": "Balance adjustments for competitive play (October 2025)."
        },
        {
            "title": "Munitorum Field Manual",
            "url": "https://assets.warhammer-community.com/warhammer40000_core&key_munitorumfieldmanual_eng_16.10.pdf",
            "description": "Points values for all factions."
        },
        {
            "title": "Quick Start Guide",
            "url": "https://assets.warhammer-community.com/warhammer40000_core&key_quickstartguide_eng_24.09-s2afk26smk.pdf",
            "description": "Beginner introduction to gameplay."
        },
        {
            "title": "Crusade Rules",
            "url": "https://assets.warhammer-community.com/warhammer40000_crusade_crusaderules_eng_24.09-x7lpyyilc9.pdf",
            "description": "Narrative campaign rules."
        }
    ],
    "Age of Sigmar": [
        {
            "title": "Core Rules",
            "url": "https://assets.warhammer-community.com/ageofsigmar_corerules&keydownloads_therules_eng_24.09-tbf4egjql3.pdf",
            "description": "Fundamental rules for Age of Sigmar battles."
        },
        {
            "title": "Rules Updates",
            "url": "https://assets.warhammer-community.com/eng_24-09_aos_core_rules_rules_updates_september_2025-meyxmktmox-qwey0jc7h2.pdf",
            "description": "Core rules amendments (September 2025)."
        },
        {
            "title": "Battle Profiles and Rules Updates",
            "url": "https://assets.warhammer-community.com/eng_24-09_aos_core_rules_battle_profiles_and_rules_updates_september_2025-fjrsbz5oll-rxddil82hp.pdf",
            "description": "Unit profiles and updates (September 2025)."
        },
        {
            "title": "Quick Start Guide",
            "url": "https://assets.warhammer-community.com/ageofsigmar_corerules&keydownloads_quickstartguide_eng_24.09-xoffxcicsi.pdf",
            "description": "Introductory gameplay guide."
        }
    ],
    "Kill Team": [
        {
            "title": "Lite Rules",
            "url": "https://assets.warhammer-community.com/eng_jul25_kt_lite_rules-jmjv4hdamy-qlsqxdf83p.pdf",
            "description": "Simplified rules for Kill Team skirmishes (July 2025)."
        },
        {
            "title": "Universal Equipment Rules",
            "url": "https://assets.warhammer-community.com/rules-downloads/kill-team/key-downloads/universal-equipment-rules/killteam_keydownloads_universalequipment_eng_02.10.24.pdf",
            "description": "Equipment options for all teams."
        }
    ]
}

# -------------------------
# Helpers: PDF download/extract/chunk
# -------------------------
def download_pdfs(pdf_tuples: List[Tuple[str, str]], tempdir: str, progress_callback=None) -> List[Tuple[str, str]]:
    """
    Download PDFs from (url, title) tuples.
    Returns list of (local_path, title) for successful downloads.
    """
    downloaded = []
    total = len(pdf_tuples)
    for i, (url, title) in enumerate(pdf_tuples):
        try:
            filename = f"{title.replace(' ', '_')}.pdf"
            path = os.path.join(tempdir, filename)
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            with open(path, 'wb') as f:
                f.write(resp.content)
            downloaded.append((path, title))
            if progress_callback:
                progress_callback(i + 1, total, title)
        except Exception as e:
            st.warning(f"Failed to download {title}: {e}")
    return downloaded

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from a PDF using PyPDF2."""
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception as e:
        st.warning(f"Text extraction failed for {pdf_path}: {e}")
        return ""

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Split text into overlapping chunks."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

# -------------------------
# OCR Helper using OpenAI Vision
# -------------------------
<<<<<<< HEAD
@st.cache_resource(show_spinner=False)
def get_vision_client():
    return OpenAI(api_key=OPENAI_API_KEY)

def extract_text_from_image(image_bytes: bytes) -> str:
    """Extract text from an image using OpenAI GPT-4 Vision (OCR)."""
    try:
        vision_client = get_vision_client()
=======
def extract_text_from_image(image_bytes: bytes) -> str:
    """Extract text from an image using OpenAI GPT-4 Vision (OCR)."""
    try:
>>>>>>> 6ac279395fa616c1bd912e81e61228dacf3a1e84
        # Encode image to base64
        image = Image.open(io.BytesIO(image_bytes))
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        # Vision prompt for OCR
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Extract all text from this image accurately. Focus on rules, labels, or descriptions. Output only the extracted text, no additional commentary."},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_str}"}}
                ]
            }
        ]
<<<<<<< HEAD
        response = vision_client.chat.completions.create(
=======
        response = client.chat.completions.create(
>>>>>>> 6ac279395fa616c1bd912e81e61228dacf3a1e84
            model=VISION_MODEL,
            messages=messages,
            max_tokens=1000,
            temperature=0.0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.warning(f"Vision OCR extraction failed: {e}")
        return ""

# -------------------------
# Embedding & FAISS helpers
# -------------------------
def embed_texts(texts: List[str], model: str = EMBEDDING_MODEL, batch_size: int = 16) -> List[List[float]]:
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        try:
            resp = client.embeddings.create(input=batch, model=model)
            batch_emb = [item.embedding for item in resp.data]
            embeddings.extend(batch_emb)
        except APIError as e:
            if e.code == 'insufficient_quota':
                raise Exception(f"OpenAI quota exceeded. Please check your plan and billing details: {e}")
            elif e.code == 429:
                st.warning("Rate limit hit. Retrying after delay...")
                time.sleep(60)  # Wait 1 minute for rate limit
                resp = client.embeddings.create(input=batch, model=model)
                batch_emb = [item.embedding for item in resp.data]
                embeddings.extend(batch_emb)
            else:
                raise e
        time.sleep(0.1)
    return embeddings

@st.cache_resource(show_spinner=False)
def get_faiss_store() -> Dict[str, Dict]:
    """Central cache store for all system-specific FAISS indexes."""
    return {}

<<<<<<< HEAD
def create_faiss_resource(system: str, dim: int = EMBEDDING_DIM):
=======
def create_faiss_resource(system: str, dim: int = 1536):
>>>>>>> 6ac279395fa616c1bd912e81e61228dacf3a1e84
    """Creates or retrieves a FAISS index specific to a Warhammer ruleset."""
    store = get_faiss_store()
    if system not in store:
        index = faiss.IndexFlatIP(dim)
        metadata = {"texts": [], "sources": []}
        store[system] = {"index": index, "metadata": metadata}
        st.info(f"🧠 Created new FAISS index for {system}.")
    return store[system]

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
def build_index_from_pdfs(pdf_paths: Tuple[str], openai_api_key: str, system: str, chunk_size: int = 1000, overlap: int = 200, _progress_hook=None):
    res = create_faiss_resource(system)
    index = res["index"]
    metadata = res["metadata"]

    local_texts = []
    local_sources = []
    for p in pdf_paths:
        text = extract_text_from_pdf(p)
        # OCR fallback for image-based PDFs (simplistic: attempts on first ~3MB; requires pdf2image for full impl.)
        if not text.strip():
            try:
                with open(p, "rb") as f:
                    first_page_bytes = f.read(3000000)  # Approx first page
                st.info(f"⚙️ Attempting OCR on {os.path.basename(p)} (image-based PDF)...")
                ocr_text = extract_text_from_image(first_page_bytes)
                if ocr_text:
                    text += ocr_text
            except Exception as ocr_e:
                st.warning(f"OCR fallback failed for {os.path.basename(p)}: {ocr_e}")
        chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        for c in chunks:
            local_texts.append(c)
            local_sources.append(os.path.basename(p))
        if _progress_hook:
            _progress_hook(len(local_texts))
    if not local_texts:
        raise ValueError("No text extracted from PDFs.")
    embeddings = embed_texts(local_texts)
    add_to_faiss(index, metadata, embeddings, local_texts, local_sources)
    return {"count": len(local_texts), "docs": len(pdf_paths)}

# -------------------------
# UI
# -------------------------
st.sidebar.header("Configuration")
system = st.sidebar.selectbox("Select ruleset", list(WARHAMMER_PDFS.keys()))

# Detect ruleset changes and reset session-specific state
if "active_ruleset" not in st.session_state or st.session_state["active_ruleset"] != system:
    st.session_state["active_ruleset"] = system
    st.session_state["last_indexed"] = None
    st.info(f"🔄 Switched to {system}. You may rebuild or query its unique index.")

max_pdfs = st.sidebar.slider("Max PDFs to fetch (smaller = faster)", 1, 20, 5)
chunk_size = st.sidebar.number_input("Chunk token size", min_value=200, max_value=2000, value=1000, step=100)
overlap = st.sidebar.number_input("Chunk overlap", min_value=0, max_value=500, value=200, step=50)

if "last_indexed" not in st.session_state:
    st.session_state["last_indexed"] = None

st.sidebar.markdown("---")
st.sidebar.markdown("Note: Index is in-memory. For larger corpora use persistent storage.")

if st.sidebar.button("Fetch & Index Official PDFs"):
    pdf_list = WARHAMMER_PDFS[system]
    selected_pdfs = pdf_list[:max_pdfs]  # Limit to slider value

    st.info(f"✅ Using {len(selected_pdfs)} predefined Warhammer PDFs for {system}.")
    for item in selected_pdfs:
        st.write(f"- {item['title']}: {item['description']}")
        st.write(f"  URL: {item['url']}")

    tempdir = tempfile.mkdtemp(prefix="rules_")
    progress = st.progress(0)
    status = st.empty()

    def prog_cb(count, total, name=""):
        progress.progress(int((count / total) * 100))
        status.text(f"Downloaded {count}/{total}: {name}")

    with st.spinner("Downloading PDFs..."):
        downloaded = download_pdfs(
            [(item["url"], item["title"]) for item in selected_pdfs],
            tempdir,
            progress_callback=prog_cb
        )

    if not downloaded:
        st.error("No PDFs downloaded.")
    else:
        status.text("Extracting text and building index...")
        def build_progress_hook(n):
            status.text(f"Prepared {n} chunks (approx.)")

        try:
            res = build_index_from_pdfs(
                tuple([path for path, _ in downloaded]),
                OPENAI_API_KEY,
                system,
                chunk_size=chunk_size,
                overlap=overlap,
                _progress_hook=build_progress_hook
            )
            st.session_state["last_indexed"] = f"{len(downloaded)} PDFs, {res['count']} chunks"
            st.success(f"Indexed {res['docs']} PDFs into {res['count']} chunks.")
            st.session_state["current_system"] = system
        except APIError as e:
            if e.code == 'insufficient_quota':
                st.error("OpenAI quota exceeded. Please upgrade your plan or wait for reset. Details: https://platform.openai.com/account/usage")
            else:
                st.error(f"OpenAI API error during indexing: {e}")
        except Exception as e:
            st.error(f"Index build failed: {e}")

    try:
        shutil.rmtree(tempdir)
    except Exception:
        pass

st.header("Ask the Rules Assistant")
st.caption(f"📘 Active ruleset: **{system}**")
st.markdown(st.secrets.get("WELCOME_MESSAGE", "Ask a core rules question about 40K, Age of Sigmar, or Kill Team."))

question = st.text_input("Enter your question here (e.g., 'Can a unit that Advanced charge?'- not : 'can ultramarine advance after charging?' )")

# OCR Image Upload
uploaded_image = st.file_uploader("Upload an image for OCR (optional, e.g., screenshot of rules)", type=['png', 'jpg', 'jpeg', 'tiff'])

image_text = ""
if uploaded_image is not None:
    image_text = extract_text_from_image(uploaded_image.read())
    st.text_area("Extracted text from image (via OpenAI Vision):", value=image_text, height=150, disabled=True)
    if st.button("Add image text to query context"):
        st.session_state["image_text"] = image_text
        st.success("Image text added to context!")

col1, col2 = st.columns([3,1])
with col2:
    st.markdown("**Index status**")
    st.write(st.session_state.get("last_indexed", "No index built yet"))
    if st.button("Show indexed sources"):
        try:
            store = get_faiss_store()
            if system in store:
                st.json(store[system]["metadata"]["sources"])
            else:
                st.info(f"No index found for {system}.")
        except Exception as e:
            st.error(f"Failed to show sources: {e}")
    if st.button("Clear index"):
        try:
            store = get_faiss_store()
            if system in store:
                del store[system]
                st.success(f"Cleared FAISS index for {system}.")
            else:
                st.info(f"No index found for {system}.")
            build_index_from_pdfs.clear()
            st.session_state["last_indexed"] = None
            if "image_text" in st.session_state:
                del st.session_state["image_text"]
        except Exception as e:
            st.error(f"Failed to clear cache: {e}")

if st.button("Ask") and question:
    res = create_faiss_resource(st.session_state.get("current_system", system))
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
                
                # Include image text if available
                if "image_text" in st.session_state and st.session_state["image_text"]:
                    context += f"\n\n---\n\nImage OCR Context:\n{st.session_state['image_text']}"
                
                system_prompt = st.secrets.get("SYSTEM_PROMPT", "You are a rules assistant. Answer concisely and cite sources when possible.")
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Use only the context below to answer the question.\n\nContext:\n{context}\n\nQuestion: {question}"}
                ]
                chat_resp = client.chat.completions.create(model=DEFAULT_MODEL, messages=messages, max_tokens=512, temperature=0.0)
                answer = chat_resp.choices[0].message.content.strip()
                st.markdown("### Answer")
                st.write(answer)
                if results:
                    st.markdown("### Retrieved snippets (sources)")
                    for r in results:
                        st.write(f"**Source:** {r['source']} — score: {r['score']:.3f}")
                        st.write(r['text'][:1000] + ("..." if len(r['text']) > 1000 else ""))
            except APIError as e:
                if e.code == 'insufficient_quota':
                    st.error("OpenAI quota exceeded. Please upgrade your plan or wait for reset. Details: https://platform.openai.com/account/usage")
                elif e.code == 429:
                    st.error("Rate limit hit. Please try again later.")
                else:
                    st.error(f"OpenAI API error during query: {e}")
            except Exception as e:
                st.error(f"Failed to generate answer: {e}")
<<<<<<< HEAD

st.markdown("---")
st.caption("💾 Powered by OpenAI embeddings and FAISS · © Games Workshop data used for personal reference only, all property and info belong to them")
=======
>>>>>>> 6ac279395fa616c1bd912e81e61228dacf3a1e84
