# app.py
"""
Warhammer Rules Assistant — FAISS + OpenAI embeddings + Streamlit
Features: conversation memory, auto PDF discovery, custom PDF URLs, OCR via Vision.
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
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader
import numpy as np
import faiss
from openai import OpenAI, APIError
from PIL import Image

# ─── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Warhammer Rules Assistant", layout="wide", page_icon="⚙️")
st.title(st.secrets.get("ASSISTANT_NAME", "⚙️ The Omnissiah's Cogitator"))
st.caption("AI-assisted rules analyst for Warhammer 40,000, Age of Sigmar, and Kill Team.")

DEFAULT_MODEL   = st.secrets.get("DEFAULT_MODEL",   "gpt-4o-mini")
VISION_MODEL    = st.secrets.get("VISION_MODEL",    "gpt-4o")
EMBEDDING_MODEL = st.secrets.get("EMBEDDING_MODEL", "text-embedding-ada-002")
EMBEDDING_DIM   = 3072 if "3-large" in EMBEDDING_MODEL else 1536
OPENAI_API_KEY  = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.error("OpenAI API key not found. Add OPENAI_API_KEY to Streamlit Secrets or environment.")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)

# Community download page URLs (JS-rendered — used as reference and for discovery)
COMMUNITY_DOWNLOAD_URLS = {
    "40K":          "https://www.warhammer-community.com/en-gb/downloads/warhammer-40000/",
    "Age of Sigmar":"https://www.warhammer-community.com/en-gb/downloads/warhammer-age-of-sigmar/",
    "Kill Team":    "https://www.warhammer-community.com/en-gb/downloads/kill-team/",
}

# ─── Curated PDF catalogue ─────────────────────────────────────────────────────
# category field is used in the "Browse available PDFs" expander.
# The MFM is now a live web app at https://www.warhammer-community.com/en-gb/munitorum-field-manual/
# and is no longer distributed as a downloadable PDF.
WARHAMMER_PDFS: Dict[str, List[Dict]] = {
    "40K": [
        {
            "title": "Core Rules",
            "url": "https://assets.warhammer-community.com/warhammer40000_core&key_corerules_eng_24.09-5xfayxjekm.pdf",
            "description": "Core rules for Warhammer 40,000 battles.",
            "category": "Core",
        },
        {
            "title": "Core Rules Updates and Commentary",
            "url": "https://assets.warhammer-community.com/eng_17-09_warhammer40000_core_rules_updates_and_commentary-htinngebrw-te32nyhkht.pdf",
            "description": "Amendments and commentary (September 2025).",
            "category": "Core",
        },
        {
            "title": "Balance Dataslate",
            "url": "https://assets.warhammer-community.com/eng_08-10_warhammer40000_core_rules_balance_dataslate-f47uib0gs9-9kju9nznun.pdf",
            "description": "Balance adjustments for competitive play (October 2025).",
            "category": "Core",
        },
        {
            "title": "Quick Start Guide",
            "url": "https://assets.warhammer-community.com/warhammer40000_core&key_quickstartguide_eng_24.09-s2afk26smk.pdf",
            "description": "Beginner introduction to gameplay.",
            "category": "Core",
        },
        {
            "title": "Crusade Rules",
            "url": "https://assets.warhammer-community.com/warhammer40000_crusade_crusaderules_eng_24.09-x7lpyyilc9.pdf",
            "description": "Narrative campaign rules.",
            "category": "Crusade",
        },
    ],
    "Age of Sigmar": [
        {
            "title": "Core Rules",
            "url": "https://assets.warhammer-community.com/ageofsigmar_corerules&keydownloads_therules_eng_24.09-tbf4egjql3.pdf",
            "description": "Fundamental rules for Age of Sigmar battles.",
            "category": "Core",
        },
        {
            "title": "Rules Updates",
            "url": "https://assets.warhammer-community.com/eng_24-09_aos_core_rules_rules_updates_september_2025-meyxmktmox-qwey0jc7h2.pdf",
            "description": "Core rules amendments (September 2025).",
            "category": "Core",
        },
        {
            "title": "Battle Profiles and Rules Updates",
            "url": "https://assets.warhammer-community.com/eng_24-09_aos_core_rules_battle_profiles_and_rules_updates_september_2025-fjrsbz5oll-rxddil82hp.pdf",
            "description": "Unit profiles and updates (September 2025).",
            "category": "Core",
        },
        {
            "title": "Quick Start Guide",
            "url": "https://assets.warhammer-community.com/ageofsigmar_corerules&keydownloads_quickstartguide_eng_24.09-xoffxcicsi.pdf",
            "description": "Introductory gameplay guide.",
            "category": "Core",
        },
    ],
    "Kill Team": [
        {
            "title": "Lite Rules",
            "url": "https://assets.warhammer-community.com/eng_jul25_kt_lite_rules-jmjv4hdamy-qlsqxdf83p.pdf",
            "description": "Simplified rules for Kill Team skirmishes (July 2025).",
            "category": "Core",
        },
        {
            "title": "Universal Equipment Rules",
            "url": "https://assets.warhammer-community.com/rules-downloads/kill-team/key-downloads/universal-equipment-rules/killteam_keydownloads_universalequipment_eng_02.10.24.pdf",
            "description": "Equipment options for all teams.",
            "category": "Core",
        },
        {
            "title": "Core Rules Update Log",
            "url": "https://assets.warhammer-community.com/eng_kt_core_rules_update_log-l0ivf5fkvl-jgopbphagb.pdf",
            "description": "Update log for Kill Team core rules.",
            "category": "Core",
        },
    ],
}

# ─── HTTP headers for GW asset requests ───────────────────────────────────────
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}

# ─── PDF discovery ─────────────────────────────────────────────────────────────
def discover_pdfs_from_community(system: str) -> List[Dict]:
    """
    Attempt to find new PDFs on the Warhammer Community downloads page.
    The site is JavaScript-rendered so results may be sparse; returns whatever
    PDF links are present in the static HTML.
    """
    page_url = COMMUNITY_DOWNLOAD_URLS.get(system, "")
    if not page_url:
        return []
    found = []
    try:
        resp = requests.get(page_url, timeout=15, headers=_HEADERS)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if "assets.warhammer-community.com" in href and href.lower().endswith(".pdf"):
                title = (a.get_text(strip=True) or os.path.basename(href.split("?")[0]))[:80]
                found.append({
                    "title": title,
                    "url": href,
                    "description": "Auto-discovered from Warhammer Community",
                    "category": "Discovered",
                })
    except Exception as exc:
        st.warning(f"Discovery request failed: {exc}")
    return found


# ─── Download / extract / chunk ────────────────────────────────────────────────
def download_pdfs(pdf_tuples: List[Tuple[str, str]], tempdir: str, progress_callback=None) -> List[Tuple[str, str]]:
    downloaded = []
    total = len(pdf_tuples)
    for i, (url, title) in enumerate(pdf_tuples):
        try:
            filename = f"{title.replace(' ', '_').replace('/', '_')}.pdf"
            path = os.path.join(tempdir, filename)
            resp = requests.get(url, timeout=30, headers=_HEADERS)
            resp.raise_for_status()
            with open(path, "wb") as f:
                f.write(resp.content)
            downloaded.append((path, title))
            if progress_callback:
                progress_callback(i + 1, total, title)
        except Exception as e:
            st.warning(f"Failed to download **{title}**: {e}")
    return downloaded


def extract_text_from_pdf(pdf_path: str) -> str:
    try:
        reader = PdfReader(pdf_path)
        return "".join(page.extract_text() or "" for page in reader.pages)
    except Exception as e:
        st.warning(f"Text extraction failed for {os.path.basename(pdf_path)}: {e}")
        return ""


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i : i + chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks


# ─── OCR via OpenAI Vision ─────────────────────────────────────────────────────
def extract_text_from_image(image_bytes: bytes) -> str:
    try:
        image = Image.open(io.BytesIO(image_bytes))
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        img_b64 = base64.b64encode(buf.getvalue()).decode()
        messages = [{
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Extract all text from this image accurately. "
                        "Focus on rules text, labels, and descriptions. "
                        "Output only the extracted text, no additional commentary."
                    ),
                },
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
            ],
        }]
        resp = client.chat.completions.create(
            model=VISION_MODEL, messages=messages, max_tokens=1000, temperature=0.0
        )
        return resp.choices[0].message.content.strip()
    except APIError as e:
        if getattr(e, "code", None) == "model_not_found":
            st.error(
                f"Vision model '{VISION_MODEL}' not found. "
                "Update VISION_MODEL in secrets (try 'gpt-4o')."
            )
        else:
            st.warning(f"Vision OCR failed: {e}")
        return ""
    except Exception as e:
        st.warning(f"Vision OCR failed: {e}")
        return ""


# ─── Embedding & FAISS ─────────────────────────────────────────────────────────
def embed_texts(texts: List[str], batch_size: int = 16) -> List[List[float]]:
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        try:
            resp = client.embeddings.create(input=batch, model=EMBEDDING_MODEL)
            embeddings.extend(item.embedding for item in resp.data)
        except APIError as e:
            if getattr(e, "code", None) == 429:
                st.warning("Rate limit hit — waiting 60 s…")
                time.sleep(60)
                resp = client.embeddings.create(input=batch, model=EMBEDDING_MODEL)
                embeddings.extend(item.embedding for item in resp.data)
            else:
                raise
        time.sleep(0.1)
    return embeddings


@st.cache_resource(show_spinner=False)
def get_faiss_store() -> Dict:
    return {}


def create_faiss_resource(system: str) -> Dict:
    store = get_faiss_store()
    if system not in store:
        index = faiss.IndexFlatIP(EMBEDDING_DIM)
        store[system] = {"index": index, "metadata": {"texts": [], "sources": []}}
    return store[system]


def add_to_faiss(index, metadata, embeddings, texts, sources):
    arr = np.array(embeddings, dtype="float32")
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    arr /= norms
    index.add(arr)
    metadata["texts"].extend(texts)
    metadata["sources"].extend(sources)


def search_faiss(index, metadata, query_embedding, top_k: int = 6) -> List[Dict]:
    q = np.array(query_embedding, dtype="float32").reshape(1, -1)
    q /= np.linalg.norm(q) + 1e-12
    D, I = index.search(q, top_k)
    return [
        {
            "score": float(D[0][j]),
            "text": metadata["texts"][I[0][j]],
            "source": metadata["sources"][I[0][j]],
        }
        for j in range(len(I[0]))
        if I[0][j] >= 0
    ]


@st.cache_data(show_spinner=False)
def build_index_from_pdfs(
    pdf_paths: Tuple,
    system: str,
    chunk_size: int = 1000,
    overlap: int = 200,
):
    res = create_faiss_resource(system)
    local_texts, local_sources = [], []
    for p in pdf_paths:
        text = extract_text_from_pdf(p)
        if not text.strip():
            try:
                with open(p, "rb") as f:
                    raw = f.read(3_000_000)
                st.info(f"Attempting OCR on {os.path.basename(p)} (image-based PDF)…")
                text = extract_text_from_image(raw)
            except Exception:
                pass
        for chunk in chunk_text(text, chunk_size, overlap):
            local_texts.append(chunk)
            local_sources.append(os.path.basename(p))
    if not local_texts:
        raise ValueError("No text extracted from any of the provided PDFs.")
    add_to_faiss(res["index"], res["metadata"], embed_texts(local_texts), local_texts, local_sources)
    return {"count": len(local_texts), "docs": len(pdf_paths)}


# ─── Merge helper ──────────────────────────────────────────────────────────────
def get_all_pdfs(system: str) -> List[Dict]:
    """Merge base catalogue + discovered + user-added PDFs, de-duplicated by URL."""
    base       = WARHAMMER_PDFS.get(system, [])
    discovered = st.session_state["discovered_pdfs"].get(system, [])
    custom     = st.session_state["custom_pdfs"].get(system, [])
    seen, out = set(), []
    for item in base + discovered + custom:
        if item["url"] not in seen:
            seen.add(item["url"])
            out.append(item)
    return out


# ─── Session state initialisation ─────────────────────────────────────────────
_defaults = {
    "active_ruleset":  None,
    "last_indexed":    None,
    "chat_history":    [],        # [{"role": "user"|"assistant", "content": str}]
    "custom_pdfs":     {},        # {system: [{title, url, description, category}]}
    "discovered_pdfs": {},        # {system: [{title, url, description, category}]}
    "image_text":      "",
    "current_system":  list(WARHAMMER_PDFS.keys())[0],
}
for key, default in _defaults.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ─── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Configuration")
system = st.sidebar.selectbox("Select ruleset", list(WARHAMMER_PDFS.keys()))

if st.session_state["active_ruleset"] != system:
    st.session_state["active_ruleset"] = system
    st.session_state["last_indexed"] = None
    st.info(f"Switched to **{system}**. Fetch & index PDFs or query an existing index.")

max_pdfs   = st.sidebar.slider("Max PDFs to fetch", 1, 20, 5)
chunk_size = st.sidebar.number_input("Chunk token size", 200, 2000, 1000, 100)
overlap    = st.sidebar.number_input("Chunk overlap",     0,   500,  200,  50)

st.sidebar.markdown("---")

# ── Auto-discover new PDFs ─────────────────────────────────────────────────────
st.sidebar.markdown("**Auto-discover new PDFs**")
if st.sidebar.button("🔍 Check for New PDFs"):
    with st.spinner(f"Scanning community site for {system} PDFs…"):
        found = discover_pdfs_from_community(system)
    existing_urls = {p["url"] for p in st.session_state["discovered_pdfs"].get(system, [])}
    new_items = [p for p in found if p["url"] not in existing_urls]
    if new_items:
        st.session_state["discovered_pdfs"].setdefault(system, []).extend(new_items)
        st.sidebar.success(f"Found {len(new_items)} new PDF(s) — they appear in the PDF list below.")
    else:
        st.sidebar.info(
            "No new PDFs found in the static HTML (the downloads page is JavaScript-rendered).\n\n"
            f"Browse manually: {COMMUNITY_DOWNLOAD_URLS[system]}\n\n"
            "Paste any PDF links you find into **Add Custom PDF URL** below."
        )

# ── Add custom PDF URL ─────────────────────────────────────────────────────────
st.sidebar.markdown("**Add Custom PDF URL**")
custom_url   = st.sidebar.text_input("PDF URL", key="custom_url_input", label_visibility="collapsed",
                                     placeholder="https://assets.warhammer-community.com/…pdf")
custom_title = st.sidebar.text_input("Title (optional)", key="custom_title_input")
if st.sidebar.button("➕ Add PDF"):
    if custom_url:
        title = custom_title.strip() or os.path.basename(custom_url.split("?")[0])
        entry = {
            "title": title,
            "url": custom_url.strip(),
            "description": "User-added PDF",
            "category": "Custom",
        }
        st.session_state["custom_pdfs"].setdefault(system, []).append(entry)
        st.sidebar.success(f"Added: **{title}**")
    else:
        st.sidebar.warning("Please enter a URL.")

st.sidebar.markdown("---")

# ── Fetch & Index ──────────────────────────────────────────────────────────────
if st.sidebar.button("📥 Fetch & Index PDFs"):
    all_pdfs = get_all_pdfs(system)
    selected = all_pdfs[:max_pdfs]

    st.info(f"Preparing to download **{len(selected)}** PDF(s) for **{system}**.")
    for item in selected:
        st.markdown(f"- **{item['title']}** — {item['description']}")

    tempdir  = tempfile.mkdtemp(prefix="rules_")
    progress = st.progress(0)
    status   = st.empty()

    def prog_cb(count, total, name=""):
        progress.progress(int(count / total * 100))
        status.text(f"Downloaded {count}/{total}: {name}")

    with st.spinner("Downloading PDFs…"):
        downloaded = download_pdfs(
            [(p["url"], p["title"]) for p in selected], tempdir, prog_cb
        )

    if not downloaded:
        st.error("No PDFs downloaded. Check URLs and network access.")
    else:
        status.text("Building FAISS index…")
        try:
            res = build_index_from_pdfs(
                tuple(path for path, _ in downloaded),
                system,
                chunk_size=chunk_size,
                overlap=overlap,
            )
            st.session_state["last_indexed"] = f"{len(downloaded)} PDFs · {res['count']} chunks"
            st.session_state["current_system"] = system
            st.success(f"Indexed **{res['docs']}** PDFs → **{res['count']}** chunks.")
        except APIError as e:
            if getattr(e, "code", None) == "insufficient_quota":
                st.error("OpenAI quota exceeded. Check https://platform.openai.com/account/usage")
            else:
                st.error(f"OpenAI API error during indexing: {e}")
        except Exception as e:
            st.error(f"Index build failed: {e}")
    try:
        shutil.rmtree(tempdir)
    except Exception:
        pass

st.sidebar.markdown("---")
st.sidebar.markdown("**Index status**")
st.sidebar.write(st.session_state.get("last_indexed") or "_No index built yet_")

if st.sidebar.button("📋 Show indexed sources"):
    store = get_faiss_store()
    if system in store:
        sources = list(dict.fromkeys(store[system]["metadata"]["sources"]))
        st.sidebar.json(sources)
    else:
        st.sidebar.info(f"No index for **{system}**.")

if st.sidebar.button("🗑️ Clear index + chat"):
    store = get_faiss_store()
    if system in store:
        del store[system]
    build_index_from_pdfs.clear()
    st.session_state["last_indexed"]  = None
    st.session_state["chat_history"]  = []
    st.session_state["image_text"]    = ""
    st.sidebar.success(f"Cleared index and conversation for **{system}**.")

# ─── Main area ─────────────────────────────────────────────────────────────────
st.header("Ask the Rules Assistant")
st.caption(
    f"📘 Active ruleset: **{system}** · "
    f"[Official downloads]({COMMUNITY_DOWNLOAD_URLS[system]}) · "
    f"[Munitorum Field Manual (web)](https://www.warhammer-community.com/en-gb/munitorum-field-manual/)"
)

# ── Conversation history ───────────────────────────────────────────────────────
if st.session_state["chat_history"]:
    st.markdown("#### Conversation history")
    for msg in st.session_state["chat_history"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    if st.button("🗑️ Clear conversation"):
        st.session_state["chat_history"] = []
        st.rerun()
    st.markdown("---")

# ── OCR image upload ───────────────────────────────────────────────────────────
with st.expander("📷 Upload image for OCR (optional)"):
    uploaded_image = st.file_uploader(
        "Upload a PNG/JPG screenshot of a rules card",
        type=["png", "jpg", "jpeg", "tiff"],
        label_visibility="collapsed",
    )
    if uploaded_image is not None:
        ocr_text = extract_text_from_image(uploaded_image.read())
        if ocr_text:
            st.text_area("Extracted text:", value=ocr_text, height=120, disabled=True)
            if st.button("Add image text to next query context"):
                st.session_state["image_text"] = ocr_text
                st.success("Image text will be included in your next question.")
        else:
            st.warning("No text could be extracted from the image.")

# ── Question input ─────────────────────────────────────────────────────────────
question = st.text_input(
    "Enter your rules question:",
    placeholder="e.g. Can a unit that Advanced charge in the same turn?",
)

if st.button("Ask ⚙️", type="primary") and question:
    active_system = st.session_state.get("current_system", system)
    res   = create_faiss_resource(active_system)
    index = res["index"]
    meta  = res["metadata"]

    if index.ntotal == 0:
        st.error("Index is empty — use **Fetch & Index PDFs** in the sidebar first.")
    else:
        with st.spinner("Searching rules and generating answer…"):
            try:
                q_emb   = embed_texts([question])[0]
                hits    = search_faiss(index, meta, q_emb, top_k=6)
                context = "\n\n---\n\n".join(h["text"] for h in hits)

                if st.session_state.get("image_text"):
                    context += f"\n\n---\n\nImage OCR context:\n{st.session_state['image_text']}"

                system_prompt = st.secrets.get(
                    "SYSTEM_PROMPT",
                    (
                        "You are an expert Warhammer rules assistant. "
                        "Answer concisely and accurately, citing the source document when possible. "
                        "If the answer cannot be determined from the provided context, say so clearly "
                        "rather than speculating. Do not invent rules."
                    ),
                )

                # Include last 3 Q&A pairs (6 messages) as conversation memory
                history_msgs = st.session_state["chat_history"][-6:]
                messages = (
                    [{"role": "system", "content": system_prompt}]
                    + history_msgs
                    + [{
                        "role": "user",
                        "content": (
                            f"Use only the context below to answer the question.\n\n"
                            f"Context:\n{context}\n\n"
                            f"Question: {question}"
                        ),
                    }]
                )

                chat_resp = client.chat.completions.create(
                    model=DEFAULT_MODEL, messages=messages, max_tokens=600, temperature=0.0
                )
                answer = chat_resp.choices[0].message.content.strip()

                # Persist to conversation history
                st.session_state["chat_history"].append({"role": "user",      "content": question})
                st.session_state["chat_history"].append({"role": "assistant",  "content": answer})
                # Clear one-shot image context after use
                st.session_state["image_text"] = ""

                st.markdown("### Answer")
                with st.chat_message("assistant"):
                    st.markdown(answer)

                with st.expander("📄 Retrieved source snippets"):
                    for h in hits:
                        st.write(f"**{h['source']}** — score: {h['score']:.3f}")
                        st.write(h["text"][:800] + ("…" if len(h["text"]) > 800 else ""))
                        st.markdown("---")

            except APIError as e:
                code = getattr(e, "code", None)
                if code == "insufficient_quota":
                    st.error(
                        "OpenAI quota exceeded. "
                        "Check [usage dashboard](https://platform.openai.com/account/usage)."
                    )
                elif code == 429:
                    st.error("Rate limit hit. Please wait a moment and try again.")
                else:
                    st.error(f"OpenAI API error: {e}")
            except Exception as e:
                st.error(f"Failed to generate answer: {e}")

# ─── Browse available PDFs ─────────────────────────────────────────────────────
with st.expander("📚 Browse available PDFs for this ruleset"):
    all_pdfs   = get_all_pdfs(system)
    categories = list(dict.fromkeys(p.get("category", "Other") for p in all_pdfs))
    for cat in categories:
        st.markdown(f"**{cat}**")
        for p in all_pdfs:
            if p.get("category", "Other") == cat:
                st.markdown(f"- [{p['title']}]({p['url']}) — _{p['description']}_")
    if not all_pdfs:
        st.info("No PDFs available. Use **Check for New PDFs** or **Add Custom PDF URL** in the sidebar.")

st.markdown("---")
st.caption(
    "💾 Powered by OpenAI embeddings · FAISS · Streamlit · "
    "© Games Workshop data used for personal reference only. "
    "Munitorum Field Manual is a live web app — points values are not indexed here."
)
