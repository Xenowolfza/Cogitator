import streamlit as st
import os
import tempfile
import shutil
from typing import List
from urllib.parse import urljoin

# LangChain & OpenAI/old
#from langchain.chains import RetrievalQA
#from langchain.chat_models import ChatOpenAI
#from langchain.document_loaders import PyPDFLoader
#from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain.embeddings import OpenAIEmbeddings
#from langchain.vectorstores import FAISS



# ---- LangChain Modular Imports (modern structure) ----
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
try:
    # LangChain <=0.3.9 style
    from langchain.chains.retrieval import create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain
except ImportError:
    # LangChain >=0.3.10+
    from langchain_community.chains.retrieval import create_retrieval_chain
    from langchain_community.chains.combine_documents import create_stuff_documents_chain








# Utils for fetching
import requests
from bs4 import BeautifulSoup

st.set_page_config(page_title="Warhammer Rules Assistant", layout="wide")
st.title(st.secrets.get("ASSISTANT_NAME", "Warhammer Rules Assistant"))

# Configuration
DEFAULT_MODEL = st.secrets.get("DEFAULT_MODEL", "gpt-3.5-turbo")
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.warning("OpenAI API key not found. Add OPENAI_API_KEY to Streamlit Secrets or environment to enable cloud LLM.")
    LLM_AVAILABLE = False
else:
    LLM_AVAILABLE = True
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

APP_PIN = st.secrets.get("APP_ACCESS_PIN", None)
if APP_PIN:
    entered = st.sidebar.text_input("Access PIN", type="password")
    if entered != APP_PIN:
        st.stop()

WARHAMMER_URLS = {
    "40K": "https://www.warhammer-community.com/en-gb/downloads/warhammer-40000/",
    "Age of Sigmar": "https://www.warhammer-community.com/en-gb/downloads/warhammer-age-of-sigmar/",
    "Kill Team": "https://www.warhammer-community.com/en-gb/downloads/kill-team/"
}

st.sidebar.header("Configuration")
system = st.sidebar.selectbox("Select ruleset", list(WARHAMMER_URLS.keys()))
action = st.sidebar.radio("Action", ["Fetch & Index Official PDFs", "Upload PDFs and Index"])

if "vectorstore" not in st.session_state:
    st.session_state["vectorstore"] = None
if "last_indexed" not in st.session_state:
    st.session_state["last_indexed"] = None

def fetch_pdf_links(url: str) -> List[str]:
    resp = requests.get(url, timeout=20)
    soup = BeautifulSoup(resp.text, "html.parser")
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.lower().endswith(".pdf"):
            full = urljoin(url, href)
            if full not in links:
                links.append(full)
    return links

def download_pdfs(links: List[str], target_dir: str) -> List[str]:
    os.makedirs(target_dir, exist_ok=True)
    downloaded = []
    for link in links:
        filename = os.path.join(target_dir, os.path.basename(link.split("?")[0]))
        if os.path.exists(filename):
            continue
        try:
            r = requests.get(link, timeout=30)
            r.raise_for_status()
            with open(filename, "wb") as f:
                f.write(r.content)
            downloaded.append(filename)
        except Exception as e:
            st.warning(f"Failed to download {link}: {e}")
    return downloaded

def extract_and_index(pdf_paths: List[str], openai_api_key: str, temp_dir: str):
    docs = []
    for p in pdf_paths:
        try:
            loader = PyPDFLoader(p)
            loaded = loader.load()
            docs.extend(loaded)
        except Exception as e:
            st.warning(f"Failed to load {p}: {e}")
    if not docs:
        st.error("No documents loaded to index.")
        return None
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = splitter.split_documents(docs)
    os.environ["OPENAI_API_KEY"] = openai_api_key
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    return vectorstore

if action == "Fetch & Index Official PDFs":
    st.sidebar.markdown("This will fetch official PDFs from Games Workshop and build a temporary index.")
    if st.sidebar.button("Fetch & Index"):
        with st.spinner("Fetching PDF links..."):
            try:
                links = fetch_pdf_links(WARHAMMER_URLS[system])
            except Exception as e:
                st.error(f"Failed to fetch links from {WARHAMMER_URLS[system]}: {e}")
                links = []
        st.sidebar.info(f"Found {len(links)} PDF links.")
        temp_download_dir = os.path.join("tmp_rules", system.replace(" ", "_"))
        with st.spinner("Downloading PDFs..."):
            downloaded = download_pdfs(links, temp_download_dir)
        all_pdf_files = [os.path.join(temp_download_dir, f) for f in os.listdir(temp_download_dir)] if os.path.exists(temp_download_dir) else []
        st.sidebar.success(f"Downloaded {len(all_pdf_files)} files (cached).")
        if LLM_AVAILABLE:
            with st.spinner("Indexing documents (this may take a few minutes)..."):
                try:
                    vectorstore = extract_and_index(all_pdf_files, OPENAI_API_KEY, temp_download_dir)
                    st.session_state["vectorstore"] = vectorstore
                    st.session_state["last_indexed"] = f"{len(all_pdf_files)} files"
                    st.success("Indexing complete. Ready for questions.")
                except Exception as e:
                    st.error(f"Indexing failed: {e}")
        else:
            st.info("OpenAI key not available — indexing will still occur but LLM answers won't be generated here.")
            try:
                vectorstore = extract_and_index(all_pdf_files, OPENAI_API_KEY or "", temp_download_dir)
                st.session_state["vectorstore"] = vectorstore
                st.session_state["last_indexed"] = f"{len(all_pdf_files)} files"
                st.success("Indexing complete (embeddings attempted).")
            except Exception as e:
                st.warning(f"Indexing skipped due to missing key or error: {e}")

if action == "Upload PDFs and Index":
    uploaded = st.sidebar.file_uploader("Upload PDFs (multiple)", accept_multiple_files=True, type=["pdf"])
    if uploaded and st.sidebar.button("Index uploaded PDFs"):
        tmpdir = tempfile.mkdtemp()
        paths = []
        for up in uploaded:
            p = os.path.join(tmpdir, up.name)
            with open(p, "wb") as f:
                f.write(up.getbuffer())
            paths.append(p)
        with st.spinner("Indexing uploaded PDFs..."):
            try:
                vectorstore = extract_and_index(paths, OPENAI_API_KEY or "", tmpdir)
                st.session_state["vectorstore"] = vectorstore
                st.session_state["last_indexed"] = f"{len(paths)} uploaded PDFs"
                st.success("Index created from uploaded PDFs.")
            except Exception as e:
                st.error(f"Indexing failed: {e}")
        shutil.rmtree(tmpdir, ignore_errors=True)

st.header("Ask the Rules Assistant")
st.markdown(st.secrets.get("WELCOME_MESSAGE", "Ask a rules question about 40K, Age of Sigmar, or Kill Team."))

question = st.text_input("Enter your question here (e.g., 'Can a unit that Advanced charge?')")

col1, col2 = st.columns([3,1])
with col2:
    if st.button("Clear index"):
        st.session_state["vectorstore"] = None
        st.session_state["last_indexed"] = None
        st.success("Index cleared.")
# --- ASK LOGIC ---
if st.button("Ask") and question:
    if st.session_state.get("vectorstore") is None:
        st.error("No index available. Please fetch & index official PDFs or upload your own PDFs first.")
    else:
        retriever = st.session_state["vectorstore"].as_retriever(search_kwargs={"k": 4})
        if not LLM_AVAILABLE:
            st.error("OpenAI API key is not configured; can't generate answers. Add OPENAI_API_KEY to secrets.")
        else:
            with st.spinner("Generating answer..."):
                try:
                    # Initialize model
                    llm = ChatOpenAI(model_name=DEFAULT_MODEL, temperature=0)

                    # Create new-style retrieval chain
                    prompt = ChatPromptTemplate.from_template(
                        "You are a Warhammer rules assistant. "
                        "Answer this question using the context from the documents.\n\n"
                        "Context:\n{context}\n\nQuestion: {input}"
                    )

                    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
                    retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

                    # Run the chain
                    result = retrieval_chain.invoke({"input": question})
                    answer = result.get("answer", "No answer found.")
                    sources = result.get("context", [])

                    # Display
                    st.markdown("### Answer")
                    st.write(answer)

                    if sources:
                        st.markdown("### Source snippets")
                        for i, src in enumerate(sources):
                            st.markdown(f"**Source {i+1}:**")
                            snippet = str(src)
                            st.write(snippet[:1000] + ("..." if len(snippet) > 1000 else ""))

                except Exception as e:
                    st.error(f"Error during answer generation: {e}")
