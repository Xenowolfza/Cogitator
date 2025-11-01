# Warhammer Rules Assistant — v2

This Streamlit app fetches official Warhammer (40K, Age of Sigmar, Kill Team) PDF rules
from the Warhammer Community website, indexes them with OpenAI embeddings, and
provides a RAG-powered rules assistant.

## Quickstart (Local)

1. Create a virtual environment and install requirements:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. Add your Streamlit secrets at `.streamlit/secrets.toml` (or via Streamlit Cloud):
```
OPENAI_API_KEY = "sk-..."
APP_ACCESS_PIN = "forgeworld"
DEFAULT_MODEL = "gpt-3.5-turbo"
ASSISTANT_NAME = "The Omnissiah's Cogitator"
WELCOME_MESSAGE = "Greetings, Commander..."
```

3. Run locally:
```bash
streamlit run app.py
```

## Deploy to Streamlit Cloud

1. Push this repo to GitHub.
2. In Streamlit Cloud, create a new app linked to this repo and branch, main file `app.py`.
3. Add the secrets in Advanced Settings.

## Notes

- PDFs are downloaded dynamically (not stored in the repo).
- Indexing uses OpenAI embeddings — costs apply based on OpenAI pricing.
- This is a starter project. For production use, consider caching indices and adding rate limits.
