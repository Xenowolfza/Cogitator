# Warhammer tools

This repository contains two apps:

1. **Strike Force** (`web/`) — unofficial Warhammer 40,000 army list builder with printable datasheet cards.
2. **Rules Assistant** (`app.py`) — Streamlit RAG helper that indexes official PDF rules.

---

## Strike Force (army builder)

Unofficial fan tool. Not affiliated with Games Workshop. Profiles are compact mechanical summaries you can override from your own books.

```bash
cd web
npm install
npm run dev
```

Open http://127.0.0.1:5173

- Build a list: faction, detachment, points cap, units, wargear, enhancements
- **Datasheets** tab: one 10th-edition-style data card per unit (chosen loadout only)
- Print pack (browser print → PDF) and JSON export/import
- Lists persist in the browser (`localStorage`)

---

## Warhammer Rules Assistant — v2

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
