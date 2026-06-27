# rules_updater.py
"""
Utilities for discovering and downloading Warhammer Community PDFs.

Note: the main downloads pages (warhammer-community.com/en-gb/downloads/*)
are JavaScript-rendered, so BeautifulSoup only captures PDF links present
in the static HTML. Use the Streamlit "Add Custom PDF URL" sidebar input
to add links found by browsing the downloads page manually.
"""

import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from typing import List, Dict

COMMUNITY_DOWNLOAD_URLS = {
    "40K":          "https://www.warhammer-community.com/en-gb/downloads/warhammer-40000/",
    "Age of Sigmar":"https://www.warhammer-community.com/en-gb/downloads/warhammer-age-of-sigmar/",
    "Kill Team":    "https://www.warhammer-community.com/en-gb/downloads/kill-team/",
}

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}


def fetch_pdf_links(page_url: str) -> List[str]:
    """
    Fetch a page and return all PDF hrefs found in the static HTML.
    May return an empty list for JS-rendered pages.
    """
    r = requests.get(page_url, timeout=20, headers=_HEADERS)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.lower().endswith(".pdf"):
            links.append(urljoin(page_url, href))
    return list(dict.fromkeys(links))


def discover_pdfs(system: str) -> List[Dict]:
    """
    Return a list of {title, url, description, category} dicts for any PDFs
    discoverable from the static HTML of the community downloads page.
    """
    page_url = COMMUNITY_DOWNLOAD_URLS.get(system, "")
    if not page_url:
        return []
    results = []
    try:
        links = fetch_pdf_links(page_url)
        for link in links:
            title = os.path.basename(link.split("?")[0]).replace("_", " ").replace("-", " ")
            results.append({
                "title": title,
                "url": link,
                "description": f"Auto-discovered from {page_url}",
                "category": "Discovered",
            })
    except Exception as exc:
        print(f"[rules_updater] Discovery failed for {system}: {exc}")
    return results


def download_pdfs(links: List[str], target_dir: str) -> List[str]:
    """Download a list of PDF URLs into target_dir; skip already-present files."""
    os.makedirs(target_dir, exist_ok=True)
    downloaded = []
    for link in links:
        filename = os.path.join(target_dir, os.path.basename(link.split("?")[0]))
        if os.path.exists(filename):
            continue
        try:
            r = requests.get(link, timeout=30, headers=_HEADERS)
            r.raise_for_status()
            with open(filename, "wb") as f:
                f.write(r.content)
            downloaded.append(filename)
        except Exception as exc:
            print(f"[rules_updater] Failed to download {link}: {exc}")
    return downloaded
