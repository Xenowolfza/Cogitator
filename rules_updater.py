# rules_updater.py
import os, requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

WARHAMMER_URLS = {
    "40K": "https://www.warhammer-community.com/en-gb/downloads/warhammer-40000/",
    "Age of Sigmar": "https://www.warhammer-community.com/en-gb/downloads/warhammer-age-of-sigmar/",
    "Kill Team": "https://www.warhammer-community.com/en-gb/downloads/kill-team/"
}

def fetch_pdf_links(page_url):
    r = requests.get(page_url, timeout=20)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.lower().endswith(".pdf"):
            links.append(urljoin(page_url, href))
    return list(dict.fromkeys(links))

def download_pdfs(links, target_dir):
    os.makedirs(target_dir, exist_ok=True)
    downloaded = []
    for link in links:
        filename = os.path.join(target_dir, os.path.basename(link.split("?")[0]))
        if os.path.exists(filename):
            continue
        r = requests.get(link, timeout=30)
        r.raise_for_status()
        with open(filename, "wb") as f:
            f.write(r.content)
        downloaded.append(filename)
    return downloaded
