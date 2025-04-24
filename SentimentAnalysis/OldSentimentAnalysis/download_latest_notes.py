#!/usr/bin/env python3
"""
Download the *latest* SEC Financial‑Statement‑and‑Notes monthly data set,
unpack it under  SentimentAnalysis/SECData/<YYYY_MM>_notes/.

Usage:
    python download_latest_notes.py
"""
import re
import sys
import zipfile
import requests
from pathlib import Path
from tempfile import NamedTemporaryFile
from datetime import datetime
from bs4 import BeautifulSoup

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "SECData"
SEC_PAGE = ("https://www.sec.gov/data-research/"
              "sec-markets-data/financial-statement-notes-data-sets")
UA_HEADER  = {
    "User-Agent": "Your Name your.email@example.com",
    "Accept-Encoding": "gzip, deflate",
    "Host": "www.sec.gov",
}

# parse html page, return first monthly ZIP link
def find_latest_zip_url() -> str:
    print("[*] Fetching data‑set index page...")
    resp = requests.get(SEC_PAGE, headers=UA_HEADER, timeout=30)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    pattern = re.compile(r"\d{4}_\d{2}_notes\.zip$", re.IGNORECASE)

    for a in soup.find_all("a", href=True):
        href = a["href"]
        if pattern.search(href):
            url = href if href.startswith("http") else f"https://www.sec.gov{href}"
            print(f"[+] Found latest dataset link: {url}")
            return url

    raise RuntimeError("Could not locate a *_notes.zip link on the SEC page.")

# download the ZIP to a temp file and return its Path.
def download_zip(url: str) -> Path:
    print(f"[*] Downloading {url} ...")
    tag = re.search(r"(\d{4}_\d{2})_notes\.zip", url).group(1)
    with requests.get(url, headers=UA_HEADER, stream=True, timeout=60) as r:
        r.raise_for_status()
        tmp = NamedTemporaryFile(
            delete=False, suffix=f"_{tag}_notes.zip"
        )
        for chunk in r.iter_content(chunk_size=2 ** 20):
            tmp.write(chunk)
    print(f"[+] Saved ZIP → {tmp.name}")
    return Path(tmp.name)

# unzip into SECData/YYYY_MM_notes/
def unzip_to_folder(zip_path: Path):
    m = re.search(r"(\d{4}_\d{2})_notes\.zip", zip_path.name)
    if not m:
        raise ValueError(f"Unexpected zip filename: {zip_path.name}")
    folder_name = f"{m.group(1)}_notes"
    dest_dir = DATA_DIR / folder_name
    dest_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest_dir)
    print(f"[+] Unzipped contents → {dest_dir.resolve()}")

def main():
    try:
        url = find_latest_zip_url()
        zip_path = download_zip(url)
        unzip_to_folder(zip_path)
        print("Completed!")
    except Exception as e:
        print(f"[!] Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
