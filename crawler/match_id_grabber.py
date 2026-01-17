import requests
from bs4 import BeautifulSoup
import re
import time
import sys

# === CONFIGURATION for 2024/25===
SOURCE_URL = "https://nbl.basketball/zapasy?y=2024&p1=0&c=0&d_od=&d_do=&k=0"
TARGET_FILE = "input.txt"
OUTPUT_FOLDER_NAME = "nbl_season_2024"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}


def get_match_urls(url):
    print(f"Fetching match list from NBL website: {url}")
    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        links = set()
        for a in soup.find_all('a', href=True):
            if re.search(r'/zapas/\d+', a['href']):
                clean_path = re.search(r'/zapas/\d+', a['href']).group(0)
                links.add(f"https://nbl.basketball{clean_path}")

        sorted_links = sorted(list(links))
        print(f"Found {len(sorted_links)} match pages on the web.")
        return sorted_links
    except Exception as e:
        print(f"❌ Error fetching list: {e}")
        return []


def extract_fiba_id(match_url):
    try:
        response = requests.get(match_url, headers=HEADERS, timeout=10)
        if response.status_code != 200: return None

        soup = BeautifulSoup(response.text, 'html.parser')

        for a in soup.find_all('a', href=True):
            href = a['href']
            if "fibalivestats" in href or "geniussports" in href:
                # Pattern 1: .../webcast/CODE/12345/
                match_path = re.search(r'webcast/[^/]+/(\d+)', href)
                if match_path: return match_path.group(1)

                # Pattern 2: ...?id=12345
                match_param = re.search(r'id=(\d+)', href)
                if match_param: return match_param.group(1)
        return None
    except Exception:
        return None


if __name__ == "__main__":
    match_urls = get_match_urls(SOURCE_URL)

    if not match_urls:
        print("❌ No matches found.")
        sys.exit(1)

    fiba_ids = []
    total = len(match_urls)
    print(f"\nScanning {total} matches for FIBA LiveStats IDs...\n")

    for i, url in enumerate(match_urls):
        cz_web_id = url.split('/')[-1]

        fiba_id = extract_fiba_id(url)

        if fiba_id:
            fiba_ids.append(fiba_id)
            print(f"[{i + 1}/{total}] CZ Web ID {cz_web_id} -> LiveStats ID: {fiba_id}")
        else:
            print(f"[{i + 1}/{total}] CZ Web ID {cz_web_id} -> ❌ No LiveStats link found")

        time.sleep(0.2)

    if fiba_ids:
        unique_ids = sorted(list(set(fiba_ids)), key=int)

        with open(TARGET_FILE, "w", encoding="utf-8") as f:
            f.write(f"{OUTPUT_FOLDER_NAME}\n\n")
            f.write(", ".join(unique_ids))

        print(f"\nSUCCESS! Saved {len(unique_ids)} LiveStats IDs to '{TARGET_FILE}'.")
        print(f"Target folder set to: {OUTPUT_FOLDER_NAME}")
    else:
        print("\n❌ Failed to extract any IDs.")