"""Download and unpack the CUAD dataset (or subset) for benchmark testing.

Usage:
    python -m scripts.setup_cuad [--full]
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import urllib.request
import zipfile
from pathlib import Path

CUAD_REPO_URL = "https://raw.githubusercontent.com/The-Atticus-Project/cuad/main"
DATA_ZIP_URL = f"{CUAD_REPO_URL}/data.zip"
CATEGORIES_CSV_URL = f"{CUAD_REPO_URL}/category_descriptions.csv"

TARGET_DIR = Path(__file__).resolve().parent.parent / "data" / "cuad"


def download_cuad(full: bool = False) -> None:
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = TARGET_DIR / "category_descriptions.csv"

    # Step 1: Download Category Descriptions (41 categories)
    print("Step 1/2: Fetching CUAD 41 category descriptions...")
    try:
        req = urllib.request.Request(CATEGORIES_CSV_URL, headers={"User-Agent": "Legal-Triage-Benchmarker/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            content = resp.read()
            csv_path.write_bytes(content)
        print(f" Saved category descriptions to {csv_path} ({len(content)} bytes)")
    except Exception as e:
        print(f" Warning: Could not fetch category descriptions from remote: {e}")

    # Step 2: Download data.zip
    zip_target = TARGET_DIR / "data.zip"
    if not zip_target.exists():
        print(f"Step 2/2: Downloading CUAD data.zip (~18 MB) from {DATA_ZIP_URL}...")
        try:
            req = urllib.request.Request(DATA_ZIP_URL, headers={"User-Agent": "Legal-Triage-Benchmarker/1.0"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = resp.read()
                zip_target.write_bytes(data)
            print(f" Downloaded data.zip ({len(data):,} bytes)")
        except Exception as e:
            print(f" Download failed: {e}")
            print("Please download data.zip manually from https://github.com/The-Atticus-Project/cuad and place in backend/data/cuad/data.zip")
            return
    else:
        print(f"Found existing data.zip at {zip_target}")

    # Step 3: Unpack
    print("Extracting files from data.zip...")
    with zipfile.ZipFile(zip_target, "r") as z:
        for name in z.namelist():
            print(f" Extracting {name}...")
            z.extract(name, TARGET_DIR)


    print(f"CUAD setup complete! Contracts and labels available in {TARGET_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and setup CUAD dataset")
    parser.add_argument("--full", action="store_true", help="Extract all 510 contracts (takes longer)")
    args = parser.parse_args()
    download_cuad(full=args.full)
