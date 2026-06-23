"""
Download and extract the NASA CMAPSS dataset.
Run this script once before running the main experiment.
"""

import urllib.request
import zipfile
import os
from pathlib import Path

DATA_DIR = Path("cmapss_data")

# Direct download URL (NASA Open Data Portal via archive)
DOWNLOAD_URL = "https://data.nasa.gov/api/views/ff5v-kuh6/rows.csv?accessType=DOWNLOAD"

# Kaggle alternative (requires kaggle CLI):
# kaggle datasets download -d behrad3d/nasa-cmaps

def download_cmapss():
    if DATA_DIR.exists() and any(DATA_DIR.glob("train_FD001.txt")):
        print("CMAPSS dataset already present.")
        return

    DATA_DIR.mkdir(exist_ok=True)

    # Try direct download from a reliable mirror
    zip_path = DATA_DIR / "cmapss.zip"

    print("Downloading NASA CMAPSS dataset...")
    try:
        url = "https://data.nasa.gov/api/views/ff5v-kuh6/rows.zip?accessType=DOWNLOAD"
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(DATA_DIR)
        zip_path.unlink()
        print(f"Dataset extracted to: {DATA_DIR.absolute()}")
    except Exception as e:
        print(f"Automatic download failed: {e}")
        print()
        print("Please download manually:")
        print("1. Go to: https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data")
        print("2. Download the ZIP file")
        print(f"3. Extract contents to: {DATA_DIR.absolute()}/")
        print()
        print("Expected files:")
        for name in ["train_FD001.txt", "test_FD001.txt", "RUL_FD001.txt",
                     "train_FD002.txt", "test_FD002.txt", "RUL_FD002.txt",
                     "train_FD003.txt", "test_FD003.txt", "RUL_FD003.txt",
                     "train_FD004.txt", "test_FD004.txt", "RUL_FD004.txt"]:
            print(f"   cmapss_data/{name}")

if __name__ == "__main__":
    download_cmapss()
