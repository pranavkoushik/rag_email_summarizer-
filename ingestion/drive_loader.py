"""
Google Drive Loader — downloads all files from a public Google Drive folder.
Uses gdown which handles public shared folders without needing API credentials.
"""

import os
import gdown


def extract_folder_id(drive_url: str) -> str:
    """Extract folder ID from a Google Drive folder URL."""
    if "folders/" in drive_url:
        folder_id = drive_url.split("folders/")[1].split("?")[0]
        return folder_id
    return drive_url  # assume raw ID was passed


def download_drive_folder(drive_url: str, output_dir: str = "data") -> str:
    """
    Download all files from a public Google Drive folder.

    Args:
        drive_url: Full Google Drive folder URL or folder ID.
        output_dir: Local directory to save downloaded files.

    Returns:
        Path to the output directory containing downloaded files.
    """
    os.makedirs(output_dir, exist_ok=True)

    folder_id = extract_folder_id(drive_url)
    url = f"https://drive.google.com/drive/folders/{folder_id}"

    print(f"Downloading files from Google Drive folder: {folder_id}")
    gdown.download_folder(url=url, output=output_dir, quiet=False)
    print(f"Downloaded files to: {output_dir}")

    return output_dir


if __name__ == "__main__":
    DRIVE_URL = "https://drive.google.com/drive/folders/1ASVgNJwn5_IcvXpF2SxyfYa8uO5oyozG?usp=sharing"
    download_drive_folder(DRIVE_URL)
