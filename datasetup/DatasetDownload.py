import zipfile
from pathlib import Path
from urllib.parse import urlparse

import requests
from tqdm import tqdm


def download_and_extract_zip(url: str, destination_folder: Path):
    # Step 1: Download the zip file with progress bar
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Check for request errors (e.g., 404 or 500)

        # Extract filename from URL using pathlib
        parsed_url = urlparse(url)
        zip_file_name = Path(parsed_url.path).name  # Get the file name from the URL using pathlib

        if not zip_file_name:
            print("Error: URL does not contain a file name.")
            return False

        # Define the local path for the downloaded file
        destination_folder = Path(destination_folder)
        zip_file_path = destination_folder / zip_file_name

        # Get the total size of the file from the response headers
        total_size = int(response.headers.get('content-length', 0))

        # Create a tqdm progress bar to show download progress
        with open(zip_file_path, 'wb') as zip_file, tqdm(
                desc="Downloading",
                total=total_size,
                unit="B",
                unit_scale=True,
                ncols=100
        ) as bar:
            # Download in chunks and update the progress bar
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    zip_file.write(chunk)
                    bar.update(len(chunk))

        print(f"\nDownloaded the zip file to {zip_file_path}")

    except requests.exceptions.RequestException as e:
        print(f"Failed to download the file: {e}")
        return False

    # Step 2: Unzip the file
    try:
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(destination_folder)
        print(f"Extracted contents to {destination_folder}")

    except zipfile.BadZipFile:
        print("The downloaded file is not a valid zip file.")
        return False

    # Step 3: Delete the original ZIP file if extraction was successful
    try:
        zip_file_path.unlink()  # pathlib method to delete the file
        print(f"Successfully deleted the original zip file: {zip_file_path}")
    except OSError as e:
        print(f"Error deleting the file: {e}")
        return False

    return True


def download_dataset(url: str, destination_folder: Path):
    destination_folder.mkdir(exist_ok=True, parents=True)

    if download_and_extract_zip(url, destination_folder):
        print("Download, extraction, and cleanup were successful.")
    else:
        print("There was an error in the process.")
    print()
