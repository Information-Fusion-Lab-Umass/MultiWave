import requests
from tqdm import tqdm
import math
import zipfile
import os

# Download the processed WESAD data from Google Drive to datasets/WESAD

# https://drive.google.com/file/d/1ve9ChpHoQ9SpplZAu6e2gdXvWOJyBIP6/view?usp=sharing
def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download&confirm=1"

    session = requests.Session()

    response = session.get(URL, params={"id": id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {"id": id, "confirm": token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value

    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)

if __name__ == "__main__":
    file_id = '1ve9ChpHoQ9SpplZAu6e2gdXvWOJyBIP6'
    DataSetPath = 'datasets/WESAD/'
    os.makedirs(DataSetPath, exist_ok=True)
    destination = os.path.join(DataSetPath, 'WESAD_processed.zip')
    # download_public_file(file_id, destination)
    download_file_from_google_drive(file_id, destination)
    zip_ref = zipfile.ZipFile(destination, 'r')
    zip_ref.extractall(DataSetPath)
    zip_ref.close()