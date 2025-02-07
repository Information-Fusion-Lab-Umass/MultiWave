import os
import zipfile
import gdown

if __name__ == '__main__':
    file_id = '1ve9ChpHoQ9SpplZAu6e2gdXvWOJyBIP6'
    DataSetPath = 'datasets/WESAD'
    os.makedirs(DataSetPath, exist_ok=True)
    destination = os.path.join(DataSetPath, 'WESAD_processed.zip')
    
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, destination, quiet=False)
    
    with zipfile.ZipFile(destination, 'r') as zip_ref:
        zip_ref.extractall(DataSetPath)