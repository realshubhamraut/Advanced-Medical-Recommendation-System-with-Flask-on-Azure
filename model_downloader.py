import os
import requests
from dotenv import load_dotenv

load_dotenv()

def download_from_azure(blob_relative_path, local_path):
    sas_token = os.getenv('AZURE_STORAGE_SAS_TOKEN')
    if not sas_token:
        raise Exception("SAS token not found in environment variables")

    storage_account_name = "storageforhealthmodels"
    base_url = f"https://{storage_account_name}.blob.core.windows.net/models"
    url = f"{base_url}/{blob_relative_path}?{sas_token}"
    print(f"Attempting to download from: {url}")
    
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        with open(local_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print(f"Downloaded {blob_relative_path} to {local_path}")
    else:
        print(f"Failed to download {url}: HTTP {response.status_code}")
        raise Exception(f"Failed to download {url}: HTTP {response.status_code}")

def ensure_local_model(model_relative_path):
    local_path = os.path.join("models", model_relative_path)
    if not os.path.exists(local_path):
        print(f"{local_path} not found locally. Downloading from Azure...")
        download_from_azure(model_relative_path, local_path)
    else:
        print(f"{local_path} already exists.")
    return local_path
