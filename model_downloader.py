import os
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobClient

def download_blob_with_managed_identity(blob_relative_path, local_path):
    # Construct the full URL for the blob
    storage_account_name = "storageforhealthmodels"  # Update this with your actual storage account name
    blob_url = f"https://{storage_account_name}.blob.core.windows.net/models/{blob_relative_path}"
    print(f"Attempting to download from: {blob_url}")

    # Use Managed Identity to authenticate
    credential = DefaultAzureCredential()  
    blob_client = BlobClient.from_blob_url(blob_url, credential=credential)

    # Ensure the directory for local storage exists
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    # Download the blob to a local file
    with open(local_path, "wb") as f:
        download_stream = blob_client.download_blob()
        f.write(download_stream.readall())
    print(f"Downloaded {blob_relative_path} to {local_path}")

def ensure_local_model(model_relative_path):
    local_path = os.path.join("/tmp/models", model_relative_path)
    if not os.path.exists(local_path):
        print(f"{local_path} not found locally. Downloading from Azure...")
        download_blob_with_managed_identity(model_relative_path, local_path)
    else:
        print(f"{local_path} already exists.")
    return local_path