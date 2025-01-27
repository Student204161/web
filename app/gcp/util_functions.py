from google.cloud import storage
import os

def upload_folder_to_gcs(local_folder, bucket_name, gcs_folder):
    """
    Uploads a local folder to a Google Cloud Storage bucket.
    Args:
        local_folder (str): Path to the local folder to upload.
        bucket_name (str): Name of the GCS bucket.
        gcs_folder (str): Folder path in the GCS bucket where files will be uploaded.
    """
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    for root, _, files in os.walk(local_folder):
        for file in files:
            local_file_path = os.path.join(root, file)
            # Determine GCS path
            relative_path = os.path.relpath(local_file_path, local_folder)
            blob_path = os.path.join(gcs_folder, relative_path).replace("\\", "/")

            # Upload file
            blob = bucket.blob(blob_path)
            blob.upload_from_filename(local_file_path)
            print(f"Uploaded {local_file_path} to gs://{bucket_name}/{blob_path}")

def upload_file_to_gcs(local_file, bucket_name, gcs_folder):
    """
    Uploads a local file to a Google Cloud Storage bucket.
    Args:
        local_file (str): Path to the local file to upload.
        bucket_name (str): Name of the GCS bucket.
        gcs_folder (str): Folder path in the GCS bucket where file will be uploaded.
    """
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    # Determine GCS path
    blob_path = os.path.join(gcs_folder, os.path.basename(local_file)).replace("\\", "/")

    # Upload file
    blob = bucket.blob(blob_path)
    blob.upload_from_filename(local_file)
    print(f"Uploaded {local_file} to gs://{bucket_name}/{blob_path}")

def download_folder_from_gcs(local_folder, bucket_name, gcs_folder):
    """
    Downloads a folder from GCS to a local path.
    Args:
        local_folder (str): Local folder path to download to.
        bucket_name (str): Name of the GCS bucket.
        gcs_folder (str): Folder path in the GCS bucket.
    """
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    blobs = bucket.list_blobs(prefix=gcs_folder)
    os.makedirs(local_folder, exist_ok=True)

    for blob in blobs:
        # Get relative path
        relative_path = os.path.relpath(blob.name, gcs_folder)
        local_file_path = os.path.join(local_folder, relative_path)

        # Ensure the local directory exists
        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

        # Download the file
        blob.download_to_filename(local_file_path)
        print(f"Downloaded {blob.name} to {local_file_path}")

def download_file_from_gcs(local_file, bucket_name, gcs_folder):
    """
    Downloads a file from GCS to a local path.
    Args:
        local_file (str): Local file path to download to.
        bucket_name (str): Name of the GCS bucket.
        gcs_folder (str): Folder path in the GCS bucket.
    """
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    # Determine GCS path
    blob_path = os.path.join(gcs_folder, os.path.basename(local_file)).replace("\\", "/")

    # Download file
    blob = bucket.blob(blob_path)
    blob.download_to_filename(local_file)
    print(f"Downloaded {blob_path} to {local_file}")

def check_if_file_exists_in_gcs(bucket_name, file_path):
    """
    Checks if a file exists in a Google Cloud Storage bucket.
    Args:
        bucket_name (str): Name of the GCS bucket.
        file_path (str): Path to the file in the GCS bucket.
    Returns:
        bool: True if the file exists, False otherwise.
    """
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(file_path)
    return blob.exists()