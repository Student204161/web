import os
import logging
import sys
from google.cloud import storage
import subprocess
import shutil

def download_data_from_gcs(bucket_name, gcs_folder, local_folder):
    """
    Download data from a GCS folder to a local folder.
    """
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=gcs_folder)

    os.makedirs(local_folder, exist_ok=True)
    for blob in blobs:
        if not blob.name.endswith("/"):  # Skip folder markers
            local_file = os.path.join(local_folder, os.path.basename(blob.name))
            blob.download_to_filename(local_file)
    logging.info(f"Downloaded data from {gcs_folder} to {local_folder}.")

def upload_results_to_gcs(bucket_name, local_folder, gcs_folder):
    """
    Upload local files to a GCS folder.
    """
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    for root, _, files in os.walk(local_folder):
        for file in files:
            local_file = os.path.join(root, file)
            blob = bucket.blob(f"{gcs_folder}/{os.path.relpath(local_file, local_folder)}")
            blob.upload_from_filename(local_file)
    logging.info(f"Uploaded results from {local_folder} to {gcs_folder}.")

def check_if_gcs_folder_exists(bucket_name, gcs_folder):
    """
    Check if a GCS folder exists.
    """
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=gcs_folder)
    return any(True for _ in blobs)