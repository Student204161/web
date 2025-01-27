import os
import logging
import sys
from google.cloud import storage
import subprocess
import shutil
def download_data_from_gcs(bucket_name, gcs_folder, local_folder):
    """
    Simulate downloading data by copying files from a local folder.
    """
    os.makedirs(local_folder, exist_ok=True)
    local_gcs_folder = f"mock_gcs/{gcs_folder}"  # Replace with a local path for testing
    for file in os.listdir(local_gcs_folder):
        src = os.path.join(local_gcs_folder, file)
        dst = os.path.join(local_folder, file)
        if os.path.isfile(src):
            shutil.copy2(src, dst)
    logging.info(f"Simulated download from {gcs_folder} to {local_folder}.")

def upload_results_to_gcs(bucket_name, local_folder, gcs_folder):
    """
    Simulate uploading data by copying files to a local folder.
    """
    local_gcs_folder = f"mock_gcs/{gcs_folder}"  # Replace with a local path for testing
    os.makedirs(local_gcs_folder, exist_ok=True)
    for root, _, files in os.walk(local_folder):
        for file in files:
            src = os.path.join(root, file)
            dst = os.path.join(local_gcs_folder, os.path.relpath(src, local_folder))
            shutil.copy2(src, dst)
    logging.info(f"Simulated upload from {local_folder} to {gcs_folder}.")

def check_if_gcs_folder_exists(bucket_name, gcs_folder):
    """
    Simulate checking a GCS folder by checking a local path.
    """
    local_gcs_folder = f"mock_gcs/{gcs_folder}"  # Replace with a local path for testing
    return os.path.exists(local_gcs_folder) and os.listdir(local_gcs_folder)
