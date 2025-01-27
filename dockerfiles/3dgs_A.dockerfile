# This could also be another Ubuntu or Debian based distribution
FROM ubuntu:22.04
RUN export GOOGLE_APPLICATION_CREDENTIALS="webproj-447013-a1db168cc4ae.json"
# Install Open3D system dependencies and pip
RUN apt-get update && apt-get install --no-install-recommends -y \
    libegl1 \
    libgl1 \
    libgomp1 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install Open3D from the PyPI repositories
RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir --upgrade open3d

    

RUN gcloud services enable artifactregistry.googleapis.com
RUN gcloud services enable cloudbuild.googleapis.com
# gcloud artifacts repositories add-iam-policy-binding webprojreg \
#     --location=europe-west1 \
#     --member="serviceAccount:1061166014129-compute@developer.gserviceaccount.com" \
#     --role="roles/artifactregistry.writer"
    