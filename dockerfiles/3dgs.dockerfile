# Use NVIDIA's official PyTorch base image with CUDA 11.8 support
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04
ARG TORCH_CUDA_ARCH_LIST="3.5;5.0;6.0;6.1;7.0;7.5;8.0;8.6+PTX"
# Set the working directory
WORKDIR /
RUN export GOOGLE_APPLICATION_CREDENTIALS="webproj-447013-a1db168cc4ae.json"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-distutils \
    python3.10-dev \
    git \
    wget \
    libgl1-mesa-glx \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install pip for Python 3.10
RUN wget https://bootstrap.pypa.io/get-pip.py && \
    python3.10 get-pip.py && \
    rm get-pip.py

# Upgrade pip to the specified version
RUN pip install --no-cache-dir --upgrade pip==22.3.1

# Install Python dependencies
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Clone the main repository
RUN git clone https://github.com/graphdeco-inria/gaussian-splatting.git /app/gaussian-splatting

# Install submodules as Python packages
RUN pip install --no-cache-dir \
    git+https://github.com/graphdeco-inria/gaussian-splatting.git@main#subdirectory=submodules/diff-gaussian-rasterization \
    git+https://github.com/graphdeco-inria/gaussian-splatting.git@main#subdirectory=submodules/simple-knn \
    git+https://github.com/graphdeco-inria/gaussian-splatting.git@main#subdirectory=submodules/fused-ssim

# Ensure the container can use NVIDIA GPUs
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

COPY train3dgs.py train3dgs.py

# RUN gcloud services enable artifactregistry.googleapis.com
# RUN gcloud services enable cloudbuild.googleapis.com
# Set the entrypoint
CMD ["bash"]

# PROJECT=webproj-447013
# MEMBER=user:khalil0221@gmail.com
# ROLE="roles/artifactregistry.reader"

# gcloud projects add-iam-policy-binding $PROJECT \
#     --member=$MEMBER \
#     --role=$ROLE

# newgrp docker -- this fixes the issue of not being able to pull images from the registry
# gcloud artifacts repositories set-cleanup-policies train3dgs --project=webproj-447013 --location=europe-west1   --policy=policy.yaml
#export PATH=$PATH:/usr/local/google-cloud-sdk/bin

# docker run --gpus all europe-west1-docker.pkg.dev/webproj-447013/train3dgs/train python3 train3dgs.py --video_name test_video --iterations 100
