from flask import Flask, Blueprint, render_template, request, jsonify
import os, cv2, shutil, requests
from app.colmap.colmap_wrapper import run_colmap  # Import COLMAP wrapper
from app.utils import mov_to_frames  # Assuming mov_to_frames extracts frames from the video
from app.gcp.util_functions import *
from app.utils import resize_images_in_folder
main = Blueprint('main', __name__)

# Credentials for cloud storage
BUCKET_NAME = "gpu-train-3dgs"
# Define paths
OUTPUT_FOLDER = os.path.join('app/outputs')
MOVS_FOLDER = 'app/static/uploads/movs'  # Path for storing videos
IMAGES_FOLDER = 'app/static/uploads/images'  # Path for storing extracted images

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(MOVS_FOLDER, exist_ok=True)
os.makedirs(IMAGES_FOLDER, exist_ok=True)

TARGET_WIDTH = 240
TARGET_HEIGHT = 180

# Metadata helper
def get_metadata(metadata_path):
    url = f"http://metadata.google.internal/computeMetadata/v1/{metadata_path}"
    headers = {"Metadata-Flavor": "Google"}
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.text

@main.route('/instance-ip', methods=['GET'])
def instance_ip():
    try:
        internal_ip = get_metadata("instance/network-interfaces/0/ip")
        external_ip = get_metadata("instance/network-interfaces/0/access-configs/0/external-ip")
        return jsonify({"internal_ip": internal_ip, "external_ip": external_ip})
    except requests.RequestException as e:
        return jsonify({"error": str(e)}), 500

@main.route('/')
def home():
    return render_template('index.html', title="Home Page")

@main.route('/tasks', methods=['GET', 'POST'])
def tasks():
    tasks = [
        {"id": "calibration", "name": "Camera Calibration", "description": "Runs COLMAP for camera calibration."},
        {"id": "3DGS", "name": "3DGS Task", "description": "Run the 3DGS processing pipeline."}
    ]
    result = None

    if request.method == 'POST':
        task_type = request.form.get('task_type')
        video = request.files.get('video')
        num_frames = int(request.form.get('frame_count'))
        iterations = int(request.form.get('iterations'))

        if not video:
            result = {"error": "Please upload a video file."}
        else:
            video_name = os.path.splitext(video.filename)[0]
            video_path = os.path.join(MOVS_FOLDER, video.filename)
            if os.path.exists(os.path.join(IMAGES_FOLDER, video_name)):
                shutil.rmtree(os.path.join(IMAGES_FOLDER, video_name))

            video.save(video_path)
            # mov_to_frames(video_path, os.path.join(IMAGES_FOLDER, video_name), num_frames)
            # # Resize images
            # resize_images_in_folder(os.path.join(IMAGES_FOLDER, video_name), TARGET_WIDTH, TARGET_HEIGHT)
            # gcs_folder = f"images/{video_name}"
            # upload_folder_to_gcs(os.path.join(IMAGES_FOLDER, video_name), BUCKET_NAME, gcs_folder)

            gpu_endpoint = f"http://{get_metadata('instance/network-interfaces/0/access-configs/0/external-ip')}:5000/run_train"
            try:
                response = requests.post(gpu_endpoint, json={"task_type": task_type, "video_name": video_name, "iterations": iterations})
                if response.status_code == 200:
                    result = response.json()
                else:
                    result = {"error": f"Failed to run {task_type}: {response.json()}"}
            except Exception as e:
                result = {"error": f"Error communicating with GPU instance: {str(e)}"}

    return render_template('tasks.html', title="Task Runner", tasks=tasks, result=result)
