from flask import Flask, Blueprint, render_template, request, jsonify
import os, shutil, requests, threading
from app.utils import resize_images_in_folder, mov_to_frames
from app.gcp.util_functions import upload_data_to_gcs
from google.cloud import aiplatform

main = Blueprint('main', __name__)

# Global Task Status Tracker
task_status = {}
TARGET_WIDTH = 240
TARGET_HEIGHT = 180
# Credentials for cloud storage
BUCKET_NAME = "gpu-train-3dgs"
# Define paths
OUTPUT_FOLDER = os.path.join('app/outputs')
MOVS_FOLDER = 'app/static/uploads/movs'  # Path for storing videos
IMAGES_FOLDER = 'app/static/uploads/images'  # Path for storing extracted images
# Initialize the Vertex AI SDK
PROJECT_ID = "webproj-447013"
LOCATION = "eu-west1"

aiplatform.init(project=PROJECT_ID, location=LOCATION)

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


def run_task(task_id, task_func, *args, **kwargs):
    """Run a task in a separate thread and update the task status."""
    task_status[task_id] = "running"
    try:
        task_func(*args, **kwargs)
        task_status[task_id] = "completed"
    except Exception as e:
        task_status[task_id] = f"failed: {str(e)}"

@main.route('/task_status', methods=['GET'])
def task_status_route():
    """Return the status of all tasks."""
    return jsonify(task_status)

@main.route('/tasks', methods=['GET', 'POST'])
def tasks():
    tasks = [
        {"id": "3DGS", "name": "3DGS Task", "description": "Run the 3DGS processing pipeline."},
        {"id": "segmentation", "name": "Segmentation", "description": "Run the segmentation task."}
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
            image_output_folder = os.path.join(IMAGES_FOLDER, video_name)
            local_output_folder = os.path.join(OUTPUT_FOLDER, video_name)
            gcs_output_folder = f"results/{video_name}"

            # Save the uploaded video
            video.save(video_path)

            # Create a unique task ID
            task_id = f"{task_type}_{video_name}_{len(task_status)}"

            if not os.path.exists(image_output_folder) or len(os.listdir(image_output_folder)) == 0:
                # Process frames if not already done
                mov_to_frames(video_path, image_output_folder, num_frames)
                resize_images_in_folder(image_output_folder, TARGET_WIDTH, TARGET_HEIGHT)
            else:
                print(f"Frames for {video_name} already processed. Skipping frame extraction and resizing.")
            #upload the images to GCS
            upload_data_to_gcs(BUCKET_NAME, image_output_folder, f"images/{video_name}")
            if task_type == "3DGS":
                # Task 1: 3DGS Training
                if task_type == "3DGS":
                    def task_func():
                        # if not skip_colmap:
                        #     result = run_colmap_task(local_output_folder, local_input_folder)        
                        #     result = run_3dgs_task(local_output_folder, iterations=iterations)
                        upload_data_to_gcs(BUCKET_NAME, f"{local_output_folder}/model_{iterations}", f"{gcs_output_folder}/model_{iterations}")
                        run_vertex_ai_3dgs(video_name, iterations)

            elif task_type == "segmentation":
                # Task 2: Segmentation Task
                # Args: video_name, clicks, labels
                clicks = request.form.get('clicks')
                labels = request.form.get('labels')

                #return not implemented error
                return jsonify({"error": "Not implemented yet."}), 400    

            # Run the task in a separate thread
            threading.Thread(target=run_task, args=(task_id, task_func)).start()
            task_status[task_id] = "queued"

            result = {"message": f"{task_type} task started.", "task_id": task_id}

    return render_template('tasks.html', title="Task Runner", tasks=tasks, result=result)

                    

def run_vertex_ai_3dgs(video_name, iterations):
    """
    Function to trigger Vertex AI 3DGS job
    """
    from google.cloud import aiplatform
    job = aiplatform.CustomJob.from_local_script(
        display_name="3DGS-Training",
        script_path="train3dgs.py",
        container_uri=f"gcr.io/{PROJECT_ID}/train3dgs.py",  # Replace with your image
        #specify machine type and gpu
        machine_type="n1-standard-8",
        accelerator_type="NVIDIA_TESLA_T4",
        accelerator_count=1,
        environment_variables={
            "VIDEO_NAME": video_name,
            "ITERATIONS": str(iterations)
        },
        args=[
            "--video_name", video_name,
            "--iterations", str(iterations),
        ]
    )
    job.run(sync=False)

    