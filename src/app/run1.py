from flask import Flask, request, jsonify
import os
import logging
import sys
from google.cloud import storage, aiplatform
import subprocess
from utils.util import download_data_from_gcs, upload_results_to_gcs, check_if_gcs_folder_exists
app = Flask(__name__)

import shutil

# Configure Google Cloud Storage
BUCKET_NAME = "gpu-train-3dgs"
OUTPUT_FOLDER = 'src/app/'

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Initialize the Vertex AI SDK
PROJECT_ID = "webproj-447013"
LOCATION = "eu-west1"

aiplatform.init(project=PROJECT_ID, location=LOCATION)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

docker_image_is_segment = False
if docker_image_is_segment == True:
    from sam2.build_sam import build_sam2_video_predictor
    import torch
    sam2_checkpoint = "../checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
    inference_state = None

@app.route('/run_train', methods=['POST'])
def run_train():
    try:
        # Parse incoming JSON payload
        data = request.get_json()
        task_type = data.get("task_type")
        video_name = data.get("video_name")
        iterations = int(data.get("iterations"))
        if not task_type or not video_name or not iterations:
            return jsonify({"error": "Missing task_type, video_name or iterations in request."}), 400

        # Define GCS paths
        gcs_input_folder = f"images/{video_name}"
        gcs_output_folder = f"results/{video_name}"

        # Define local paths
        local_input_folder = os.path.join(OUTPUT_FOLDER, "data", video_name)
        local_output_folder = os.path.join(OUTPUT_FOLDER, "results", video_name)

        #check if gcs output folder exists, if so, don't run calibration again, just download the results
        # if check_if_gcs_folder_exists(BUCKET_NAME, gcs_output_folder):
        #     download_data_from_gcs(BUCKET_NAME, gcs_output_folder, local_output_folder)
        #     skip_colmap=True
        # else:
        #     skip_colmap=False
        # Download input data from GCS
        download_data_from_gcs(BUCKET_NAME, gcs_input_folder, local_input_folder)
        skip_colmap=False
        # Run the requested task
        if task_type == "3DGS":
            if not skip_colmap:
                result = run_colmap_task(local_output_folder, local_input_folder)        
            result = run_3dgs_task(local_output_folder, iterations=iterations)
            upload_results_to_gcs(BUCKET_NAME, f"{local_output_folder}/model_{iterations}", f"{gcs_output_folder}/model_{iterations}")
        elif task_type == "segmentation":
            #samv2 segmentation

            return jsonify({"error": "Not implementd yet "}), 400

        elif task_type == "extraction":


            return jsonify({"error": "Not implementd yet "}), 400

        else:
            return jsonify({"error": "Unsupported task type."}), 400

        # Upload results to GCS
        upload_results_to_gcs(BUCKET_NAME, local_output_folder, gcs_output_folder)

        return jsonify({"result": result}), 200

    except Exception as e:
        logging.error(f"Error in /run_train: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/run_segmentation', methods=['POST'])
def run_segmentation():
    try:
        # Parse incoming JSON payload
        data = request.get_json()
        clicks = data.get("clicks")  # User clicks on the image
        labels = data.get("labels")  # Corresponding labels
        image_path = data.get("image_path")  # Path to the image
        if not clicks or not labels or not image_path:
            return jsonify({"error": "Missing clicks, labels, or image_path in request."}), 400

        # Load the segmentation model
        from segmentation_model import Segmenter  # Assume Segmenter is the segmentation model
        model = Segmenter()

        # Perform segmentation
        mask = model.run_segmentation(image_path, clicks, labels)

        # Convert mask to a suitable format (e.g., base64 or binary)
        _, mask_encoded = cv2.imencode('.png', mask)
        mask_base64 = base64.b64encode(mask_encoded).decode('utf-8')

        return jsonify({"mask": mask_base64}), 200

    except Exception as e:
        logging.error(f"Error in /run_segmentation: {str(e)}")
        return jsonify({"error": str(e)}), 500


def run_colmap_task(output_folder, image_folder, camera_model="OPENCV", colmap_command="colmap", use_gpu=True):
    """
    Run COLMAP pipeline for camera calibration.
    """
    try:
        # Ensure paths exist
        os.makedirs(output_folder, exist_ok=True)
        distorted_path = os.path.join(output_folder, "distorted")
        sparse_path = os.path.join(distorted_path, "sparse")
        os.makedirs(sparse_path, exist_ok=True)

        # Database path
        database_path = os.path.join(distorted_path, "database.db")

        # Feature extraction
        feat_extraction_cmd = (f"{colmap_command} feature_extractor "
                                f"--database_path {database_path} "
                                f"--image_path {image_folder} "
                                f"--ImageReader.single_camera 1 "
                                f"--ImageReader.camera_model {camera_model} "
                                f"--SiftExtraction.use_gpu {int(use_gpu)}")
        exit_code = os.system(feat_extraction_cmd)
        if exit_code != 0:
            logging.error(f"Feature extraction failed with code {exit_code}. Exiting.")
            return {"status": "failure", "details": "Feature extraction failed."}

        # Feature matching
        feat_matching_cmd = (f"{colmap_command} exhaustive_matcher "
                              f"--database_path {database_path} "
                              f"--SiftMatching.use_gpu {int(use_gpu)}")
        exit_code = os.system(feat_matching_cmd)
        if exit_code != 0:
            logging.error(f"Feature matching failed with code {exit_code}. Exiting.")
            return {"status": "failure", "details": "Feature matching failed."}

        # Bundle adjustment (Mapping)
        mapper_cmd = (f"{colmap_command} mapper "
                      f"--database_path {database_path} "
                      f"--image_path {image_folder} "
                      f"--output_path {sparse_path} "
                      f"--Mapper.ba_global_function_tolerance=0.000001")
        exit_code = os.system(mapper_cmd)
        if exit_code != 0:
            logging.error(f"Mapper failed with code {exit_code}. Exiting.")
            return {"status": "failure", "details": "Mapping failed."}

        # Image undistortion
        undistorted_output_path = output_folder
        img_undist_cmd = (f"{colmap_command} image_undistorter "
                          f"--image_path {image_folder} "
                          f"--input_path {os.path.join(sparse_path, '0')} "
                          f"--output_path {undistorted_output_path} "
                          f"--output_type COLMAP")
        exit_code = os.system(img_undist_cmd)
        if exit_code != 0:
            logging.error(f"Image undistortion failed with code {exit_code}. Exiting.")
            return {"status": "failure", "details": "Image undistortion failed."}

        # Restructure sparse files
        sparse_final_path = os.path.join(output_folder, "sparse", "0")
        os.makedirs(sparse_final_path, exist_ok=True)
        sparse_files = os.listdir(os.path.join(output_folder, "sparse"))
        for file in sparse_files:
            if file != '0':
                shutil.move(os.path.join(output_folder, "sparse", file),
                            os.path.join(sparse_final_path, file))

        logging.info("COLMAP processing completed successfully.")
        return {"status": "success", "details": "COLMAP processing completed."}

    except Exception as e:
        logging.error(f"COLMAP task encountered an error: {e}")
        return {"status": "failure", "details": str(e)}


def run_3dgs_task(output_folder, iterations=300):
    """
    Run the 3DGS task pipeline.
    """
    try:
        # Resolve the absolute path to train.py
        base_dir = os.path.dirname(os.path.abspath(__file__))  # Path of this script
        train_script_path = os.path.join(base_dir, "../../gaussian-splatting/train.py")

        # Example: Placeholder for 3DGS command
        command = [
            "python", train_script_path,
            "-s", output_folder,
            "--model_path", f"{output_folder}/model_{iterations}",
            "--iterations", str(iterations)
        ]

        # Execute the command
        subprocess.run(command, check=True)
        #upload model_path to gcs
        logging.info("3DGS processing completed successfully")
        return {"status": "success", "details": "3DGS processing completed."}

    except subprocess.CalledProcessError as e:
        logging.error(f"3DGS command failed: {e}")
        raise

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
