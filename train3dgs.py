import os
import shutil
import subprocess
import logging
import argparse
BUCKET_NAME = "gpu-train-3dgs"
def process_video_pipeline(iterations, output_folder, image_folder, camera_model="OPENCV", colmap_command="colmap", use_gpu=True):
    """
    Run the COLMAP and 3DGS pipeline sequentially.

    Args:
        iterations (int): Number of training iterations for 3DGS.
        output_folder (str): Directory to store the output.
        image_folder (str): Directory containing the images for COLMAP.
        camera_model (str): Camera model to use in COLMAP.
        colmap_command (str): Command to run COLMAP.
        use_gpu (bool): Whether to use GPU for COLMAP tasks.
    """
    try:
        # 1. Run COLMAP Task
        logging.info(f"Starting COLMAP processing at: {output_folder}")
        os.makedirs(output_folder, exist_ok=True)
        distorted_path = os.path.join(output_folder, "distorted")
        sparse_path = os.path.join(distorted_path, "sparse")
        os.makedirs(sparse_path, exist_ok=True)

        database_path = os.path.join(distorted_path, "database.db")

        feat_extraction_cmd = (f"{colmap_command} feature_extractor "
                                f"--database_path {database_path} "
                                f"--image_path {image_folder} "
                                f"--ImageReader.single_camera 1 "
                                f"--ImageReader.camera_model {camera_model} "
                                f"--SiftExtraction.use_gpu {int(use_gpu)}")
        if os.system(feat_extraction_cmd) != 0:
            raise RuntimeError("Feature extraction failed.")

        feat_matching_cmd = (f"{colmap_command} exhaustive_matcher "
                              f"--database_path {database_path} "
                              f"--SiftMatching.use_gpu {int(use_gpu)}")
        if os.system(feat_matching_cmd) != 0:
            raise RuntimeError("Feature matching failed.")

        mapper_cmd = (f"{colmap_command} mapper "
                      f"--database_path {database_path} "
                      f"--image_path {image_folder} "
                      f"--output_path {sparse_path} "
                      f"--Mapper.ba_global_function_tolerance=0.000001")
        if os.system(mapper_cmd) != 0:
            raise RuntimeError("Mapping failed.")

        undistorted_output_path = output_folder
        img_undist_cmd = (f"{colmap_command} image_undistorter "
                          f"--image_path {image_folder} "
                          f"--input_path {os.path.join(sparse_path, '0')} "
                          f"--output_path {undistorted_output_path} "
                          f"--output_type COLMAP")
        if os.system(img_undist_cmd) != 0:
            raise RuntimeError("Image undistortion failed.")

        sparse_final_path = os.path.join(output_folder, "sparse", "0")
        os.makedirs(sparse_final_path, exist_ok=True)
        sparse_files = os.listdir(os.path.join(output_folder, "sparse"))
        for file in sparse_files:
            if file != '0':
                shutil.move(os.path.join(output_folder, "sparse", file),
                            os.path.join(sparse_final_path, file))

        logging.info("COLMAP processing completed successfully.")

        # 2. Run 3DGS Task
        logging.info(f"Starting 3DGS processing for video at: {output_folder}")
        base_dir = os.path.dirname(os.path.abspath(__file__))
        train_script_path = os.path.join(base_dir, "gaussian-splatting/train.py")

        command = [
            "python", train_script_path,
            "-s", output_folder,
            "--model_path", f"{output_folder}/model_{iterations}",
            "--iterations", str(iterations)
        ]

        subprocess.run(command, check=True)
        logging.info("3DGS processing completed successfully.")

        return {"status": "success", "details": "Pipeline completed successfully."}

    except Exception as e:
        logging.error(f"Pipeline encountered an error: {e}")
        return {"status": "failure", "details": str(e)}

def main():
    # Configure argument parser
    parser = argparse.ArgumentParser(description="Run COLMAP and 3DGS pipeline.")
    parser.add_argument("--video_name", type=str, help="Name of the video to process.")
    parser.add_argument("--iterations", type=int, help="Number of iterations for 3DGS.")
    parser.add_argument("--camera_model", type=str, default="OPENCV", help="Camera model to use in COLMAP.")
    parser.add_argument("--colmap_command", type=str, default="colmap", help="Command to run COLMAP.")
    parser.add_argument("--use_gpu", type=int, choices=[0, 1], default=1, help="Use GPU for COLMAP tasks (1 for True, 0 for False).")

    args = parser.parse_args()

    # Run the pipeline
    logging.basicConfig(level=logging.INFO)
    result = process_video_pipeline(
        iterations=args.iterations,
        output_folder=f"/gcs/{BUCKET_NAME}/results/{args.video_name}",
        image_folder=f"/gcs/{BUCKET_NAME}/images/{args.video_name}",
        camera_model=args.camera_model,
        colmap_command=args.colmap_command,
        use_gpu=bool(args.use_gpu)
    )
    print(result)

if __name__ == "__main__":
    main()
