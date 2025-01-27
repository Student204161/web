import os
import subprocess

def run_colmap(image_folder, output_folder):
    """
    Run COLMAP for camera calibration.
    Args:
        image_folder (str): Path to the folder containing input images.
        output_folder (str): Path to the folder where COLMAP outputs will be saved.
    Returns:
        str: Path to the cameras file containing calibration data.
    """
    sparse_folder = os.path.join(output_folder, "sparse")
    database_path = os.path.join(output_folder, "database.db")
    os.makedirs(sparse_folder, exist_ok=True)

    # Set the QT_QPA_PLATFORM environment variable to offscreen
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    
    try:
        subprocess.run([
            "colmap", "feature_extractor",
            "--database_path", database_path,
            "--image_path", image_folder
        ], check=True)

        subprocess.run([
            "colmap", "exhaustive_matcher",
            "--database_path", database_path
        ], check=True)

        subprocess.run([
            "colmap", "mapper",
            "--database_path", database_path,
            "--image_path", image_folder,
            "--output_path", sparse_folder
        ], check=True)

        return os.path.join(sparse_folder, "0", "cameras.txt")
    
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"COLMAP failed: {e}")
