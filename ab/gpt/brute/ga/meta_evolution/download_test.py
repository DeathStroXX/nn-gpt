import kagglehub
import shutil

print("Attempting to download imagenet100...")
try:
    path = kagglehub.dataset_download("ambityga/imagenet100")
    print(f"Downloaded to {path}")
    
    dest = "imagenet100_data"
    shutil.copytree(path, dest, dirs_exist_ok=True)
    print(f"Copied successfully to {dest}")
except Exception as e:
    print(f"Error: {e}")
