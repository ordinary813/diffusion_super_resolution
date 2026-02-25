import kagglehub
import os
import random
import shutil
import glob

def download_div2k(abs_download_dir: str):
    path = kagglehub.dataset_download(
        "soumikrakshit/div2k-high-resolution-images",
        force_download=False
    )
    return path

def download_celebA():
    path = kagglehub.dataset_download(
        "badasstechie/celebahq-resized-256x256",
        force_download=False
    )
    return path

def fix_and_split(base_path: str):
    source_dir = os.path.join(base_path, "celeba_hq_256")
    
    train_dir = os.path.join(base_path, "train")
    test_dir = os.path.join(base_path, "test")

    if not os.path.exists(source_dir):
        print(f"Error: Could not find {source_dir}. Please check your current directory.")
        return

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    images = glob.glob(os.path.join(source_dir, "*.jpg"))

    if len(images) == 0:
        return

    random.shuffle(images)
    split_idx = int(len(images) * 0.9)
    
    train_files = images[:split_idx]
    test_files = images[split_idx:]

    for f in train_files:
        shutil.move(f, os.path.join(train_dir, os.path.basename(f)))
    for f in test_files:
        shutil.move(f, os.path.join(test_dir, os.path.basename(f)))

if __name__ == "__main__":
    download_dir = "./data" 
    abs_download_dir = os.path.abspath(download_dir)
    os.makedirs(download_dir, exist_ok=True)

    # Set kaggle download path
    os.environ["KAGGLEHUB_CACHE"] = abs_download_dir

    path = download_div2k()
    print(f"div2k dataset is located at: {path}")
    path = download_celebA()
    print(f"celebA dataset is located at: {path}")
    fix_and_split("./data/datasets/badasstechie/celebahq-resized-256x256/versions/1")
