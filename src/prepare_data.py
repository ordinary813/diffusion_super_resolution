import kagglehub
import os

def download_div2k():
    download_dir = "../data" 
    abs_download_dir = os.path.abspath(download_dir)
    os.makedirs(download_dir, exist_ok=True)

    # Set download path
    os.environ["KAGGLEHUB_CACHE"] = abs_download_dir

    path = kagglehub.dataset_download(
        "soumikrakshit/div2k-high-resolution-images",
        force_download=False
    )
    return path

def download_celebA():
    download_dir = "../data" 
    abs_download_dir = os.path.abspath(download_dir)
    os.makedirs(download_dir, exist_ok=True)

    # Set download path
    os.environ["KAGGLEHUB_CACHE"] = abs_download_dir

    path = kagglehub.dataset_download(
        "badasstechie/celebahq-resized-256x256",
        force_download=False
    )
    return path

if __name__ == "__main__":
    path = download_div2k()
    print(f"div2k dataset is located at: {path}")
    path = download_celebA()
    print(f"celebA dataset is located at: {path}")
