import os
import glob
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

def load_image_as_np(path):
    img = Image.open(path).convert('RGB')
    return np.array(img) / 255.0

def calculate_metrics(target_np, predicted_np):
    mse_val = np.mean((target_np - predicted_np) ** 2)
    psnr_val = psnr(target_np, predicted_np, data_range=1.0)
    ssim_val = ssim(target_np, predicted_np, data_range=1.0, channel_axis=2)
    return {"MSE": mse_val, "PSNR": psnr_val, "SSIM": ssim_val}

def evaluate_experiment(exp_path):
    images_dir = os.path.join(exp_path, "images")
    if not os.path.exists(images_dir):
        return None

    hr_files = glob.glob(os.path.join(images_dir, "*_HR.png"))
    if not hr_files:
        return None
    
    hr_np = load_image_as_np(hr_files[0])
    all_pngs = glob.glob(os.path.join(images_dir, "*.png"))
    result_files = []
    
    for f in all_pngs:
        filename = os.path.basename(f).replace('.png', '')
        if filename.isdigit():
            result_files.append(f)
    
    if not result_files:
        return None

    result_files.sort(key=lambda x: int(os.path.basename(x).replace('.png', '')))

    stats = {"epochs": [], "psnr": [], "ssim": []}
    
    for res_path in result_files:
        epoch_num = int(os.path.basename(res_path).replace('.png', ''))
        res_np = load_image_as_np(res_path)
        if hr_np.shape != res_np.shape:
            continue
        m = calculate_metrics(hr_np, res_np)
        stats["epochs"].append(epoch_num)
        stats["psnr"].append(m["PSNR"])
        stats["ssim"].append(m["SSIM"])
    return stats

if __name__ == "__main__":
    base_dir = "experiments"
    if os.path.exists(base_dir):
        for d in os.listdir(base_dir):
            exp_dir = os.path.join(base_dir, d)
            if os.path.isdir(exp_dir):
                results = evaluate_experiment(exp_dir)
                if results and results["epochs"]:
                    plt.figure(figsize=(12, 5))
                    plt.subplot(1, 2, 1)
                    plt.plot(results["epochs"], results["psnr"], 'b-o', markersize=4)
                    plt.title(f"PSNR - {d}")
                    plt.xlabel("Epoch")
                    plt.ylabel("dB")
                    plt.grid(True)
                    
                    plt.subplot(1, 2, 2)
                    plt.plot(results["epochs"], results["ssim"], 'r-o', markersize=4)
                    plt.title(f"SSIM - {d}")
                    plt.xlabel("Epoch")
                    plt.ylabel("Score")
                    plt.grid(True)
                    
                    plt.tight_layout()
                    save_path = os.path.join(exp_dir, f"metrics_plot_{d}.png")
                    plt.savefig(save_path)
                    plt.close()