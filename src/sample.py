import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import glob
import random
import math
import matplotlib.pyplot as plt

from model import StandardUNet
from diffusion import DiffusionModel

USE_RANDOM_IMAGE = True 
FIXED_IMAGE_PATH = "../data/datasets/soumikrakshit/div2k-high-resolution-images/versions/1/DIV2K_valid_HR/DIV2K_valid_HR/0016.png"

# Load and preprocess an image
def load_and_prepare_image(hr_image_path, target_size, scale_factor, device):
    hr_img = Image.open(hr_image_path).convert('RGB')
    w, h = hr_img.size
    new_w = (w // 32) * 32
    new_h = (h // 32) * 32
    
    hr_transform = transforms.Compose([
        transforms.CenterCrop((new_h, new_w)), # Crop to center
        transforms.Resize(target_size),        # Resize to target
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    hr_tensor = hr_transform(hr_img).to(device)
    
    lr_size = target_size[0] // scale_factor
    
    # Un-normalize to [0,1] first because Resize expects standard image ranges
    un_norm_transform = transforms.Normalize(
        mean=[-1.0, -1.0, -1.0],
        std=[2.0, 2.0, 2.0]
    )
    hr_tensor_unnormalized = un_norm_transform(hr_tensor.cpu())
    
    lr_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(lr_size, interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.Resize(target_size, interpolation=transforms.InterpolationMode.BICUBIC), # Upscale back to input size
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    lr_tensor = lr_transform(hr_tensor_unnormalized).to(device)
    
    return lr_tensor.unsqueeze(0), hr_tensor.unsqueeze(0)

# Convert [-1, 1] tensor back to [0, 1] for plotting
def un_normalize(tensor):
    tensor = (tensor.clamp(-1, 1) + 1) / 2
    tensor = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    return tensor

def run_sampling_loop(model, diffusion, condition, cfg_scale):
    x_t = torch.randn_like(condition)

    print(f"Sampling {diffusion.timesteps} steps") 
    with torch.no_grad():
        for t in reversed(range(diffusion.timesteps)):
            x_t = diffusion.sample_step(model, x_t, t, condition, cfg_scale=cfg_scale)
    return x_t

# Selects an image at random
def get_image_path(config):
    if USE_RANDOM_IMAGE:
        search_path = os.path.join(config['data']['path'], "**", "*.png")
        all_images = glob.glob(search_path, recursive=True)
        
        if not all_images:
            print(f"No images found in {config['data']['path']}")
            return None
            
        selected = random.choice(all_images)
        print(f"Dynamically selected image: {os.path.basename(selected)}")
        return selected
    else:
        return FIXED_IMAGE_PATH

# Finds and sorts all checkpoints in a directory
def get_model_paths(save_dir):
    search_path = os.path.join(save_dir, "*.pth")
    files = glob.glob(search_path)
    
    if not files:
        return []

    def sort_key(f):
        filename = os.path.basename(f)
        if 'latest' in filename:
            return 999999
        nums = [int(s) for s in filename.replace('_', ' ').replace('.', ' ').split() if s.isdigit()]
        return nums[0] if nums else 0

    files = sorted(files, key=sort_key)
    return files

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    dataset_dir = '../data/datasets/soumikrakshit/div2k-high-resolution-images/versions/1/DIV2K_valid_HR'
    base_experiments_dir = "experiments"

    # Get all experiment sub directories
    experiment_dirs = [os.path.join(base_experiments_dir, d) for d in os.listdir(base_experiments_dir) 
                       if os.path.isdir(os.path.join(base_experiments_dir, d))]
    
    for exp_dir in experiment_dirs:
        exp_name = os.path.basename(exp_dir)
        print(f"Processing: {exp_name}")

        model_paths = get_model_paths(exp_dir)
        if not model_paths:
            continue
        
        # Extraction of number of channels
        # (Each experiment had a different amount of channels, I decided to automatically check for it)
        test_ckpt = torch.load(model_paths[-1], map_location='cpu')
        state_dict = test_ckpt['model'] if 'model' in test_ckpt else test_ckpt

        current_channels = state_dict['conv0.weight'].shape[0]
        has_attention = any("attn" in key for key in state_dict.keys())
        print(f"{current_channels} channels, Attention: {has_attention}")

        # Model creation for inference
        model = StandardUNet(model_channels=current_channels, use_attention=has_attention).to(device)
        diffusion = DiffusionModel(device=device)

        # Image selection
        config = {'data': {'path': dataset_dir, 'hr_size': 256, 'scale_factor': 2}}
        img_path = get_image_path(config)
        lr_upscaled, hr_original = load_and_prepare_image(img_path, (256, 256), 2, device)

        # Output layout
        num_models = len(model_paths)
        cols = 4
        rows = math.ceil(num_models / cols) + 1
        fig = plt.figure(figsize=(20, 5 * rows))
        plt.suptitle(f"Training Quality Timeline: {os.path.basename(img_path)} (Exp: {exp_name})", fontsize=16)

        # Ground Truth
        ax = plt.subplot(rows, 2, 1)
        ax.imshow(un_normalize(hr_original))
        ax.set_title("Ground Truth (HR)", fontsize=14)
        ax.axis('off')
        
        # Input (Low res - Upscaled)
        ax = plt.subplot(rows, 2, 2)
        ax.imshow(un_normalize(lr_upscaled))
        ax.set_title(f"Input (LR - {config['data']['scale_factor']}x upscaled)", fontsize=14)
        ax.axis('off')


        for i, path in enumerate(model_paths):
            print(f"Processing: {os.path.basename(path)}")
            checkpoint = torch.load(path, map_location=device)
            state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
            model.load_state_dict(state_dict)
            model.eval()
            
            result = run_sampling_loop(model, diffusion, lr_upscaled, cfg_scale=4.0)
            
            ax = plt.subplot(rows, cols, i + 1 + cols)
            epoch_num = path.split('_')[-1].split('.')[0]
            ax.imshow(un_normalize(result))
            ax.set_title(f"Epoch {epoch_num}", fontsize=10)
            ax.axis('off')
        
        output_path = os.path.join(exp_dir, f"comparison_{exp_name}.png")
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        print(f"Saved results for {exp_name} to {output_path}\n")