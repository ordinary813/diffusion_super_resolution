import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import matplotlib.pyplot as plt

from model import StandardUNet
from diffusion import DiffusionModel

# Loads an HR image, creates an LR image and crops them respectively
def load_and_prepare_image(hr_image_path, target_size, scale_factor, device):
    hr_img = Image.open(hr_image_path).convert('RGB')
    
    hr_transform = transforms.Compose([
        transforms.CenterCrop(target_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    hr_tensor = hr_transform(hr_img).to(device)
    
    lr_size = target_size[0] // scale_factor
    un_norm_transform = transforms.Normalize(
        mean=[-1.0, -1.0, -1.0], # (0.5*2) - 1
        std=[2.0, 2.0, 2.0]      # 1 / 0.5
    )
    hr_tensor_unnormalized = un_norm_transform(hr_tensor.cpu())
    lr_transform = transforms.Compose([
        transforms.ToPILImage(), # This will now get a [0, 1] tensor
        transforms.Resize(lr_size, interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.Resize(target_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    lr_tensor = lr_transform(hr_tensor_unnormalized).to(device)
    
    # Add batch dimension
    return lr_tensor.unsqueeze(0), hr_tensor.unsqueeze(0)

# Converts [-1, 1] tensor to a [0, 1] tensor
def un_normalize(tensor):
    tensor = (tensor.clamp(-1, 1) + 1) / 2
    tensor = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    return tensor

# Runs reverse diffusion
def run_sampling_loop(model, diffusion, condition, cfg_scale=4.0):
    # Start with pure noise
    x_t = torch.randn_like(condition)
    
    print(f"Sampling {diffusion.timesteps} steps")
    
    with torch.no_grad():
        for t in reversed(range(diffusion.timesteps)):
            x_t = diffusion.sample_step(model, x_t, t, condition, cfg_scale=cfg_scale)
    return x_t


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Same config as in training
    config = {
        'data': {'path': '', 'hr_size': 128, 'scale_factor': 2},
        'model': {'in_channels': 3, 'model_channels': 64, 'time_emb_dim': 256},
        'diffusion': {'timesteps': 1000, 'beta_start': 1e-4, 'beta_end': 0.02}
    }
    
    MODEL_PATHS = [
        "models/model_epoch_50.pth",
        "models/model_epoch_100.pth",
        "models/model_epoch_150.pth",
        "models/model_epoch_200.pth",
        "models/model_epoch_250.pth",
        "models/model_epoch_300.pth",
        "models/model_epoch_350.pth",
        "models/model_epoch_400.pth",
        "models/model_epoch_450.pth",
        "models/model_epoch_500.pth" 
    ]
    HR_INPUT_PATH = "../data/datasets/soumikrakshit/div2k-high-resolution-images/versions/1/DIV2K_train_HR/DIV2K_train_HR/0016.png" 
    OUTPUT_IMAGE_PATH = "comparison_plot_all_models.png"
    
    # Create the dynamic subplot
    num_models = len(MODEL_PATHS)
    fig, ax = plt.subplots(num_models, 3, figsize=(15, num_models * 5))
    
    print(f"Loading base image: {HR_INPUT_PATH}")
    target_size = (config['data']['hr_size'], config['data']['hr_size'])
    scale_factor = config['data']['scale_factor']

    if not os.path.exists(HR_INPUT_PATH):
        print(f"Error: Input image not found at {HR_INPUT_PATH}")
        exit()
    
    condition, hr_original = load_and_prepare_image(HR_INPUT_PATH, target_size, scale_factor, device)
    
    diffusion = DiffusionModel(
        timesteps=config['diffusion']['timesteps'],
        device=device
    )

    # Loop through each model and plot
    for i, model_path in enumerate(MODEL_PATHS):
        if not os.path.exists(model_path):
            print(f"Warning: Model file not found at {model_path}. Skipping row {i+1}.")
            continue
            
        print(f"Processing Model {i+1}/{num_models}: {model_path}")
        
        # Load Model Architecture
        model = StandardUNet(
            in_channels=config['model']['in_channels'],
            model_channels=config['model']['model_channels'],
            time_emb_dim=config['model']['time_emb_dim']
        ).to(device)
        
        # Load Trained Weights
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        
        # Run inference
        predicted_x_t = run_sampling_loop(model, diffusion, condition, cfg_scale=4.0)
        
        # Original HR
        ax[i, 0].imshow(un_normalize(hr_original))
        ax[i, 0].set_title(f"Original (HR)")
        ax[i, 0].axis('off')
        
        # LR
        ax[i, 1].imshow(un_normalize(condition))
        ax[i, 1].set_title(f"(LR)")
        ax[i, 1].axis('off')
        
        # Predicted HR
        model_name = os.path.basename(model_path)
        ax[i, 2].imshow(un_normalize(predicted_x_t))
        ax[i, 2].set_title(f"Predicted ({model_name})")
        ax[i, 2].axis('off')

    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE_PATH)
    print(f"Comparison plot saved to {OUTPUT_IMAGE_PATH}")