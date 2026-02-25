import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data import SuperResDataset
from model import StandardUNet
from diffusion import DiffusionModel
from torchvision.models import vgg16, VGG16_Weights

class VGGLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[:16].to(device).eval()
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg
        self.mse = nn.MSELoss()

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1))

    def forward(self, input, target):
        input_01 = (input + 1.0) / 2.0
        target_01 = (target + 1.0) / 2.0
        
        input_norm = (input_01 - self.mean) / self.std
        target_norm = (target_01 - self.mean) / self.std
        
        vgg_input = self.vgg(input_norm)
        vgg_target = self.vgg(target_norm)
        return self.mse(vgg_input, vgg_target)



def train(experiment_name, channels = 128, use_attention = True, schedule = "cosine", use_VGG = False):
    device = "cuda"
    epochs = 3000
    save_dir = f"./experiments/{experiment_name}"
    os.makedirs(save_dir, exist_ok=True)
    
    model = StandardUNet(model_channels=channels, use_attention=use_attention).to(device)
    diffusion = DiffusionModel(device=device, schedule_type=schedule)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    vgg_criterion = VGGLoss(device)

    # Resume Logic
    ckpt_path = os.path.join(save_dir, "latest.pth")
    start_epoch = 0
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch'] + 1
        print(f"Resuming from epoch {start_epoch}")
    1, 
    log_path = os.path.join(save_dir, "train_log.csv")
    if not os.path.exists(log_path):
        with open(log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'loss'])


    dataset = SuperResDataset("./data/datasets/soumikrakshit/div2k-high-resolution-images/versions/1/DIV2K_train_HR/DIV2K_train_HR", 256, 2)
    # dataset = SuperResDataset("./data/datasets/badasstechie/celebahq-resized-256x256/versions/1/train", 128, 2)

    loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4, pin_memory=True)

    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_loss = 0
        for lr_img, hr_img in loader:
            lr_img, hr_img = lr_img.to(device), hr_img.to(device)
            t = torch.randint(0, 1000, (hr_img.shape[0],), device=device)
            x_t, noise = diffusion.add_noise(hr_img, t)
            
            # CFG Dropout
            if torch.rand(1).item() < 0.1: lr_img = torch.zeros_like(lr_img)
            
            # MSE loss
            pred_noise = model(x_t, t, lr_img)
            loss_mse = criterion(noise, pred_noise)
            
            # VGG loss
            loss_vgg = torch.tensor(0.0, device=device)
            if use_VGG:
                s1 = diffusion.sqrt_alpha_hat[t][:, None, None, None]
                s2 = diffusion.sqrt_one_minus_alpha_hat[t][:, None, None, None]
                pred_x0 = (x_t - s2 * pred_noise) / s1
                pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)
                valid_t = t > 500
                if valid_t.any():
                    loss_vgg = vgg_criterion(pred_x0[valid_t], hr_img[valid_t])

            batch_loss = loss_mse + 0.01 * loss_vgg

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            epoch_loss += batch_loss.item()
        
        avg_loss = epoch_loss / len(loader)
        print(f"Epoch {epoch}: Loss {avg_loss:.6f}")

        # Log saving
        with open(log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, avg_loss])
        
        # Save checkpoints
        if epoch % 5 == 0:
            torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}, ckpt_path)
        if epoch % 500 == 0:
            torch.save(model.state_dict(), os.path.join(save_dir, f"model_{epoch}.pth"))

if __name__ == "__main__":
    # train(experiment_name = "attention_64ch_cosine_celebA", channels = 64, use_attention = True, schedule = "cosine", use_VGG=False)
    # train(experiment_name = "attention_64ch_cosine_VGG", channels = 64, use_attention = True, schedule = "cosine", use_VGG=True)
    train(experiment_name = "no_attention_128ch_linear", channels = 128, use_attention = False, schedule = "linear", use_VGG=False)