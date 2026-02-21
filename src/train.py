import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data import SuperResDataset
from model import StandardUNet
from diffusion import DiffusionModel

def train(experiment_name, channels = 128, use_attention = True, schedule = "cosine"):
    device = "cuda"
    epochs = 3000
    save_dir = f"./experiments/{experiment_name}"
    os.makedirs(save_dir, exist_ok=True)
    
    model = StandardUNet(model_channels=channels, use_attention=use_attention).to(device)
    diffusion = DiffusionModel(device=device, schedul_type=schedule)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    
    # Resume Logic
    ckpt_path = os.path.join(save_dir, "latest.pth")
    start_epoch = 0
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch'] + 1
        print(f"Resuming from epoch {start_epoch}")

    dataset = SuperResDataset("../data/datasets/soumikrakshit/div2k-high-resolution-images/versions/1/DIV2K_train_HR/DIV2K_train_HR", 256, 2)
    loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4, pin_memory=True)

    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0
        for lr_img, hr_img in loader:
            lr_img, hr_img = lr_img.to(device), hr_img.to(device)
            t = torch.randint(0, 1000, (hr_img.shape[0],), device=device)
            x_t, noise = diffusion.add_noise(hr_img, t)
            
            # CFG Dropout
            if torch.rand(1).item() < 0.1: lr_img = torch.zeros_like(lr_img)
            
            pred_noise = model(x_t, t, lr_img)
            loss = criterion(noise, pred_noise)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch}: Loss {total_loss/len(loader):.6f}")
        
        # Save checkpoints
        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}, ckpt_path)
        if epoch % 500 == 0:
            torch.save(model.state_dict(), os.path.join(save_dir, f"model_{epoch}.pth"))

if __name__ == "__main__":
    train(experiment_name = "attention_128ch_linear", channels = 128, use_attention = True, schedule = "linear")