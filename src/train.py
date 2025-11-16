import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from data import SuperResDataset
from model import StandardUNet
from diffusion import DiffusionModel

def train(config):
    ### Setup ###
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_path = config['data']['path']
    hr_size = config['data']['hr_size']
    scale_factor = config['data']['scale_factor']
    
    ### Load Data ###
    print("Loading dataset...")
    dataset = SuperResDataset(data_path, hr_size, scale_factor)
    dataloader = DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=4
    )
    
    ### Build Model ###
    print("Building models...")
    model = StandardUNet(
        in_channels=config['model']['in_channels'],
        model_channels=config['model']['model_channels'],
        time_emb_dim=config['model']['time_emb_dim']
    ).to(device)
    
    ### Build Diffusion Helper ###
    diffusion = DiffusionModel(
        timesteps=config['diffusion']['timesteps'],
        beta_start=config['diffusion']['beta_start'],
        beta_end=config['diffusion']['beta_end'],
        device=device
    )

    ### Setup Training ###
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    scheduler = CosineAnnealingLR(optimizer, T_max=config['training']['epochs'])
    mse = nn.MSELoss()
    
    print("Starting training...")
    for epoch in range(config['training']['epochs']):
        epoch_loss = 0
        for step, (lr_batch, hr_batch) in enumerate(dataloader):
            optimizer.zero_grad()
            
            lr_batch = lr_batch.to(device)
            hr_batch = hr_batch.to(device)
            
            t = torch.randint(0, diffusion.timesteps, (hr_batch.shape[0],), device=device)
            
            x_t, epsilon = diffusion.add_noise(hr_batch, t) # Epsilon is the true noise
            
            predicted_noise = model(x_t, t, lr_batch)
            
            loss = mse(epsilon, predicted_noise)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
        
        scheduler.step()
        avg_loss = epoch_loss / len(dataloader)
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | LR: {current_lr:.6f}")

        # Checkpoint
        if (epoch + 1) % 50 == 0:
            print(f"Saving checkpoint at epoch {epoch+1}...")
            os.makedirs(config['training']['save_dir'], exist_ok=True)
            save_path = os.path.join(config['training']['save_dir'], f"model_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Checkpoint saved to {save_path}")
        
    ### Save the model ###
    os.makedirs(config['training']['save_dir'], exist_ok=True)
    save_path = os.path.join(config['training']['save_dir'], f"model_epoch_{epoch+1}.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    dataset_dir = '../data/datasets/soumikrakshit/div2k-high-resolution-images/versions/1/DIV2K_train_HR'
    config = {
        'data': {'path': dataset_dir, 'hr_size': 128, 'scale_factor': 2},
        'model': {'in_channels': 3, 'model_channels': 64, 'time_emb_dim': 256},
        'diffusion': {'timesteps': 1000, 'beta_start': 1e-4, 'beta_end': 0.02},
        'training': {'epochs': 50, 'batch_size': 16, 'learning_rate': 1e-4, 'save_dir': 'models'}
    }
    
    train(config)