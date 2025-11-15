import torch
import torch.nn as nn
import torch.nn.functional as F

# Sinusoidal PE
class TimeEmbedding(nn.Module):
    def __init__(self, time_emb_dim, n_channels):
        super().__init__
        
        self.n_channels = n_channels
        self.time_emb_dim = time_emb_dim

        self.lin1 = nn.Linear(time_emb_dim, n_channels)
        self.act = nn.SiLU()
        self.lin2 = nn.Linear(n_channels, n_channels)

    def forward(self, t):
        half_dim = self.time_emb_dim // 2
        emb = torch.exp(
            torch.arange(half_dim, device=t.device) * -torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        )
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=1)
        emb = self.act(self.lin1(emb))
        emb = self.lin2(emb)
        return emb
    
class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.act1 = nn.SiLU()
        
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.act2 = nn.SiLU()
        
        self.time_proj = nn.Linear(time_emb_dim, out_ch)

    def forward(self, x, t_emb):
        h = self.act1(self.norm1(self.conv1(x)))
    
        time_info = self.act1(self.time_proj(t_emb))
        h = h + time_info[:, :, None, None]
        
        h = self.act2(self.norm2(self.conv2(h)))
        return h

class StandardUNet(nn.Module):
    def __init__(self, in_channels=3, model_channels=64, time_emb_dim=256):
        super().__init__()
        
        self.time_mlp = TimeEmbedding(time_emb_dim)
        self.conv0 = nn.Conv2d(in_channels * 2, model_channels, 3, 1)
        
        self.down1 = Block(model_channels, model_channels, time_emb_dim)
        self.down2 = Block(model_channels, model_channels * 2, time_emb_dim)
        self.down3 = Block(model_channels * 2, model_channels * 4, time_emb_dim)
        self.pool = nn.MaxPool2d(2)

        self.bot = Block(model_channels * 4, model_channels * 8, time_emb_dim)

        self.up_conv1 = nn.ConvTranspose2d(model_channels * 8, model_channels * 4, 2, 2)
        self.up1 = Block(model_channels * 8, model_channels * 4, time_emb_dim) # Cat(4+4)
        
        self.up_conv2 = nn.ConvTranspose2d(model_channels * 4, model_channels * 2, 2, 2)
        self.up2 = Block(model_channels * 4, model_channels * 2, time_emb_dim) # Cat(2+2)
        
        self.up_conv3 = nn.ConvTranspose2d(model_channels * 2, model_channels, 2, 2)
        self.up3 = Block(model_channels * 2, model_channels, time_emb_dim) # Cat(1+1)

        self.out = nn.Conv2d(model_channels, in_channels, 1)

    def forward(self, x, t, condition):
        t_emb = self.time_mlp(t)
        x = torch.cat((x, condition), dim=1) # [B, 2*C, H, W]
        x1 = self.conv0(x)
        
        x2 = self.down1(x1, t_emb)
        x2_p = self.pool(x2)
        x3 = self.down2(x2_p, t_emb)
        x3_p = self.pool(x3)
        x4 = self.down3(x3_p, t_emb)
        x4_p = self.pool(x4)
        
        x_bot = self.bot(x4_p, t_emb)

        x_up1 = self.up_conv1(x_bot)
        x_up1_cat = torch.cat((x_up1, x4), dim=1) # Skip connection
        x_up1 = self.up1(x_up1_cat, t_emb)
        
        x_up2 = self.up_conv2(x_up1)
        x_up2_cat = torch.cat((x_up2, x3), dim=1) # Skip connection
        x_up2 = self.up2(x_up2_cat, t_emb)
        
        x_up3 = self.up_conv3(x_up2)
        x_up3_cat = torch.cat((x_up3, x2), dim=1) # Skip connection
        x_up3 = self.up3(x_up3_cat, t_emb)

        return self.out(x_up3)