import torch
import torch.nn as nn
import torch.nn.functional as F

# Sinusoidal Positional Embedding implementation
class TimeEmbedding(nn.Module):
    def __init__(self, time_emb_dim, n_channels):
        super().__init__()
        self.lin1 = nn.Linear(time_emb_dim, n_channels)
        self.act = nn.SiLU()
        self.lin2 = nn.Linear(n_channels, n_channels)

    def forward(self, t):
        half_dim = self.lin1.in_features // 2
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -torch.log(torch.tensor(10000.0)) / (half_dim - 1))
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=1)
        return self.lin2(self.act(self.lin1(emb)))

class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t):
        h = F.silu(self.norm1(self.conv1(x)))
        h = h + self.time_mlp(t)[:, :, None, None]
        h = F.silu(self.norm2(self.conv2(h)))
        return h + self.shortcut(x)

class SelfAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])

    def forward(self, x):
        b, c, h, w = x.shape
        x_flat = x.view(b, c, -1).permute(0, 2, 1)
        x_ln = self.ln(x_flat)
        attn_out, _ = self.mha(x_ln, x_ln, x_ln)
        return (attn_out + x_flat).permute(0, 2, 1).view(b, c, h, w)

class StandardUNet(nn.Module):
    def __init__(self, in_channels=3, model_channels=64, time_emb_dim=256, use_attention=True):
        super().__init__()
        self.use_attention = use_attention

        self.conv0 = nn.Conv2d(in_channels * 2, model_channels, 3, padding=1)
        self.time_mlp = TimeEmbedding(time_emb_dim, time_emb_dim)
        
        # Down blocks
        self.down1 = Block(model_channels, model_channels, time_emb_dim)
        self.down2 = Block(model_channels, model_channels * 2, time_emb_dim)
        self.down3 = Block(model_channels * 2, model_channels * 4, time_emb_dim)
        self.pool = nn.MaxPool2d(2)

        # Bottleneck with Attention
        self.bot1 = Block(model_channels * 4, model_channels * 8, time_emb_dim)
        if self.use_attention:
            self.attn = SelfAttention(model_channels * 8)
        self.bot2 = Block(model_channels * 8, model_channels * 4, time_emb_dim)

        # Up blocks
        self.up_conv1 = nn.ConvTranspose2d(model_channels * 4, model_channels * 4, 2, 2)
        self.up1 = Block(model_channels * 8, model_channels * 2, time_emb_dim)
        self.up_conv2 = nn.ConvTranspose2d(model_channels * 2, model_channels * 2, 2, 2)
        self.up2 = Block(model_channels * 4, model_channels, time_emb_dim)
        self.up_conv3 = nn.ConvTranspose2d(model_channels, model_channels, 2, 2)
        self.up3 = Block(model_channels * 2, model_channels, time_emb_dim)
        self.out = nn.Conv2d(model_channels, in_channels, 1)

    def forward(self, x, t, condition):
        t_emb = self.time_mlp(t)
        x = torch.cat((x, condition), dim=1)
        x1 = self.conv0(x)
        
        d1 = self.down1(x1, t_emb)
        d2 = self.down2(self.pool(d1), t_emb)
        d3 = self.down3(self.pool(d2), t_emb)
        
        mid = self.bot1(self.pool(d3), t_emb)
        if self.use_attention:
            mid = self.attn(mid)
        mid = self.bot2(mid, t_emb)
        
        u1 = self.up_conv1(mid)
        u1 = self.up1(torch.cat((u1, d3), dim=1), t_emb)
        u2 = self.up_conv2(u1)
        u2 = self.up2(torch.cat((u2, d2), dim=1), t_emb)
        u3 = self.up_conv3(u2)
        u3 = self.up3(torch.cat((u3, d1), dim=1), t_emb)
        return self.out(u3)