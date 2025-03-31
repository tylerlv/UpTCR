import torch
import torch.nn.functional as F
from torch import nn, Tensor
import numpy as np
class ResidueConvBlock(nn.Module):
    def __init__(self, embed_dim, kernel_size, padding, dropout=0.0):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim, kernel_size,
                      stride=1, padding=padding),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        )

    def forward(self, x):
        residual = x
        x = self.layer(x)
        x = residual + x

        return x

class pMHCFusion(nn.Module):

    def __init__(self, num_layers, embed_dim, kernel_size=3):
        super().__init__()

        self.num_layers = num_layers
        self.in_dim = embed_dim
        self.embed_dim = embed_dim
        self.kernel_size = kernel_size
        self.padding = (kernel_size - 1) // 2

        self.resconv = nn.ModuleList(
            [
                ResidueConvBlock(self.embed_dim, self.kernel_size, self.padding) for _ in range(self.num_layers - 1)
            ]
        )

        self.bn = nn.BatchNorm1d(self.embed_dim)

        self.self_attention = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=8, batch_first=False)
        self.ffn = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, self.embed_dim),
        )

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
    
    def forward(self, mhc, epitope, mhc_mask, epitope_mask, return_attention=False):
        x = torch.cat([mhc, epitope], dim=1)
        mask = torch.cat([mhc_mask, epitope_mask], dim=1)

        x = x.transpose(1,2) # B.L,E => B,E,L

        for layer in self.resconv:
            x = layer(x)
        
        # BatchNorm
        x = self.bn(x)
        x = x.transpose(1,2) # B, E, L => B, L, E

        residual = x

        # self-attention
        x = x.transpose(0, 1) # B, L, E => L, B, E
        x, attn = self.self_attention(x, x, x, key_padding_mask=mask)
        x = x.transpose(0, 1) # L, B, E => B, L, E

        # Feed-Forward Network
        x = residual + x
        x = self.ffn(x)

        # Pooling
        x_pooling = x.mean(dim=1)

        if return_attention:
            return x, x_pooling, mask, self.logit_scale.exp(), attn
        else:
            return x, x_pooling, mask, self.logit_scale.exp()

class TCRABFusion(nn.Module):

    def __init__(self, num_layers, embed_dim, kernel_size=3):
        super().__init__()

        self.num_layers = num_layers
        self.in_dim = embed_dim
        self.embed_dim = embed_dim
        self.kernel_size = kernel_size
        self.padding = (kernel_size - 1) // 2

        self.resconv = nn.ModuleList(
            [
                ResidueConvBlock(self.embed_dim, self.kernel_size, self.padding) for _ in range(self.num_layers - 1)
            ]
        )

        self.bn = nn.BatchNorm1d(self.embed_dim)

        self.self_attention = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=8, batch_first=False)
        self.ffn = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, self.embed_dim),
        )

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
    
    def forward(self, tcra, tcrb, tcra_mask, tcrb_mask, return_attention=False):
        x = torch.cat([tcra, tcrb], dim=1)
        mask = torch.cat([tcra_mask, tcrb_mask], dim=1)

        x = x.transpose(1,2) # B.L,E => B,E,L

        for layer in self.resconv:
            x = layer(x)
        
        # BatchNorm
        x = self.bn(x)
        x = x.transpose(1,2) # B, E, L => B, L, E

        residual = x

        # self-attention
        x = x.transpose(0, 1) # B, L, E => L, B, E
        x, attn = self.self_attention(x, x, x, key_padding_mask=mask)
        x = x.transpose(0, 1) # L, B, E => B, L, E
        
        # Feed-Forward Network
        x = residual + x
        x = self.ffn(x)

        # Pooling
        x_pooling = x.mean(dim=1)

        if return_attention:
            return x, x_pooling, mask, self.logit_scale.exp(), attn
        else:
            return x, x_pooling, mask, self.logit_scale.exp()