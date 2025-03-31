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

class MHCEncoder(nn.Module):
    
    def __init__(self, num_layers, in_dim, embed_dim, mhc_len, kernel_size=3):
        super().__init__()
        self.num_layers = num_layers
        self.in_dim = in_dim
        self.embed_dim = embed_dim
        self.mhc_len = mhc_len
        self.kernel_size = kernel_size
        self.padding = (kernel_size - 1) // 2
        
        self.conv = nn.Conv1d(self.in_dim, self.embed_dim, 1, 1, bias=False)
        self.layers = nn.ModuleList(
            [
                ResidueConvBlock(self.embed_dim, self.kernel_size, self.padding) for _ in range(self.num_layers - 1)
            ]
        )
        self.bn = nn.BatchNorm1d(self.embed_dim)
        
    def forward(self, x, mask):  # x: B, L, E
        x = x.transpose(1, 2)  # B, L, E => B, E, L
        x = self.conv(x)
        x = self.bn(x)
        
        for layer in self.layers:
            x = layer(x)
        x = x.transpose(1, 2) # B, E, L => B, L, E
        x_pooling = torch.mean(x, dim=1)
        return x, x_pooling, mask

class EpitopeEncoder(nn.Module):

    def __init__(self, num_layers, in_dim, embed_dim, esm_dim, epitope_len, kernel_size=3):
        super().__init__()

        self.num_layers = num_layers
        self.in_dim = in_dim
        self.embed_dim = embed_dim
        self.epitope_len = epitope_len
        self.kernel_size = kernel_size
        self.padding = (kernel_size - 1) // 2
        
        self.conv = nn.Conv1d(self.in_dim, self.embed_dim, 1, 1, bias=False)
        self.layers = nn.ModuleList(
            [
                ResidueConvBlock(self.embed_dim, self.kernel_size, self.padding) for _ in range(self.num_layers - 1)
            ]
        )
        self.bn = nn.BatchNorm1d(self.embed_dim)

        self.self_attention = nn.MultiheadAttention(embed_dim=self.embed_dim + esm_dim, num_heads=8, batch_first=False)

        self.ffn = nn.Sequential(
            nn.LayerNorm(self.embed_dim + esm_dim),
            nn.Linear(self.embed_dim + esm_dim, self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, self.embed_dim),
        )
    def forward(self, x, emb, mask):

        x = x.transpose(1, 2)  # B, L, E => B, E, L
        x = self.conv(x)
        x = self.bn(x)
        
        for layer in self.layers:
            x = layer(x)

        x = x.transpose(1, 2) # B, E, L => B, L, E

        x = torch.cat((x, emb), dim=-1)  # B, L, (E2 + E1)

        residual = x

        # self-attention
        x = x.transpose(0, 1) # B, L, E => L, B, E
        x, attn = self.self_attention(x, x, x, key_padding_mask=mask)
        x = x.transpose(0, 1) # L, B, E => B, L, E

        x =  residual + x
        x = self.ffn(x)

        x_pooling = torch.mean(x, dim=1)  # B, L, (E2 + E1) => B, (E2 + E1)

        return x, x_pooling, mask

class TCRAEncoder(nn.Module):

    def __init__(self, num_layers, in_dim, embed_dim, esm_dim, cdr3_len, fv_len, kernel_size=3):
        super().__init__()

        self.num_layers = num_layers
        self.in_dim = in_dim
        self.embed_dim = embed_dim
        self.cdr3_len = cdr3_len
        self.fv_len = fv_len
        self.kernel_size = kernel_size
        self.padding = (kernel_size - 1) // 2

        self.conv_cdr3 = nn.Conv1d(self.in_dim, self.embed_dim, 1, 1, bias=False)
        self.layers_cdr3 = nn.ModuleList(
            [
                ResidueConvBlock(self.embed_dim, self.kernel_size, self.padding) for _ in range(self.num_layers - 1)
            ]
        )
        self.bn_cdr3 = nn.BatchNorm1d(self.embed_dim)

        self.conv_fv = nn.Conv1d(self.in_dim, self.embed_dim, 1, 1, bias=False)
        self.layers_fv = nn.ModuleList(
            [
                ResidueConvBlock(self.embed_dim, self.kernel_size, self.padding) for _ in range(self.num_layers - 1)
            ]
        )
        self.bn_fv = nn.BatchNorm1d(self.embed_dim)

        self.self_attention = nn.MultiheadAttention(embed_dim=self.embed_dim + esm_dim, num_heads=8, batch_first=False)

        self.ffn = nn.Sequential(
            nn.LayerNorm(self.embed_dim + esm_dim),
            nn.Linear(self.embed_dim + esm_dim, self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, self.embed_dim),
        )

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
    
    def forward(self, x_cdr3, x_fv, emb, mask, position, return_attention=False):
        # position start from 0

        x_cdr3 = x_cdr3.transpose(1, 2)  # B, L1, E => B, E, L1
        x_cdr3 = self.conv_cdr3(x_cdr3)
        #x_cdr3 = self.bn_cdr3(x_cdr3)
        
        for layer in self.layers_cdr3:
            x_cdr3 = layer(x_cdr3)

        x_cdr3 = x_cdr3.transpose(1, 2) # B, E, L1 => B, L, E

        x_fv = x_fv.transpose(1, 2)  # B, L1, E => B, E, L1
        x_fv = self.conv_fv(x_fv)
        #x_fv = self.bn_fv(x_fv)
        
        for layer in self.layers_fv:
            x_fv = layer(x_fv)

        x_fv = x_fv.transpose(1, 2) # B, E, L => B, L, E

        B, L, E = x_fv.shape
        L1 = x_cdr3.shape[1]

        x_seq = torch.zeros_like(x_fv)


        batch_indices = torch.arange(B, device=position.device)[:, None].expand(B, L1)

        position_indices = (
            position[:, None] + 
            torch.arange(L1, device=position.device)[None, :]
        ).clamp(max=L-1).long()

        if torch.any(position_indices >= L):
            raise ValueError("Position and length of x_cdr3 exceed the length of x_fv")

        x_seq[batch_indices, position_indices] = x_fv[batch_indices, position_indices] + x_cdr3

        x = torch.cat((x_seq, emb), dim=-1)  # B, L, (E2 + E1)

        residual = x

        # self-attention
        x = x.transpose(0, 1) # B, L, E => L, B, E
        x, attn = self.self_attention(x, x, x, key_padding_mask=mask)
        x = x.transpose(0, 1) # L, B, E => B, L, E

        x =  residual + x
        x = self.ffn(x)

        x_pooling = torch.mean(x, dim=1)  # B, L, (E2 + E1) => B, (E2 + E1)

        if return_attention:
            return x, x_pooling, mask, self.logit_scale.exp(), attn
        else:
            return x, x_pooling, mask, self.logit_scale.exp()

class TCRBEncoder(nn.Module):

    def __init__(self, num_layers, in_dim, embed_dim, esm_dim, cdr3_len, fv_len, kernel_size=3):
        super().__init__()

        self.num_layers = num_layers
        self.in_dim = in_dim
        self.embed_dim = embed_dim
        self.cdr3_len = cdr3_len
        self.fv_len = fv_len
        self.kernel_size = kernel_size
        self.padding = (kernel_size - 1) // 2

        self.conv_cdr3 = nn.Conv1d(self.in_dim, self.embed_dim, 1, 1, bias=False)
        self.layers_cdr3 = nn.ModuleList(
            [
                ResidueConvBlock(self.embed_dim, self.kernel_size, self.padding) for _ in range(self.num_layers - 1)
            ]
        )
        self.bn_cdr3 = nn.BatchNorm1d(self.embed_dim)

        self.conv_fv = nn.Conv1d(self.in_dim, self.embed_dim, 1, 1, bias=False)
        self.layers_fv = nn.ModuleList(
            [
                ResidueConvBlock(self.embed_dim, self.kernel_size, self.padding) for _ in range(self.num_layers - 1)
            ]
        )
        self.bn_fv = nn.BatchNorm1d(self.embed_dim)

        self.self_attention = nn.MultiheadAttention(embed_dim=self.embed_dim + esm_dim, num_heads=8, batch_first=False)

        self.ffn = nn.Sequential(
            nn.LayerNorm(self.embed_dim + esm_dim),
            nn.Linear(self.embed_dim + esm_dim, self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, self.embed_dim),
        )

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
    
    def forward(self, x_cdr3, x_fv, emb, mask, position, return_attention=False):
        # position start from 0
        # mask for x_fv
        # emb structure embedding

        x_cdr3 = x_cdr3.transpose(1, 2)  # B, L1, E => B, E, L1
        x_cdr3 = self.conv_cdr3(x_cdr3)
        x_cdr3 = self.bn_cdr3(x_cdr3)
        
        for layer in self.layers_cdr3:
            x_cdr3 = layer(x_cdr3)

        x_cdr3 = x_cdr3.transpose(1, 2) # B, E, L1 => B, L, E

        x_fv = x_fv.transpose(1, 2)  # B, L1, E => B, E, L1
        x_fv = self.conv_fv(x_fv)
        x_fv = self.bn_fv(x_fv)
        
        for layer in self.layers_fv:
            x_fv = layer(x_fv)

        x_fv = x_fv.transpose(1, 2) # B, E, L => B, L, E

        B, L, E = x_fv.shape
        L1 = x_cdr3.shape[1]

        x_seq = torch.zeros_like(x_fv)

        batch_indices = torch.arange(B, device=position.device)[:, None].expand(B, L1)

        position_indices = (
            position[:, None] +
            torch.arange(L1, device=position.device)[None, :]
        ).clamp(max=L-1).long()

        if torch.any(position_indices >= L):
            raise ValueError("Position and length of x_cdr3 exceed the length of x_fv")

        x_seq[batch_indices, position_indices] = x_fv[batch_indices, position_indices] + x_cdr3

        x = torch.cat((x_seq, emb), dim=-1)  # B, L, (E2 + E1)

        residual = x

        # self-attention
        x = x.transpose(0, 1) # B, L, E => L, B, E
        x, attn = self.self_attention(x, x, x, key_padding_mask=mask)
        x = x.transpose(0, 1) # L, B, E => B, L, E

        x =  residual + x
        x = self.ffn(x)

        x_pooling = torch.mean(x, dim=1)  # B, L, (E2 + E1) => B, (E2 + E1)

        if return_attention:
            return x, x_pooling, mask, self.logit_scale.exp(), attn
        else:
            return x, x_pooling, mask, self.logit_scale.exp()