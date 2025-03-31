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

class DistPredictor(nn.Module):
    def __init__(self, hid_dim, out_dim):
        super().__init__()
        self.dist_map_predictor = nn.Conv2d(
            in_channels=hid_dim,
            out_channels=out_dim,
            kernel_size=3,
            padding=1
        )

    def forward(self, feat_A, feat_B, A_padding_mask, B_padding_mask):
        inter_padding_mask = torch.logical_or(A_padding_mask[:, :].unsqueeze(2), B_padding_mask[:, :].unsqueeze(1))  # B, L1, L2
        inter_padding_mask = torch.logical_not(inter_padding_mask)
        feat_A_mat = feat_A[:, :, :].unsqueeze(3).repeat([1, 1, 1, feat_B.shape[2]]) # B, E, L1, L2
        feat_B_mat = feat_B[:, :, :].unsqueeze(2).repeat([1, 1, feat_A.shape[2], 1]) # B, E, L1, L2
        
        inter_map = feat_A_mat * feat_B_mat  # B, E, L1, L2
        inter_map = inter_map.masked_fill(inter_padding_mask.unsqueeze(1), float("-inf"))
        dist_map = self.dist_map_predictor(F.relu(inter_map))
        out_dist = F.relu(dist_map[:, 0, :, :])
        
        return out_dist.unsqueeze(-1)

class Finetuner_Structure(nn.Module):

    def __init__(self, embed_dim, num_layers, dropout):
        super().__init__()

        self.embed_dim = embed_dim

        self.tcra_conv = nn.Conv1d(embed_dim, embed_dim, 1, 1, bias=False)
        self.tcrb_conv = nn.Conv1d(embed_dim, embed_dim, 1, 1, bias=False)
        self.epi_conv = nn.Conv1d(embed_dim, embed_dim, 1, 1, bias=False)
        self.mhc_conv = nn.Conv1d(embed_dim, embed_dim, 1, 1, bias=False)

        self.tcra_layer_norm = nn.LayerNorm(self.embed_dim)
        self.tcrb_layer_norm = nn.LayerNorm(self.embed_dim)
        self.epi_layer_norm = nn.LayerNorm(self.embed_dim)
        self.mhc_layer_norm = nn.LayerNorm(self.embed_dim)

        self.tcra_tcrb_dist_predictor = DistPredictor(self.embed_dim, 1)
        self.epi_tcra_dist_predictor = DistPredictor(self.embed_dim, 1)
        self.epi_tcrb_dist_predictor = DistPredictor(self.embed_dim, 1)
        self.tcra_mhc_dist_predictor = DistPredictor(self.embed_dim, 1)
        self.tcrb_mhc_dist_predictor = DistPredictor(self.embed_dim, 1)
        self.epi_mhc_dist_predictor = DistPredictor(self.embed_dim, 1)

        self.tcra_conv_layers = nn.ModuleList(
            [
                ResidueConvBlock(embed_dim=self.embed_dim, kernel_size=1, padding=0, dropout=dropout) # add dropout
                for _ in range(num_layers)
            ]
        )

        self.tcrb_conv_layers = nn.ModuleList(
            [
                ResidueConvBlock(embed_dim=self.embed_dim, kernel_size=1, padding=0, dropout=dropout) # add dropout
                for _ in range(num_layers)
            ]
        )

        self.epi_conv_layers = nn.ModuleList(
            [
                ResidueConvBlock(embed_dim=self.embed_dim, kernel_size=1, padding=0, dropout=dropout) # add dropout
                for _ in range(num_layers)
            ]
        )

        self.mhc_conv_layers = nn.ModuleList(
            [
                ResidueConvBlock(embed_dim=self.embed_dim, kernel_size=1, padding=0, dropout=dropout) # add dropout
                for _ in range(num_layers)
            ]
        )

    def forward(self, tcra, tcrb, epitope, mhc, tcra_mask, tcrb_mask, epitope_mask, mhc_mask):
        tcra = self.tcra_conv(tcra.transpose(1,2)).transpose(1,2)
        tcrb = self.tcrb_conv(tcrb.transpose(1,2)).transpose(1,2)
        epitope = self.epi_conv(epitope.transpose(1,2)).transpose(1,2)
        mhc = self.mhc_conv(mhc.transpose(1,2)).transpose(1,2)

        tcra = tcra + self.tcra_layer_norm(tcra)
        tcrb = tcrb + self.tcrb_layer_norm(tcrb)
        epitope = epitope + self.epi_layer_norm(epitope)
        mhc = mhc + self.mhc_layer_norm(mhc)

        tcra = tcra.transpose(1,2)
        tcrb = tcrb.transpose(1,2)
        epitope = epitope.transpose(1,2)
        mhc = mhc.transpose(1,2)

        tcra_mhc_dist = self.tcra_mhc_dist_predictor(tcra, mhc, tcra_mask, mhc_mask)
        tcrb_mhc_dist = self.tcrb_mhc_dist_predictor(tcrb, mhc, tcrb_mask, mhc_mask)
        tcra_tcrb_dist = self.tcra_tcrb_dist_predictor(tcra, tcrb, tcra_mask, tcrb_mask)
        epi_tcra_dist = self.epi_tcra_dist_predictor(epitope, tcra, epitope_mask, tcra_mask)
        epi_tcrb_dist = self.epi_tcrb_dist_predictor(epitope, tcrb, epitope_mask, tcrb_mask)
        epi_mhc_dist = self.epi_mhc_dist_predictor(epitope, mhc, epitope_mask, mhc_mask)

        return tcra_mhc_dist, tcrb_mhc_dist, tcra_tcrb_dist, epi_tcra_dist, epi_tcrb_dist, epi_mhc_dist

class Finetuner_TCRABpMHC(nn.Module):

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
        self.clf = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, tcrab, pmhc, tcrab_mask, pmhc_mask, return_attention=False):
        x = torch.cat([tcrab, pmhc], dim=1)
        mask = torch.cat([tcrab_mask, pmhc_mask], dim=1)

        x = x.transpose(1,2) # B.L,E => B,E,L

        for layer in self.resconv:
            x = layer(x)
        
        x = self.bn(x)
        x = x.transpose(1,2) # B, E, L => B, L, E

        residual = x

        # self-attention
        x = x.transpose(0, 1) # B, L, E => L, B, E
        x, attn = self.self_attention(x, x, x, key_padding_mask=mask)
        x = x.transpose(0, 1) # L, B, E => B, L, E

        # Feed-Forward Network
        x = residual + x
        x = x.mean(dim=1)

        x = self.clf(x).reshape([-1])

        if return_attention:
            return attn, mask
        else:
            return x

class Finetuner_pMHC(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()

        self.embed_dim = embed_dim

        self.clf = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, pmhc, pmhc_mask):
        x = pmhc.mean(dim=1)
        x = self.clf(x).reshape([-1])
        return x

class Finetuner_TCRBpMHC(nn.Module):

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
        self.clf = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, tcrb, pmhc, tcrb_mask, pmhc_mask):
        x = torch.cat([tcrb, pmhc], dim=1)
        mask = torch.cat([tcrb_mask, pmhc_mask], dim=1)

        x = x.transpose(1,2)

        for layer in self.resconv:
            x = layer(x)
        
        x = self.bn(x)
        x = x.transpose(1,2)

        residual = x

        x = x.transpose(0, 1)
        x, attn = self.self_attention(x, x, x, key_padding_mask=mask)
        x = x.transpose(0, 1)

        x = residual + x
        x = x.mean(dim=1)

        x = self.clf(x).reshape([-1])

        return x

class Finetuner_TCRBp(nn.Module):

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
        self.clf = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, tcrb, p, tcrb_mask, p_mask):
        x = torch.cat([tcrb, p], dim=1)
        mask = torch.cat([tcrb_mask, p_mask], dim=1)

        x = x.transpose(1,2) # B.L,E => B,E,L

        for layer in self.resconv:
            x = layer(x)
        
        x = self.bn(x)
        x = x.transpose(1,2) # B, E, L => B, L, E

        residual = x

        # self-attention
        x = x.transpose(0, 1) # B, L, E => L, B, E
        x, attn = self.self_attention(x, x, x, key_padding_mask=mask)
        x = x.transpose(0, 1) # L, B, E => B, L, E

        # Feed-Forward Network
        x = residual + x
        x = x.mean(dim=1)

        x = self.clf(x).reshape([-1])

        return x


class Finetuner_TCRABp(nn.Module):

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
        self.clf = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, tcrab, p, tcrab_mask, p_mask):
        x = torch.cat([tcrab, p], dim=1)
        mask = torch.cat([tcrab_mask, p_mask], dim=1)

        x = x.transpose(1,2) # B.L,E => B,E,L

        for layer in self.resconv:
            x = layer(x)
        
        x = self.bn(x)
        x = x.transpose(1,2) # B, E, L => B, L, E

        residual = x

        # self-attention
        x = x.transpose(0, 1) # B, L, E => L, B, E
        x, attn = self.self_attention(x, x, x, key_padding_mask=mask)
        x = x.transpose(0, 1) # L, B, E => B, L, E

        # Feed-Forward Network
        x = residual + x

        tcrab_size = tcrab.shape[1]
        tcrab_output = x[:, :tcrab_size, :]
        p_output = x[:, tcrab_size:, :]
        tcrab_output = tcrab_output.mean(dim=1)
        p_output = p_output.mean(dim=1)

        x = x.mean(dim=1)

        x = self.clf(x).reshape([-1])


        return x, tcrab_output, p_output

