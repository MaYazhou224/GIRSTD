from mmseg.models.builder import BACKBONES, MODELS
import numpy as np

from .dgvfm_stem import DGVFM

from .dino_v2 import DinoVisionTransformer
from .utils import set_requires_grad, set_train
import os

import torch
import torch.nn as nn
from torch.nn import Dropout, Softmax, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair

import torch.nn.functional as F
from pytorch_wavelets import DWT2D
from torch.nn import TransformerEncoder, TransformerEncoderLayer



@BACKBONES.register_module()
class DGVFMDinoVisionTransformer(DinoVisionTransformer):
    def __init__(
        self,
        dgvfm_config=None,
        embed_dim = 1024,
        num_heads=16,
        **kwargs,
    ):
        super().__init__(embed_dim=embed_dim, num_heads=num_heads, **kwargs)
        self.dgvfm: DGVFM = MODELS.build(dgvfm_config)
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.zj_attn = Attention(embed_dim=embed_dim)  
        self.zj_norm_attn = nn.LayerNorm(embed_dim, eps=1e-6).to(torch.float32)
        self.fusion_conv = nn.Conv1d(embed_dim, embed_dim, kernel_size=1)
        self.zj_mlp_attn = nn.Linear(embed_dim, embed_dim)
        self.zj_wavepool = WavePool(1024)

   
    def forward_features(self, x, masks=None):  
        B, C, h, w = x.shape
        H, W = h // self.patch_size, w // self.patch_size
        x = self.prepare_tokens_with_masks(x, masks)

        outs = []
        skip_feats = {}

        for idx, blk in enumerate(self.blocks):
            B, N, C = x.shape
            h = int(np.sqrt(N))
            w = h
            
            if idx in [14, 17, 20, 23]:
                encoder_idx = {23:0, 20: 3, 17: 7, 14: 11}[idx]
                if encoder_idx in skip_feats:
                    enc = skip_feats[encoder_idx]
                    x_patch, enc_patch = x[:, 1:, :], enc[:, 1:, :]
                    cls_token, enc_trans = torch.tensor_split(enc, [1], dim=1)  
                    enc_trans = enc_trans.reshape(B, h, w, C).permute(0, 3, 1, 2)  
                    with torch.no_grad():
                        ll, lh, hl, hh = self.zj_wavepool(enc_trans)
                    n = int(N / 4)
                    ll, lh, hl, hh = ll.reshape(B, C, n), lh.reshape(B, C, n), hl.reshape(B, C, n), hh.reshape(B, C, n)
                    enc_freq = torch.cat([ll, lh, hl, hh], dim=-1).permute(0, 2, 1)  

                    Q = enc_patch
                    K = enc_freq
                    V = enc_freq

                    attn_out = self.zj_attn(Q, K, V)
                    attn_out = self.zj_norm_attn(self.zj_mlp_attn(attn_out + enc_patch))
                    gated = enc_patch * torch.sigmoid(attn_out) + attn_out * torch.sigmoid(enc_patch)
                    fused = self.fusion_conv(gated.transpose(1, 2)).transpose(1, 2) 
                    fused = fused + x_patch + enc_patch
                    x = torch.cat([x[:, :1, :], fused], dim=1)

            x = blk(x)            
            x = self.dgvfm.forward(x, idx, batch_first=True, has_cls_token=True)

            if idx in [0, 3, 7, 11]:
                skip_feats[idx] = x.detach()

            if idx in self.out_indices:
                outs.append(
                    x[:, 1:, :].permute(0, 2, 1).reshape(B, -1, H, W).contiguous()
                )
        return self.dgvfm.return_auto(outs)
    

    def train(self, mode: bool = True):
        if not mode:
            return super().train(mode)
        set_requires_grad(self, ["dgvfm","zj_"])
        set_train(self, ["dgvfm","zj_"])

    def state_dict(self, destination, prefix, keep_vars):
        state = super().state_dict(destination, prefix, keep_vars)
        keys = [k for k in state.keys() if 'dgvfm' not in k and 'zj_' not in k]
        for key in keys:
            state.pop(key)
            if key in destination:
                destination.pop(key)
        return state


@MODELS.register_module()
class Attention(nn.Module):
    def __init__(self, embed_dim, use_softmax=True, add_residual=True, add_norm=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.use_softmax = use_softmax
        self.add_residual = add_residual

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.mlp_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value):
        B, Nq, C = query.size()

        q = self.q_proj(query)    
        k = self.k_proj(key)      
        v = self.v_proj(value)    

        attn = torch.einsum("bnc,bmc->bnm", q, k)
        attn = F.instance_norm(attn)

        if self.use_softmax:
            attn = attn * (self.embed_dim ** -0.5)
            attn = F.softmax(attn, dim=-1)
        out = torch.einsum("bnm,bmc->bnc", attn, v)

        if self.add_residual:
            return self.mlp_proj(query + out)
        else:
            return out


def get_wav(in_channels, pool=True):
    harr_wav_L = 1 / np.sqrt(2) * np.ones((1, 2))
    harr_wav_H = 1 / np.sqrt(2) * np.ones((1, 2))
    harr_wav_H[0, 0] = -1 * harr_wav_H[0, 0]

    harr_wav_LL = np.transpose(harr_wav_L) * harr_wav_L
    harr_wav_LH = np.transpose(harr_wav_L) * harr_wav_H
    harr_wav_HL = np.transpose(harr_wav_H) * harr_wav_L
    harr_wav_HH = np.transpose(harr_wav_H) * harr_wav_H

    filter_LL = torch.from_numpy(harr_wav_LL).unsqueeze(0)
    filter_LH = torch.from_numpy(harr_wav_LH).unsqueeze(0)
    filter_HL = torch.from_numpy(harr_wav_HL).unsqueeze(0)
    filter_HH = torch.from_numpy(harr_wav_HH).unsqueeze(0)

    if pool:
        net = nn.Conv2d
    else:
        net = nn.ConvTranspose2d

    LL = net(in_channels, in_channels,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)
    LH = net(in_channels, in_channels,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)
    HL = net(in_channels, in_channels,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)
    HH = net(in_channels, in_channels,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)

    LL.weight.requires_grad = False
    LH.weight.requires_grad = False
    HL.weight.requires_grad = False
    HH.weight.requires_grad = False

    LL.weight.data = filter_LL.float().unsqueeze(0).expand(in_channels, -1, -1, -1).clone()
    LH.weight.data = filter_LH.float().unsqueeze(0).expand(in_channels, -1, -1, -1).clone()
    HL.weight.data = filter_HL.float().unsqueeze(0).expand(in_channels, -1, -1, -1).clone()
    HH.weight.data = filter_HH.float().unsqueeze(0).expand(in_channels, -1, -1, -1).clone()

    return LL, LH, HL, HH


@MODELS.register_module()
class WavePool(nn.Module):
    def __init__(self, in_channels):
        super(WavePool, self).__init__()
        self.LL, self.LH, self.HL, self.HH = get_wav(in_channels)

    def forward(self, x):
        return self.LL(x), self.LH(x), self.HL(x), self.HH(x)
    