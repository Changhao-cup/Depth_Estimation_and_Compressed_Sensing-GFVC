# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

from torch import nn
import torch.nn.functional as F
import torch
from modules.util import * 
import cv2
import numpy as np
from torch.autograd import Variable
import torchvision.transforms as transforms
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
class FlowAttention(nn.Module):
    def __init__(self,
                 dim,
                 bias=True,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 norm_layer=nn.LayerNorm):

        super().__init__()
        self.dim = dim
        self.scale = qk_scale or dim**-0.5

        self.norm1 = norm_layer(dim)
        # define the projection layer
        self.q_proj = nn.Linear(dim, dim, bias=bias)
        self.k_proj = nn.Linear(dim//dim, dim, bias=bias)
        self.v_proj = nn.Linear(dim - 1, dim + 2, bias=bias)

        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(dim+2, dim+2)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * 2)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU)

        self.grids = {}

    def generate_grid(self, B, H, W, normalize=True):
        yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W))
        if normalize:
            yy = yy / (H - 1)
            xx = xx / (W - 1)
        grid = torch.stack([xx, yy], dim=0)
        grid = grid[None].expand(B, -1, -1, -1)
        return grid

    def forward(self, x, tgt, return_attn=False):
        """
        Args:
            x: input features with shape of (B, C, H, W)
        """
        B, C, H, W = x.shape
        grid = self.grids.get(f"{H}_{W}")
        if grid is None:
            grid = self.generate_grid(B, H, W).to(x)
            # grid = self.generate_grid(B, H, W, normalize=False).to(x)
            self.grids[f"{H}_{W}"] = grid.clone()
        grid = grid.flatten(2).permute(0, 2, 1)

        x = x.flatten(2).permute(0, 2, 1)
        k = tgt.flatten(2).permute(0, 2, 1)
        v = torch.cat([tgt.flatten(2).permute(0, 2, 1), grid], dim=-1)

        x = self.norm1(x)
        shortcut = x

        q = self.q_proj(x)  # [B, N, C]
        k = self.k_proj(k)
        v = self.v_proj(v)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)  # [B, H, N, N]

        x = attn @ v
        x = self.proj_drop(self.proj(x))  # .view(B, H, W, 2).permute(0, 3, 1, 2)

        # mlp
        flow = x[..., -2:] - grid
        x = x[..., :-2]
        x = shortcut + x  # [B, N, :-2], global warped features, [B, N, -2:]: correspondence
        x = x + self.mlp(self.norm2(x))

        x = torch.cat([x, flow], dim=-1)
        x = x.view(B, H, W, -1).permute(0, 3, 1, 2)  # [B, C+2, H, W]
        return x
def warp(x, flo):

    """
    warp an image/tensor (im2) back to im1, according to the optical flow
    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow  
    if x=img2, flow(pre,cur) x warp flow==pre
    if x=img1, flow(cur,pre) x warp flow==cur
    """
    B, C, H, W = x.size()
    # mesh grid 
    xx = torch.arange(0, W).view(1,-1).repeat(H,1)
    yy = torch.arange(0, H).view(-1,1).repeat(1,W)
    xx = xx.view(1,1,H,W).repeat(B,1,1,1)
    yy = yy.view(1,1,H,W).repeat(B,1,1,1)
    grid = torch.cat((xx,yy),1).float()
    
    #x = x
    
    if torch.cuda.is_available():  
        x = x.cuda()#to('cuda:0')
        
        grid = grid.cuda()  #to('cuda:0')
        
    #flo = flo.cuda().data.cpu()
    flo = flo.permute(0,3,1,2)#from B,H,W,2 -> B,2,H,W
    
    #pixel flow motion
    vgrid = Variable(grid) + flo # B,2,H,W

    # scale grid to [-1,1] 
    vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:].clone()/max(W-1,1)-1.0 
    vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:].clone()/max(H-1,1)-1.0 
    vgrid = vgrid.permute(0,2,3,1)     #from B,2,H,W -> B,H,W,2

    output = F.grid_sample(x, vgrid,mode="bilinear",padding_mode="zeros")

    return output
