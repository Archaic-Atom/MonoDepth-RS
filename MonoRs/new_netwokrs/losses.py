from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import math
import numpy as np


def LogDepth2Prob(Gt_depth, maxdepth=100.0, b=0.5):
    N, H, W = Gt_depth.shape
    log_gt = torch.log(Gt_depth.clamp(min=1e-6))
    
    depth_bins = torch.linspace(0, np.log(maxdepth), steps=int(maxdepth), device=Gt_depth.device)
    depth_bins = depth_bins.view(1, -1, 1, 1).repeat(N, 1, H, W)
    
    cost = -torch.abs(depth_bins - log_gt.unsqueeze(1)) / b
    return F.softmax(cost, dim=1)


def Adaptive_Multi_Modal_Cross_Entropy_Loss(x, depth, mask, maxdepth=100.0, m=1, n=9, top_k=9, epsilon=3, min_samples=1):

    assert depth.dim() == 4, "depth 应为 4D 张量 (N, C, H, W)"
    assert mask.dim() == 4, "mask 应为 4D 张量 (N, C, H, W)"
    

    x=x.squeeze(1)
    depth = depth.squeeze(1)  # (N, C=1, H, W) → (N, H, W)
    mask = mask.squeeze(1)     # (N, C=1, H, W) → (N, H, W)
    
    depth = depth.clamp(max=maxdepth)  # (N, H, W)
    N, H, W = depth.shape
    

    patch_h, patch_w = m, n
    depth_unfold = F.unfold(
        F.pad(depth.unsqueeze(1), (patch_w//2, patch_w//2, patch_h//2, patch_h//2), mode='reflect'),
        (patch_h, patch_w)
    ).view(N, patch_h*patch_w, H, W)
    depth_unfold_clone = depth_unfold.clone()
    

    mask_cluster = torch.zeros((N, patch_h*patch_w, patch_h*patch_w, H, W), device=depth.device).bool()
    for index in range(patch_h*patch_w):
        if index == 0:
            d_min = d_max = depth.unsqueeze(1)  # (N, 1, H, W)
        else:
            depth_unfold = depth_unfold * ~mask_cluster[:, index-1, ...]
            d_min = d_max = torch.max(depth_unfold, dim=1, keepdim=True)[0]
        
        mask_cluster[:, index, ...] = (depth_unfold > (d_min - epsilon).clamp(min=0)) & (depth_unfold < (d_max + epsilon).clamp(max=maxdepth))
        while True:
            d_min = torch.min(depth_unfold * mask_cluster[:, index, ...] + ~mask_cluster[:, index, ...] * maxdepth, dim=1, keepdim=True)[0]
            d_max = torch.max(depth_unfold * mask_cluster[:, index, ...], dim=1, keepdim=True)[0]
            mask_new = (depth_unfold > (d_min - epsilon).clamp(min=0)) & (depth_unfold < (d_max + epsilon).clamp(max=maxdepth))
            if torch.equal(mask_new, mask_cluster[:, index, ...]):
                break
            mask_cluster[:, index, ...] = mask_new
    

    depth_cluster = torch.mean(
        depth_unfold_clone.unsqueeze(1).repeat(1, patch_h*patch_w, 1, 1, 1) * mask_cluster, 
        dim=2
    ) * (patch_h*patch_w) / torch.sum(mask_cluster, dim=2).clamp(min=1)  # (N, patch_h*patch_w, H, W)
    
    GT = torch.zeros((N, patch_h*patch_w, int(maxdepth), H, W), device=depth.device)
    for index in range(patch_h*patch_w):
        GT[:, index, ...] = LogDepth2Prob(depth_cluster[:, index, ...], maxdepth)  # (N, maxdepth, H, W)
    

    mask_cluster = torch.sum(mask_cluster, dim=2, keepdim=True)  # (N, patch_h*patch_w, 1, H, W)
    mask_cluster[mask_cluster < min_samples] = 0
    
    w_cluster = 0.4 / (mask_cluster.sum(dim=1, keepdim=True) - 1).clamp(min=1) * mask_cluster  # (N, patch_h*patch_w, 1, H, W)
    w_cluster[:, 0, ...] += 0.6 - 0.4 / (mask_cluster.sum(dim=1, keepdim=False) - 1).clamp(min=1)
    

    top_k_values, top_k_indices = torch.topk(w_cluster, k=top_k, dim=1)
    w_cluster.scatter_(dim=1, index=top_k_indices, src=top_k_values)
    w_cluster = w_cluster / w_cluster.sum(dim=1, keepdim=True).clamp(min=1)  # (N, patch_h*patch_w, 1, H, W)
    

    GT = (GT * w_cluster).sum(dim=1)  # (N, maxdepth, H, W)
    GT = GT.detach()

    x = x.unsqueeze(1).repeat(1, int(maxdepth), 1, 1)  # (N, maxdepth, H, W)
    x = torch.log(x + 1e-30)

    mask = mask.unsqueeze(1).repeat(1, int(maxdepth), 1, 1)  # (N, maxdepth, H, W)
    num = mask.sum()

    loss = -(GT[mask] * x[mask]).sum() / num
    print(loss)
    return loss
