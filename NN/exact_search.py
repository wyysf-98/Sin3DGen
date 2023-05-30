import os
import os.path as osp
import numpy as np
from tqdm import tqdm
import torch
from typing import Union
import matplotlib.pyplot as plt

from utils.visualization import plot_grid_slices, write_ply, write_ply_rgb

@torch.no_grad()
def efficient_cdist(X, Y):
    dist = (X * X).sum(1)[:, None] + (Y * Y).sum(1)[None, :] - 2.0 * torch.mm(X, torch.transpose(Y, 0, 1))
    # normalize by size of vector to make dists independent of the size of d ( use same alpha for all patche-sizes)
    return dist / X.shape[-1] # DO NOT use torch.sqrt


@torch.no_grad()
def get_col_mins_efficient(dist_fn, X, Y, b=1024):
    n_batches = len(Y) // b
    mins = torch.zeros(Y.shape[0], dtype=X.dtype, device=X.device)
    for i in range(n_batches):
        mins[i * b:(i + 1) * b] = dist_fn(X, Y[i * b:(i + 1) * b]).min(0)[0]
    if len(Y) % b != 0:
        mins[n_batches * b:] = dist_fn(X, Y[n_batches * b:]).min(0)[0]

    return mins


@torch.no_grad()
def get_NNs_Dists(dist_fn, X, Y, alpha=None, b=1024):
    if alpha is not None:
        normalizing_row = get_col_mins_efficient(dist_fn, X, Y, b=b)
        normalizing_row = alpha + normalizing_row[None, :]
    else:
        normalizing_row = 1
    
    NNs = torch.zeros(X.shape[0], dtype=torch.long, device=X.device)
    Dists = torch.zeros(X.shape[0], dtype=torch.float, device=X.device)

    n_batches = len(X) // b
    for i in range(n_batches):
        dists = dist_fn(X[i * b:(i + 1) * b], Y) / normalizing_row
        NNs[i * b:(i + 1) * b] = dists.min(1)[1]
        Dists[i * b:(i + 1) * b] = dists.min(1)[0]
    if len(X) % b != 0:
        dists = dist_fn(X[n_batches * b:], Y) / normalizing_row
        NNs[n_batches * b:] = dists.min(1)[1]
        Dists[n_batches * b: ] = dists.min(1)[0]

    return NNs, Dists

@torch.no_grad()
def exact_search(content, 
                 exemplar, 
                 patch_size, 
                 mode="value",
                 alpha=None, 
                 dist_wrapper=None, 
                 chunk_size=4096,
                 num_iters=10,
                 num_coordinate_iters=1):
    dist_fn = lambda X, Y: dist_wrapper(efficient_cdist, X, Y) if dist_wrapper is not None else efficient_cdist(X, Y)

    if mode == "value":
        extract_patches_mode = "dense_wo_padding"
        combine_patches_mode = "blending"
    elif mode == "coordinate":
        extract_patches_mode = "dense"
        combine_patches_mode = "center_voxel"
        exemplar_geo_feat, exemplar_app_feat = exemplar.extract_patches(extract_patches_mode, 1, return_msk=False)
    elif mode == "value2coordinate":
        _NNs, Dists0 = exact_search(content, exemplar, patch_size, mode="value", alpha=alpha, dist_wrapper=dist_wrapper, num_iters=num_iters-num_coordinate_iters)
        NNs, Dists1 = exact_search(content, exemplar, patch_size, mode="coordinate", alpha=alpha, dist_wrapper=dist_wrapper, num_iters=num_coordinate_iters)
        return NNs, Dists0 + Dists1

    exemplar_geo_patch_feat, exemplar_app_patch_feat = exemplar.extract_patches(extract_patches_mode, patch_size, return_msk=False)

    Dists = []
    pbar = tqdm(total=num_iters, desc='Starting')
    for itr in range(1,  num_iters + 1):  
        content_geo_patch_feat, content_app_patch_feat = content.extract_patches(extract_patches_mode, patch_size, return_msk=False)

        nns, dists = get_NNs_Dists(dist_fn, 
                                   torch.cat([content_geo_patch_feat, content_app_patch_feat], dim=-1), 
                                   torch.cat([exemplar_geo_patch_feat, exemplar_app_patch_feat], dim=-1), 
                                   alpha=alpha, 
                                   b=chunk_size)
        pbar.update(1)
        pbar.set_description(f'[iter {itr}] score: {dists.mean().item():.6f}')
        Dists += [dists.mean().item()]

        # combine patches
        if combine_patches_mode == "blending":
            content.combine_patches(exemplar_geo_patch_feat[nns], exemplar_app_patch_feat[nns], mode="blending")
        elif combine_patches_mode == "center_voxel":
            content.combine_patches(exemplar_geo_feat[nns], exemplar_app_feat[nns], mode="center_voxel")

    NNF = exemplar.patches_center[nns]
    return NNF, Dists
