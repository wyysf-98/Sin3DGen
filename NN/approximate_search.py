import os
import os.path as osp
import copy
import numpy as np
from tqdm import tqdm
import torch
from typing import Union
import matplotlib.pyplot as plt

from utils.base import sample_neighbors, batchify_query
from utils.visualization import plot_grid_slices

@torch.no_grad()
def efficient_pdist(X, Y):
    if len(X.shape) == len(Y.shape) == 2:
        dist = (X * X).sum(1) + (Y * Y).sum(1) - 2.0 * torch.matmul(X.unsqueeze(1), Y.unsqueeze(2)).flatten()
    else:
        dist = (X * X).sum(-1) + (Y * Y).sum(-1) - 2.0 * torch.matmul(X, Y.transpose(1, 2)).reshape(X.shape[0], -1)
    return dist / X.shape[-1]


@torch.no_grad()
def compute_dists(dist_fn, content, exemplar, NNF, patch_size, normalizing_row=None, b=1024000):
    def _compute_dists(nnf):
        content_patch_feat = torch.cat(content.fetch_feat(nnf[:, :3], patch_size, return_msk=False), dim=-1)
        exemplar_patch_feat = torch.cat(exemplar.fetch_feat(nnf[:, 3:], patch_size, return_msk=False), dim=-1)
        if normalizing_row is None:
            return dist_fn(content_patch_feat, exemplar_patch_feat)
        else:
            return dist_fn(content_patch_feat, exemplar_patch_feat) / normalizing_row[nnf[:, 3:].unbind(-1)]
    dists = batchify_query(_compute_dists, torch.cat([content.patches_center, NNF], dim=-1), chunk_size=b)

    return dists


@torch.no_grad()
def patch_match(dist_fn, content, exemplar, patch_size, alpha=None, allow_diagonals=False, num_iters=5):
    if alpha is not None:
        _, normalizing_row = patch_match(dist_fn, exemplar, content, patch_size, allow_diagonals=allow_diagonals, num_iters=num_iters)
        normalizing_row = (alpha + normalizing_row).view(*exemplar.reso)
    else:
        normalizing_row = None
    
    extract_patches_mode = "dense"
    exemplar_geo_feat, exemplar_app_feat = exemplar.extract_patches(extract_patches_mode, 1, return_msk=False)
    content_geo_feat, content_app_feat = content.extract_patches(extract_patches_mode, 1, return_msk=False)
    del exemplar_geo_feat, exemplar_app_feat, content_geo_feat, content_app_feat

    ## step1: initionalization stage
    inds = torch.randint(high=exemplar.patches_center.shape[0], size=(content.patches_center.shape[0],), dtype=torch.int64).to(exemplar.device)
    NNF = exemplar.patches_center[inds]
    dists = compute_dists(dist_fn, content, exemplar, NNF, patch_size, normalizing_row=normalizing_row)

    for pm_itr in range(num_iters):
        ## step2: propagate stage
        for jump in [8, 4, 2, 1]: # In each iteration, improve the NNs, by jumping flooding.
            offsets = sample_neighbors(np.linspace(-jump, jump, 3), allow_center=False, allow_diagonals=allow_diagonals).to(exemplar.device).long()

            for offset in offsets:
                onnf = torch.roll(NNF.view(*content.reso, 3), shifts=tuple(-offset), dims=(0, 1, 2)).view(-1, 3)
                onnf = torch.max(torch.min(onnf - offset, exemplar.reso.unsqueeze(0) - 1), torch.zeros((1, 3)).to(exemplar.device)).long()
                onnf_dists = compute_dists(dist_fn, content, exemplar, onnf, patch_size, normalizing_row=normalizing_row)

                # improve guess
                update_msk = onnf_dists < dists
                NNF[update_msk] = onnf[update_msk]
                dists[update_msk] = onnf_dists[update_msk]
        del offsets, onnf, onnf_dists, update_msk

        ## step3: random search stage
        max_radius = max(exemplar.reso.cpu().numpy())
        for r in np.floor(max_radius / np.power(2, np.arange(np.log2(max_radius)))).astype(np.int32):
            rand_offsets = torch.randint(low=-r, high=r+1, size=(NNF.shape[0], 3), device=exemplar.device)

            rnnf = NNF + rand_offsets 
            rnnf = torch.max(torch.min(rnnf, exemplar.reso.unsqueeze(0) - 1), torch.zeros((1, 3)).to(exemplar.device)).long()
            rand_dists = compute_dists(dist_fn, content, exemplar, rnnf, patch_size, normalizing_row=normalizing_row)

            # improve guess
            update_msk = rand_dists < dists
            NNF[update_msk] = rnnf[update_msk]
            dists[update_msk] = rand_dists[update_msk]
        del rand_offsets, rnnf, rand_dists, update_msk

    return NNF, dists

@torch.no_grad()
def approximate_search(content, 
                       exemplar, 
                       patch_size, 
                       alpha=None, 
                       dist_wrapper=None, 
                       chunk_size=1024,
                       num_iters=10,
                       num_PM_iters=10):
    dist_fn = lambda X, Y: dist_wrapper(efficient_pdist, X, Y) if dist_wrapper is not None else efficient_pdist(X, Y)

    Dists = []
    pbar = tqdm(total=num_iters, desc='Starting')
    for itr in range(1,  num_iters + 1):  
        NNF, dists = patch_match(dist_fn,
                                 content,
                                 exemplar,
                                 patch_size=patch_size,
                                 alpha=alpha,
                                 num_iters=num_PM_iters)

        pbar.update(1)
        pbar.set_description(f'[iter {itr}] score: {dists.mean().item():.6f}')
        Dists += [dists.mean().item()]

        # update content 
        content.combine_patches(*exemplar.fetch_feat(NNF, 1, return_msk=False), mode="center_voxel")

    return NNF, Dists

