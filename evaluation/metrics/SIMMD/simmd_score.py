import os
import os.path as osp
import sys 
sys.path.append("..") 
from joblib import Parallel, delayed

import trimesh
import torch
import numpy as np
from tqdm import tqdm

from knn_cuda import KNN
from utils.visualization import write_ply, write_ply_rgb

from .sampling import farthest_pts_sampling_tensor
from .earth_mover_distance import EarthMoverDistance
from .chamfer_distance import ChamferDistance

def random_extract_patches_from_scene(scene, patch_size=1024, num_patches=1000, device='cuda:0'):
    print(scene.shape, scene.max(), scene.min())

    scene = torch.tensor(scene, device=device)
    # random select patches
    sel_indices = np.random.choice(scene.shape[0], num_patches, replace=False)
    knn = KNN(k=patch_size, transpose_mode=True)
    # print(scene.unsqueeze(0).shape, scene[sel_indices].unsqueeze(0).shape)
    dist, indx = knn(scene.unsqueeze(0), scene[sel_indices].unsqueeze(0))
    # print(dist.shape, indx.shape)

    pcs = scene[indx.squeeze(0)]

    # rgb = pcs[:, 0, :]
    # rgb = torch.randn((num_patches, 3), device=device)
    # rgb = 255 * (rgb - rgb.min(0)[0]) / (rgb.max(0)[0] - rgb.min(0)[0])
    # print(torch.cat([pcs.view(-1, 3), rgb.unsqueeze(1).repeat(1, patch_size, 1).view(-1, 3)], dim=-1).shape)
    # write_ply_rgb(torch.cat([pcs.view(-1, 3), rgb.unsqueeze(1).repeat(1, patch_size, 1).view(-1, 3)], dim=-1).cpu().numpy(), 'patches.ply')
    # print(rgb.shape)

    return pcs.float()


def calculate_simmd_given_paths(ref_dir, syn_dir, patch_size=1024, num_patches=1000, dist_fn='ChamferDistance', save=True, device='cuda:0'):
    """To quantify the quality of the generated scenes, for each training example 
       we extract patches and then calculated the minimum matching similarity 
       between these patches.
    """
    print('num_patches: ', num_patches, 'ref_dir: ', ref_dir, 'syn_dir: ', syn_dir)
    
    ref_patches_path = ref_dir.replace('.obj', f'_ps_{patch_size}_np_{num_patches}.npy')
    if not osp.exists(ref_patches_path):
        ref_scene = trimesh.sample.sample_surface_even(trimesh.load(ref_dir), 102400)[0]
        # write_ply(ref_scene, 'ref_scene.ply')
        ref_patches = random_extract_patches_from_scene(ref_scene, patch_size, num_patches)
        if save:
            np.save(ref_patches_path, ref_patches.cpu().numpy())
        del ref_scene
    else:
        ref_patches = torch.tensor(np.load(ref_patches_path), device=device).float()

    syn_patches_path = syn_dir.replace('.obj', f'_ps_{patch_size}_np_{num_patches}.npy')
    if not osp.exists(syn_patches_path):
        syn_scene = trimesh.sample.sample_surface_even(trimesh.load(syn_dir), 102400)[0]
        syn_patches = random_extract_patches_from_scene(syn_scene, patch_size, num_patches)
        if save:
            np.save(syn_patches_path, syn_patches.cpu().numpy())
        del syn_scene
    else:
        syn_patches = torch.tensor(np.load(syn_patches_path), device=device).float()

    # ref_patches = torch.from_numpy(ref_patches).float().cuda().view(num_patches, -1, 3)
    # syn_patches = torch.from_numpy(syn_patches).float().cuda().view(num_patches, -1, 3)
    print('ref_patches: ', ref_patches.shape, 'syn_patches: ', syn_patches.shape)
    assert ref_patches.shape == syn_patches.shape

    if dist_fn == 'EarthMoverDistance':
        dist_func = EarthMoverDistance()
    elif dist_fn == 'ChamferDistance':
        dist_func = ChamferDistance()
    else:
        raise NotImplementedError('dist_mode not supported')


    matched_dists = []
    for i in tqdm(range(num_patches)):
        ref_i = ref_patches[i].unsqueeze(0).repeat(syn_patches.shape[0], 1, 1)
        dists = dist_func(ref_i, syn_patches, transpose=False, reduce_mean=False)
        matched_dists.append(dists.min().cpu().numpy())
    simmd = np.mean(np.array(matched_dists))

    return simmd
