import os
import os.path as osp
import sys 
sys.path.append("..") 

import torch
import trimesh
import numpy as np
from tqdm import tqdm
from .SIMMD.chamfer_distance import ChamferDistance
from .SIMMD.sampling import farthest_pts_sampling_tensor

from utils.visualization import write_ply, write_ply_rgb

def calculate_scenes_diversity(ref_dir, syn_dirs, num_pts=10240):
    """To quantify the diversity of the generated scenes,
        for each training example we calculated the standard devia-
        tion (std) of the intensity values of each voxel over 50 gen-
        erated scenes, averaged it over all voxels, and normalized
        by the std of the intensity values of the training scene.
    """
    syn_scenes = []
    for syn_dir in tqdm(syn_dirs):
        try:
            scene = trimesh.sample.sample_surface_even(trimesh.load(syn_dir), num_pts)[0]
            syn_scenes.append(torch.tensor(scene, device='cuda:0').float().unsqueeze(0))
        except:
            pass
    syn_scenes = torch.cat(syn_scenes, dim=0)

    sum_dist = 0
    dist_func = ChamferDistance()
    for j in tqdm(range(len(syn_scenes))):
        for k in range(j + 1, len(syn_scenes), 1):
            pc1 = syn_scenes[j].unsqueeze(0)
            pc2 = syn_scenes[k].unsqueeze(0)

            chamfer_dist = dist_func(pc1, pc2, transpose=False).detach().cpu().numpy()
            sum_dist += chamfer_dist
    geo_diversity = sum_dist * 2 / (len(syn_scenes) - 1)

    return geo_diversity
