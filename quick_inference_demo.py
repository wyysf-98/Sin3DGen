import os
import os.path as osp
import svox2
import argparse
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils.base import ConfigParser, list_to_str
from utils.plenoxels_wrapper import PlenoxelsWrapper
from utils.coordinate_map import CoordinateMap

from NN.exact_search import exact_search
from NN.approximate_search import approximate_search

args = argparse.ArgumentParser(description='Random shuffle the input scene')
args.add_argument('-m', '--mode', default='run', choices=['run', 'eval', 'debug'],
                  type=str, help='current mode. ')
args.add_argument('-e', '--exemplar', required=True,
                  type=str, help='exemplar scene path.')
args.add_argument('-r', '--resume', default=None,
                  type=str, help='resume folder path.')
args.add_argument('-o', '--output', required=False,
                  type=str, help='output folder path.')
args.add_argument('-c', '--config', default='./configs/default.yaml',
                  type=str, help='config file path.')
args.add_argument('-d', '--device', default="cuda:0",
                  type=str, help='device to use. (default: None)')

# for visualization
args.add_argument('--start_vis_level',
                  type=int, help='start level for rendering results.')
args.add_argument('--scene_reso',
                  type=str, help='visualization resolution.')
args.add_argument('--trajectory',
                  type=str, help='visualize trajectory path.')
args.add_argument('--fps', type=int, help='video fps.')
args.add_argument('--vis_mapping_field',
                  type=bool, help='Whether to visualize the optimized mapping field.')
args.add_argument('--only_vis_surface',
                  type=bool, help='whether to only vis voxels near surface.')
args.add_argument('--sdf_thres',
                  type=float, help='use when only_vis_surface==True.')

cfg = ConfigParser(args)


def render_scene(save_path, exemplar, coordinate_map, trajectory_path, vis_mapping_field=False, extract_mesh=False, mesh_path=None, sdf_thres=0.0, fps=8):
    scene = coordinate_map.create_scene(exemplar, use_grid_data=True)
    scene.visualization(f"{save_path}/scene", trajectory_path, fps=fps)

    if vis_mapping_field:
        if extract_mesh:
            mesh_path = f"{save_path}/mesh.obj"
            scene.extract_mesh(mesh_path, floodfill=False)
        coordinate_map.visualization(f"{save_path}/coordinate_map", trajectory_path, mesh_path, sdf_thres, fps=fps)

    del scene


if __name__ == "__main__":
    S = CoordinateMap.load(cfg.resume, device=cfg.device)
    render_scene(cfg.output, PlenoxelsWrapper({"ckpt": f"{cfg.exemplar}/ckpt_reso{list_to_str(cfg.scene_reso)}.npz"}, device=cfg.device, verbose=False), 
                    S, cfg.trajectory, cfg.vis_mapping_field, cfg.only_vis_surface, None, cfg.sdf_thres, cfg.fps)
    torch.cuda.empty_cache()
