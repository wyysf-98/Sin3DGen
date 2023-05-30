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
args.add_argument('-s', '--seed', default=None,
                  type=int, help='random seed used. (default: None)')
args.add_argument('-d', '--device', default="cuda:0",
                  type=str, help='device to use. (default: None)')

# synthesis config
args.add_argument('--pyr_reso',  
                  type=str, help='List of grid resolution')
args.add_argument('--rescale_bounds',  
                  type=str, help='Enlarge the bouding box')
args.add_argument('--rewrite_bounds',  
                  type=str, help='User pre-defined the bouding box')
args.add_argument('--jitter_once', 
                  type=int, help='whether to jitter only at the first level')
args.add_argument('--noise_type', choices=["uniform", "gaussian"],
                  type=str, help='noise type')
args.add_argument('--noise', 
                  type=float, help='noise level')

# scene config
args.add_argument('--geo_feat_type', choices=["raw", "scaling", "sdf", "tsdf"],
                  type=str, help='geometry feature type')
args.add_argument('--truncate_scale', 
                  type=int, help='truncate sdf to scale*voxel_size, use when geo_feat_type=="tsdf".')
args.add_argument('--app_feat_type', choices=["raw", "scaling", "pca"],
                  type=str, help='appearence feature type')
args.add_argument('--n_components', 
                  type=int, help='number of PCA components, use when app_feat_type=="pca".')
args.add_argument('--w_app', 
                  type=float, help='weight of appearence feature.')

# NN search config 
args.add_argument('--alpha',
                  help='completeness/diversity trade-off alpha.')
args.add_argument('--patch_size', 
                  type=int, help='patch size.')
args.add_argument('--coarse_iters', 
                  type=int, help='num of iterations at the coarest level.')
args.add_argument('--ENNF_iters',
                  type=int, help='number of iterations for exact NN search.')
args.add_argument('--Coord_iters',
                  type=int, help='number of iterations for Coordinate-based search in exact NN search.')
args.add_argument('--start_ANNF_level', 
                  type=int, help='start level of approximate NN search.')
args.add_argument('--ANNF_iters',
                  type=int, help='number of iterations for approximate NN search.')
args.add_argument('--PM_iters',
                  type=int, help='number of iterations for PatchMatch search in approximate NN search.')

# for visualization
args.add_argument('--start_vis_level',
                  type=int, help='start level for rendering results.')
args.add_argument('--scene_reso',
                  type=str, help='visualization resolution.')
args.add_argument('--trajectory',
                  type=str, help='visualize trajectory path.')
args.add_argument('--fps', 
                  type=int, help='video fps.')
args.add_argument('--vis_mapping_field',
                  type=int, help='Whether to visualize the optimized mapping field.')
args.add_argument('--only_vis_surface',
                  type=int, help='whether to only vis voxels near surface.')
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

def generate(cfg):
    print(f"{cfg} \n{cfg.device} is being used for synthesising...")

    # prepare exp folder
    prefix = f"{cfg.output}/{cfg.exemplar.split('/')[-2]}"
    prefix += f"/rewrite_bounds{list_to_str(cfg.rewrite_bounds)}" if cfg.rewrite_bounds else f"/rescale_bounds{list_to_str(cfg.rescale_bounds)}"
    prefix += f"+{list_to_str(cfg.pyr_reso)}"
    prefix += "/s%s+sANNF%s+%s%s+geo_%s+app_%s+w_app%s+alpha%s+ps%s+citers%s+jonce_%s" % (
        cfg.seed,
        cfg.start_ANNF_level,
        cfg.noise_type,
        cfg.noise,
        cfg.geo_feat_type if cfg.geo_feat_type != "tsdf" else f"tsdf{cfg.truncate_scale}",
        cfg.app_feat_type if "pca" not in cfg.app_feat_type else f"{cfg.app_feat_type}{cfg.n_components}",
        cfg.w_app,
        cfg.alpha,
        cfg.patch_size,
        cfg.coarse_iters,
        cfg.jitter_once)
    if cfg.mode == 'run' or cfg.mode == 'eval':
        pass
    elif cfg.mode == 'debug':
        debug_path = f'{prefix}/debug'
    

    # get synthesising space
    if cfg.rewrite_bounds is None:
        reso = cfg.pyr_reso[-1]
        tmp = svox2.SparseGrid.load(f"{cfg.exemplar}/ckpt_reso{list_to_str([reso] * 3)}.npz", device=cfg.device)
        valid_voxel_inds = (tmp.links >= 0).nonzero()
        valid_voxel_bounds = (valid_voxel_inds.max(0)[0] - valid_voxel_inds.min(0)[0] + 1).cpu().numpy() * cfg.rescale_bounds
        ratio = valid_voxel_bounds / valid_voxel_bounds.max()
        del tmp
    else:
        ratio = np.array(cfg.rewrite_bounds) / np.array(cfg.rewrite_bounds).max()

    # perform synthesising
    for lvl, reso in tqdm(enumerate(np.array(cfg.pyr_reso))):
        # get mesh
        mesh_path = f"{cfg.exemplar}/mesh_reso{list_to_str([reso] * 3)}.obj"
        if not osp.exists(mesh_path):
            print(f"=====> No mesh found in {mesh_path}, start generating mesh ...")
            tmp = PlenoxelsWrapper({"ckpt": f"{cfg.exemplar}/ckpt_reso{list_to_str([reso] * 3)}.npz"}, device=cfg.device, verbose=False)
            tmp.extract_mesh(mesh_path, floodfill=True)
            del tmp

        # get synthesising resolution
        syn_reso = np.ceil(np.array([reso] * 3) * ratio).astype(np.int32)
        syn_reso = np.clip(syn_reso, cfg.patch_size, reso)
        ratio = radius = syn_reso / syn_reso.max()

        # get exemplar
        E = PlenoxelsWrapper({
            "ckpt": f"{cfg.exemplar}/ckpt_reso{list_to_str([reso] * 3)}.npz",
            "reso": syn_reso,
            "radius": radius, # each voxel must be a cube
            "geo_feat_type": cfg.geo_feat_type,
            "truncate_scale": cfg.truncate_scale,
            "app_feat_type": cfg.app_feat_type,
            "n_components": cfg.n_components,
            }, cfg.device
        )
        if cfg.mode == 'debug':
            E.visualization(f"{debug_path}/{list_to_str(syn_reso)}/E", cfg.trajectory, fps=cfg.fps)

        # create S space
        if lvl == 0:
            print(f'=====> create S with resolution: {syn_reso}')
            S = CoordinateMap(reso=syn_reso, radius=radius, device=cfg.device)
            if cfg.mode == 'debug':
                render_scene(f'{debug_path}/init_S', E, S, cfg.trajectory, cfg.vis_mapping_field, None, cfg.sdf_thres, cfg.fps)
        else:
            print(f'=====> upsample S from {S.reso.cpu().numpy()} to {syn_reso}')
            if cfg.mode == 'debug':
                render_scene(f'{debug_path}/{list_to_str(syn_reso)}/S_before_upsample', PlenoxelsWrapper({"ckpt": f"{cfg.exemplar}/ckpt_reso{list_to_str(cfg.scene_reso)}.npz"}, device=cfg.device, verbose=False), 
                                S, cfg.trajectory, cfg.vis_mapping_field, mesh_path if cfg.only_vis_surface else None, cfg.sdf_thres, cfg.fps)
            S.upsample(syn_reso, radius=radius)
            if cfg.mode == 'debug':
                render_scene(f'{debug_path}/{list_to_str(syn_reso)}/S_after_upsample', PlenoxelsWrapper({"ckpt": f"{cfg.exemplar}/ckpt_reso{list_to_str(cfg.scene_reso)}.npz"}, device=cfg.device, verbose=False), 
                                S, cfg.trajectory, cfg.vis_mapping_field, mesh_path if cfg.only_vis_surface else None, cfg.sdf_thres, cfg.fps)

        # jitter
        if lvl == 0 or not cfg['jitter_once']:
            S.jitter(cfg["noise"] / 2**lvl, cfg.noise_type)
            if cfg.mode == 'debug':
                render_scene(f'{debug_path}/{list_to_str(syn_reso)}/S_jitter', E, S, cfg.trajectory, cfg.vis_mapping_field, None, cfg.sdf_thres, cfg.fps)

        # define the distance function wrapper
        def dist_wrapper(dist_fn, X, Y):
            X_geo, Y_geo = X[..., 0].reshape(*X.shape[:-2], -1), Y[..., 0].reshape(*Y.shape[:-2], -1)
            X_app, Y_app = X[..., 1:].reshape(*X.shape[:-2], -1), Y[..., 1:].reshape(*Y.shape[:-2], -1)
            
            if cfg.w_app == 0.0:
                dist = dist_fn(X_geo, Y_geo)
            elif cfg.w_app == 1.0:
                dist = dist_fn(X_app, Y_app)
            else:
                dist_geo = dist_fn(X_geo, Y_geo)
                dist_app = dist_fn(X_app, Y_app)
                dist = dist_geo * (1 - cfg.w_app) + dist_app * cfg.w_app

            return dist
            
        # synthesis E using S and perform NN search
        S_E = S.create_scene(E, use_grid_data=False, upsample_ratio=1.0)
        if lvl < cfg.start_ANNF_level: # exact NN search
            num_iters = cfg.coarse_iters if lvl == 0 else cfg.ENNF_iters

            save_name = f"{prefix}/ENNF_reso{list_to_str(syn_reso)}+itrs{num_iters}"
            NNF, Dists = exact_search(S_E, 
                                      E, 
                                      cfg.patch_size,
                                      mode="value2coordinate",
                                      alpha=cfg.alpha, 
                                      dist_wrapper=dist_wrapper, 
                                      num_iters=num_iters,
                                      num_coordinate_iters=cfg.Coord_iters)
        elif lvl >= cfg.start_ANNF_level: # approximate NN search
            num_iters = cfg.coarse_iters if lvl == 0 else cfg.ANNF_iters
            save_name = f"{prefix}/ANNF_reso{list_to_str(syn_reso)}+itrs{num_iters}"
            NNF, Dists = approximate_search(S_E, 
                                            E, 
                                            cfg.patch_size, 
                                            dist_wrapper=dist_wrapper, 
                                            num_iters=num_iters,
                                            num_PM_iters=cfg.PM_iters)

        S.update(S_E.patches_center, E.grid2world(NNF))
        del S_E, E, NNF

        # save results
        os.makedirs(save_name, exist_ok=True)
        plt.plot(Dists)  
        plt.savefig(f'{save_name}/loss.png')
        plt.close()

        S.save(f'{save_name}/S.npz')
        if lvl >= cfg.start_vis_level:
            render_scene(save_name, PlenoxelsWrapper({"ckpt": f"{cfg.exemplar}/ckpt_reso{list_to_str(cfg.scene_reso)}.npz"}, device=cfg.device, verbose=False), 
                            S, cfg.trajectory, cfg.vis_mapping_field, cfg.only_vis_surface, None, cfg.sdf_thres, cfg.fps)
        
        torch.cuda.empty_cache()

if __name__ == "__main__":
    generate(cfg)