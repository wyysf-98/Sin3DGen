import os
import os.path as osp
import sys 
sys.path.append("..") 
import json
import yaml
import pandas as pd
import torch
import imageio
import argparse
import shutil
import pandas as pd
import numpy as np
from glob import glob
from tqdm import tqdm

from metrics.SIFID.sifid_score import calculate_sifid_given_paths
from metrics.NIQE.niqe_score import calculate_niqe_given_path
from metrics.SIMMD.simmd_score import calculate_simmd_given_paths
from metrics.images_diversity import calculate_images_diversity
from metrics.scenes_diversity import calculate_scenes_diversity

CLEAN_FLAG = False

args = argparse.ArgumentParser(description='evaluate the synthesis scenes')
args.add_argument('--exp', default='../comparison/graf_my/eval', type=str, help='synthesis images path.')
args.add_argument('--mesh_gt', default='../comparison/nerf/eval', type=str, help='GT path.')
args.add_argument('--img_gt', default='../comparison/nerf/eval', type=str, help='GT path.')
args.add_argument('--out_dir', default='./results/graf', type=str, help='results folder.')
args.add_argument('-d', '--device', default="cuda:0", help='device to use. (default: None)')

# calucated metrics
args.add_argument('--no_sifid', action='store_true', help='whether to calculate SIFID.')
args.add_argument('--no_swav_sifid', action='store_true', help='whether to calculate SwAV_SIFID.')
args.add_argument('--no_niqe', action='store_true', help='whether to calculate NIQE.')
args.add_argument('--no_image_diversity', action='store_true', help='whether to calculate Image Diversity.')
args.add_argument('--no_scene_simmd', action='store_true', help='whether to calculate Scene Geo & SH Quality using MMD.')
args.add_argument('--no_scene_diversity', action='store_true', help='whether to calculate Scene Geo & SH Diversity.')

args = args.parse_args()


def test(args):
    dset = args.exp.split('/')[3]
    syn_dirs = sorted(glob(osp.join(args.exp, '*')))[:50]
    print(dset, len(syn_dirs))
    
    if osp.exists(args.out_dir) and dset in [d for d in os.listdir(args.out_dir)]:
        print(f'{dset} already exists.')
        return

    # GT data
    img_gt_path, mesh_gt_path = args.img_gt, args.mesh_gt

    sifid_scores, swav_sifid_scores, niqe_scores, simmd_cd_score = [], [], [], []
    for i, syn_dir in tqdm(enumerate(syn_dirs)):
        out_str = f'current folder: {syn_dir}'
        print('==============>', out_str)

        if not args.no_sifid:
            print('calculating sifid score...')
            sifid = calculate_sifid_given_paths(img_gt_path, syn_dir, 64, 'InceptionV3')
            sifid_scores.append(sifid)
            out_str += f' sifid: {sifid}'

        if not args.no_swav_sifid:
            print('calculating swav_sifid score...')
            swav_sifid = calculate_sifid_given_paths(img_gt_path, syn_dir, 64, 'SwAV')
            swav_sifid_scores.append(swav_sifid)
            out_str += f' swav_sifid: {swav_sifid}'
        
        if not args.no_niqe:
            print('calculating niqe score...')
            niqe = calculate_niqe_given_path(syn_dir)
            niqe_scores.append(niqe)
            out_str += f' niqe: {niqe}'

        if not args.no_scene_simmd:
            print('calculating simmd score...')
            try:
                simmd_cd = calculate_simmd_given_paths(mesh_gt_path, osp.join('/'.join(syn_dir.split('/')[:-3]), "meshes", syn_dir.split('/')[-1] + '.obj'), num_patches=1000, dist_fn='ChamferDistance')
            except:
                simmd_cd = np.nan
            simmd_cd_score.append(simmd_cd)
            out_str += f' simmd_cd: {simmd_cd}'

        print(out_str)
        
    if not args.no_image_diversity:
        # calculate images diversity
        images_diversity_score = calculate_images_diversity(img_gt_path, syn_dirs)
        print(f'images_diversity: {images_diversity_score}')


    if not args.no_scene_diversity:
        scenes_geo_diversity_score = calculate_scenes_diversity(mesh_gt_path, [osp.join('/'.join(syn_dir.split('/')[:-3]), "meshes", syn_dir.split('/')[-1] + '.obj') for syn_dir in syn_dirs])
        print(f'scenes_geo_diversity: {scenes_geo_diversity_score}')

    # create output folder and save to csv
    out_dir = args.out_dir + f'/{dset}'
    os.makedirs(out_dir, exist_ok=True)
    total_res_file = f'{osp.dirname(out_dir)}/results.csv'
    res = {
        'folder': syn_dirs,
    }
    res_str = f"{os.path.basename(args.exp)}: "
    total_res = {
        'exp': dset,
    }
    if osp.exists(total_res_file):
        total_df = pd.read_csv(total_res_file)
    else:
        total_df = pd.DataFrame(columns=['exp', 'SIFID', 'SWAV_SIFID', 'NIQE', 'Image Diversity', 'SIMMD_CD', 'Scene Geometric Diversity'])

    if not args.no_sifid:
        res['sifid'] = sifid_scores
        sifid = np.mean(sifid_scores)
    if not args.no_scene_simmd:
        res['simmd_cd'] = simmd_cd_score
        simmd_cd = np.nanmean(simmd_cd_score)
    if not args.no_swav_sifid:
        res['swav_sifid'] = swav_sifid_scores
        swav_sifid = np.mean(swav_sifid_scores)
    if not args.no_niqe:
        res['niqe'] = niqe_scores
        niqe = np.mean(niqe_scores)
    print(res, len(sifid_scores), len(simmd_cd_score), len(swav_sifid_scores), len(niqe_scores))

    df = pd.DataFrame(res)
    if not args.no_sifid:
        df['SIFID'] = [sifid] + [np.nan] * (len(syn_dirs) - 1)
        res_str += f", SIFID: {sifid:.6f}"
        total_res['SIFID'] = sifid
    if not args.no_scene_simmd:
        df['SIMMD_CD'] = [simmd_cd] + [np.nan] * (len(syn_dirs) - 1)
        res_str += f", SIMMD_CD: {simmd_cd:.6f}"
        total_res['SIMMD_CD'] = simmd_cd
    if not args.no_swav_sifid:
        df['SWAV_SIFID'] = [swav_sifid] + [np.nan] * (len(syn_dirs) - 1)
        res_str += f", SWAV_SIFID: {swav_sifid:.6f}"
        total_res['SWAV_SIFID'] = swav_sifid
    if not args.no_niqe:
        df['NIQE'] = [niqe] + [np.nan] * (len(syn_dirs) - 1)
        res_str += f", NIQE: {niqe:.6f}"
        total_res['NIQE'] = niqe
    if not args.no_image_diversity:
        df['Image Diversity'] = [images_diversity_score] + [np.nan] * (len(syn_dirs) - 1)
        res_str += f", Image Diversity: {images_diversity_score:.6f}"
        total_res['Image Diversity'] = images_diversity_score
    if not args.no_scene_diversity:
        df['Scene Geometric Diversity'] = [scenes_geo_diversity_score] + [np.nan] * (len(syn_dirs) - 1)
        res_str += f", Scene Geometric Diversity: {scenes_geo_diversity_score:.6f}"
        total_res['Scene Geometric Diversity'] = scenes_geo_diversity_score

    df.to_csv(f'{out_dir}/results.csv', index=False)
    total_df = total_df.append(total_res, ignore_index=True)
    total_df.to_csv(total_res_file, index=False)

if __name__ == '__main__':
    test(args)
