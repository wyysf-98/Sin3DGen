import os
import os.path as osp
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import copy
import mcubes
import trimesh
import imageio
import unfoldNd
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pysdf import SDF
from tqdm import tqdm
from typing import Union, List, NamedTuple, Optional, Tuple

import svox2
from utils.sparse_grid import SparseGrid
from utils.base import sample_neighbors, batchify_query_np, list_to_str, is_pow3

class PlenoxelsWrapper(SparseGrid):
    def __init__(
        self, 
        cfg: dict,
        device: Union[torch.device, str] = "cpu",
        verbose: bool = True,
    ):
        if "ckpt" in cfg.keys():
            if verbose:
                print(f'Loading scene from checkpoint {cfg["ckpt"]}...')
            grid = svox2.SparseGrid.load(cfg['ckpt'], device=device)
            super(PlenoxelsWrapper, self).__init__(cfg["reso"] if "reso" in cfg.keys() else grid._grid_size(), \
                                        cfg["radius"] if "radius" in cfg.keys() else grid.radius, \
                                        cfg["center"] if "center" in cfg.keys() else grid.center, \
                                        attributes = ["geo_links", "app_links"], \
                                        device=device)
            
            self.cfg = cfg
            self.grid = grid
        else:
            if verbose:
                print(f'Creating an empty scene...')
            super(PlenoxelsWrapper, self).__init__(attributes = ["geo_links", "app_links"], device=device)
            self.cfg = cfg
            self.grid = None          

        self.device = device
        self.verbose = verbose

        self.preprocess_feat()


    def __repr__(self):
        return (
            f"PlenoxelsWrapper, "
            + f"reso={self.reso.detach().cpu().numpy().tolist()}, "
            + f"radius={self.radius.detach().cpu().numpy().tolist()}, "
            + f"center={self.center.detach().cpu().numpy().tolist()}, "
            + f"capacity:{self.capacity}), "
            + f"geo_data:{self.geo_data.shape}), "
            + f"app_data:{self.app_data.shape})"
        )


    def _fetch_links(self, grid, return_msk=False):
        assert (~self.is_inside(grid)).sum() == 0
        geo_feat = torch.ones((grid.shape[0], self.geo_data.shape[-1]), dtype=torch.float32, device=self.device) * self.geo_empty_feat
        app_feat = torch.ones((grid.shape[0], self.app_data.shape[-1]), dtype=torch.float32, device=self.device) * self.app_empty_feat
        
        geo_links = self.geo_links[grid.unbind(-1)]
        geo_msk = geo_links >= 0
        geo_feat[geo_msk] = self.geo_data[geo_links[geo_msk].long()]

        app_links = self.app_links[grid.unbind(-1)]
        app_msk = app_links >= 0
        app_feat[app_msk] = self.app_data[app_links[app_msk].long()]

        if return_msk:
            return geo_feat.detach(), app_feat.detach(), geo_msk, app_msk
        else:
            return geo_feat.detach(), app_feat.detach()


    def _update_links_and_data(self, geo_feat, app_feat):
        assert geo_feat.shape[0] == app_feat.shape[0]

        geo_msk = abs(geo_feat.flatten() - self.geo_empty_feat) > self.EPS
        geo_links = torch.cumsum(geo_msk.to(torch.int32), dim=-1).int() - 1
        geo_links[~geo_msk] = -1
        self.geo_links = geo_links.view(*self.reso)
        self.geo_data = geo_feat[geo_msk].detach()

        app_msk = abs(app_feat.sum(-1).flatten() - self.app_empty_feat) > self.EPS
        app_links = torch.cumsum(app_msk.to(torch.int32), dim=-1).int() - 1
        app_links[~app_msk] = -1
        self.app_links = app_links.view(*self.reso)
        self.app_data = app_feat[app_msk].detach()

    def save(self, path: str, compress: bool = False):
        """
        Save to a path
        """
        save_fn = np.savez_compressed if compress else np.savez
        data = {
            "reso":self.reso.detach().cpu().numpy(),
            "radius":self.radius.detach().cpu().numpy(),
            "center":self.center.detach().cpu().numpy(),
            "capacity":self.capacity.detach().cpu().numpy(),
            "geo_links":self.geo_links.detach().cpu().numpy(),
            "geo_feat_type":self.geo_feat_type,
            "geo_data":self.geo_data.detach().cpu().numpy(),
            "geo_scale":self.geo_scale.detach().cpu().numpy(),
            "app_links":self.app_links.detach().cpu().numpy(),
            "app_data":self.app_data.detach().cpu().numpy(),
            "app_feat_type":self.app_feat_type,
            "app_scale":self.app_scale.detach().cpu().numpy(),
        }
        save_fn(
            path,
            **data
        )


    @classmethod
    def load(cls, path: str, device: Union[torch.device, str] = "cpu"):
        """
        Load from path
        """
        z = np.load(path)

        for attr in ["reso", "radius", "center", "capacity", "geo_links", "geo_data", "app_links", "app_data"]:
            assert attr in z.files 

        grid = cls(
            cfg={},
            device=device,
        )
        grid.capacity = z.f.capacity
        grid.reso = torch.from_numpy(z.f.reso).to(device=device)
        grid.radius = torch.from_numpy(z.f.radius).to(device=device)
        grid.center = torch.from_numpy(z.f.center).to(device=device)
        grid.geo_links = torch.from_numpy(z.f.geo_links).to(device=device)
        grid.geo_data = torch.from_numpy(z.f.geo_data).to(device=device)
        grid.geo_feat_type = z.f.geo_feat_type
        grid.geo_scale = torch.from_numpy(z.f.geo_scale).to(device=device)
        grid.app_links = torch.from_numpy(z.f.app_links).to(device=device)
        grid.app_feat_type = z.f.app_feat_type
        grid.app_data = torch.from_numpy(z.f.app_data).to(device=device)
        grid.app_scale = torch.from_numpy(z.f.app_scale).to(device=device)

        if grid.links.is_cuda:
            grid._accelerate()
        return grid


    def forward(self, grid, return_msk=False):
        assert grid.dtype is torch.long, f"query type must be torch.long, got: {grid.dtype}"
        geo_feat = torch.ones((grid.shape[0], 1), dtype=torch.float32, device=self.device) * self.geo_empty_feat
        app_feat = torch.ones((grid.shape[0], self.app_data.shape[-1]), dtype=torch.float32, device=self.device) * self.app_empty_feat

        # get inside mask
        msk = self.is_inside(grid)

        if return_msk:
            # fetch valid voxels
            geo_feat[msk], app_feat[msk], _geo_msk, _app_msk = self._fetch_links(grid[msk], return_msk=True)
            geo_msk, app_msk = copy.deepcopy(msk), copy.deepcopy(msk)
            geo_msk[msk==True] = _geo_msk
            app_msk[msk==True] = _app_msk

            return geo_feat, app_feat, geo_msk, app_msk
        else:
            geo_feat[msk], app_feat[msk] = self._fetch_links(grid[msk], return_msk=False)
            return geo_feat, app_feat


    def add_noise(self, noise, noise_type="uniform", snap_to_grid=False, update_grid=True):
        if self.verbose:
            print(f'jitter using noise: {noise} and snap to grid: {snap_to_grid}')

        self.geo_data += torch.randn_like(self.geo_data).to(self.device) * noise
        self.app_data += torch.randn_like(self.app_data).to(self.device) * noise

    def jitter(self, noise, noise_type="uniform", snap_to_grid=False, update_grid=True):
        if self.verbose:
            print(f'add noise to feature using noise: {noise} and snap to grid: {snap_to_grid}')
        grid = self.to_svox2(use_current_feat=True)

        sample_pts = self.gen_grid_center_points().to(device=self.device, dtype=torch.float32)
        if noise_type == 'gaussian':
            sample_pts += torch.randn_like(sample_pts) * self.radius * noise
        elif noise_type == 'uniform':
            sample_pts += (2 * torch.rand_like(sample_pts) - 1) * self.radius * noise
        else:
            raise ValueError(f'noise type {noise_type} not supported')

        sample_pts = torch.fmod((sample_pts - self.center), self.radius) + self.center

        # query noise grid and update value 
        grid.to(self.device)
        geo_feat, app_feat = grid(sample_pts)
        self._update_links_and_data(geo_feat.detach(), app_feat.detach())
        del grid


    def upsample(self, reso, radius=None, center=None, update_grid=True):
        grid = self.to_svox2(use_current_feat=True)

        prev_reso, reso = self.reso, self._create_reso(reso)
        radius = self._create_radius(radius) if radius is not None else self.radius * (reso * prev_reso.max()) / (prev_reso * reso.max()) 
        center = self._create_center(center) if center is not None else self.center
        if self.verbose:
            print(f"upsample from size {prev_reso.cpu().numpy()} to {reso.cpu().numpy()} with new radius: {radius.cpu().numpy()}")

        # sampling grid and upsample
        X, Y, Z = torch.meshgrid(
            ((torch.arange(reso[0], dtype=torch.float32, device=self.device) + 0.5) * 2 - reso[0]) / reso[0] * radius[0] + center[0],
            ((torch.arange(reso[1], dtype=torch.float32, device=self.device) + 0.5) * 2 - reso[1]) / reso[1] * radius[1] + center[1],
            ((torch.arange(reso[2], dtype=torch.float32, device=self.device) + 0.5) * 2 - reso[2]) / reso[2] * radius[2] + center[2],
        )
        points = torch.stack((X, Y, Z), dim=-1).view(-1, 3)

        # sampling grid and upsample
        grid.to(self.device)
        geo_feat, app_feat = grid(points)
        self.reso, self.radius, self.center = reso, radius, center
        del grid

        self._update_links_and_data(geo_feat.detach(), app_feat.detach())

        if update_grid:
            self.grid = self.to_svox2(use_current_feat=True)


    def preprocess_feat(self):
        self.geo_feat_type = self.cfg['geo_feat_type'] if 'geo_feat_type' in self.cfg.keys() else None
        self.app_feat_type = self.cfg['app_feat_type'] if 'app_feat_type' in self.cfg.keys() else None
        self.geo_empty_feat, self.app_empty_feat = 0, 0

        if self.grid is None:
            self.grid = svox2.SparseGrid(reso=1, device=self.device)  
            self.geo_data, self.app_data = torch.empty((0, 1), device=self.device), torch.empty((0, 1), device=self.device)
        elif (self.geo_feat_type == None and self.app_feat_type == None):
            self.geo_data, self.app_data = torch.empty((0, 1), device=self.device), torch.empty((0, 1), device=self.device)
        else:
            density_data, sh_data = self.grid(self.gen_grid_center_points().to(self.device))
            density_data, sh_data = density_data.detach(), sh_data.detach()

            geo_msk = density_data.flatten() != 0.0
            geo_data = density_data[geo_msk]

            if self.geo_feat_type == 'raw':
                pass
            elif self.geo_feat_type == 'scaling':
                self.geo_scale = max(geo_data.max(), abs(geo_data.min()))
                geo_data /= self.geo_scale
            elif self.geo_feat_type == 'sdf':
                self.geo_empty_feat = -1
                mesh = trimesh.load(f"{osp.dirname(self.cfg['ckpt'])}/mesh_reso{list_to_str(self.grid._grid_size().long().cpu().numpy())}.obj")
                mesh_sdf = SDF(mesh.vertices, mesh.faces)
                geo_data = torch.tensor(mesh_sdf(self.gen_grid_center_points().cpu().numpy()), dtype=torch.float32, device=self.device).view(-1, 1) # negative outside
                geo_msk = torch.ones_like(geo_msk, dtype=torch.bool, device=self.device)
            elif self.geo_feat_type == 'tsdf':
                assert self.cfg["truncate_scale"] is not None and self.cfg["truncate_scale"] > 0, f"truncate_scale greater than 0 expected, got: {self.cfg['truncate_scale']}"
                self.geo_empty_feat = -1
                if 'mesh_path' in self.cfg.keys():
                    print(f"reload mesh {self.cfg['mesh_path']}")
                    mesh = trimesh.load(self.cfg["mesh_path"])
                else:
                    mesh = trimesh.load(f"{osp.dirname(self.cfg['ckpt'])}/mesh_reso{list_to_str(self.grid._grid_size().long().cpu().numpy())}.obj")
                mesh_sdf = SDF(mesh.vertices, mesh.faces)
                geo_data = torch.tensor(mesh_sdf(self.gen_grid_center_points().cpu().numpy()), dtype=torch.float32, device=self.device).view(-1, 1) # negative outside

                w = self.grid.radius * 2 / self.grid._grid_size()
                assert (w.max() - w.min()) < 1e-10, f"truncate mode only used in cube voxel"
                w = w.max().to(geo_data.device)  # get voxel size
                ws = w * self.cfg["truncate_scale"]
                geo_data = geo_data.clamp_(-ws, ws) / ws

                geo_msk = abs(geo_data.flatten() - self.geo_empty_feat) > self.EPS
                geo_data = geo_data[geo_msk]
            else:
                raise NotImplementedError(f'geometry feature mode: {self.geo_feat_type} is not implemented')
            if self.verbose:
                print((geo_data.shape))
                print('geometry feature ===>', geo_data.shape, geo_data.max(), geo_data.min(), geo_data.mean())
            self.geo_data = geo_data
            # update links
            geo_links = torch.cumsum(geo_msk.to(torch.int32), dim=-1).int() - 1
            geo_links[~geo_msk] = -1
            self.geo_links = geo_links.view(*self.reso).to(self.device)

            app_msk = sh_data.sum(-1).flatten() != 0.00
            app_data = sh_data[app_msk]
            if self.app_feat_type == 'raw':
                pass
            elif self.app_feat_type == 'scaling':
                self.app_scale = max(app_data.max(), abs(app_data.min()))
                app_data /= self.app_scale
            elif self.app_feat_type == 'scaling_by_channel':
                self.app_scale = torch.max(torch.cat([app_data.max(0)[0].unsqueeze(0), abs(app_data.min(0)[0]).unsqueeze(0)], dim=0), dim=0)[0]
                app_data /= self.app_scale
            elif self.app_feat_type == 'pca':
                self.app_empty_feat = 0
                app_data = (app_data - app_data.min(0)[0]) / (app_data.max(0)[0] - app_data.min(0)[0])

                assert self.cfg["n_components"] > 0, f"n_components greater than 0 expected, got: {self.cfg['n_components']}"
                from sklearn.decomposition import PCA
                pca = PCA(n_components=self.cfg["n_components"])
                app_data = torch.tensor(pca.fit_transform(app_data.cpu().numpy()), dtype=torch.float32, device=self.device)
            else:
                raise NotImplementedError(f'appearance feature mode: {self.app_feat_type} is not implemented')
            if self.verbose:    
                print('appearance feature  ===>', app_data.shape, app_data.max(), app_data.min(), app_data.mean(), app_data.max(0)[0], app_data.min(0)[0], app_data.mean(0)[0])
            self.app_data = app_data

            # update links
            app_links = torch.cumsum(app_msk.to(torch.int32), dim=-1).int() - 1
            app_links[~app_msk] = -1
            self.app_links = app_links.view(*self.reso).to(self.device)
            

    def fetch_feat(self, patches_center, patch_size, return_msk=False):
        nei = sample_neighbors(np.linspace(-(patch_size//2), patch_size//2, patch_size)).to(self.device)
        num_patches, num_neighbors = patches_center.shape[0], nei.shape[0]

        patches_center = (patches_center.unsqueeze(1).repeat(1, num_neighbors, 1) + nei).view(-1, 3) # (num_patches x patch_size^3) x 3
        
        if return_msk:
            geo_feat, app_feat, geo_msk, app_msk = self.forward(patches_center.long(), return_msk=True)
            return geo_feat.view(num_patches, num_neighbors, self.geo_data.shape[-1]), app_feat.view(num_patches, num_neighbors, self.app_data.shape[-1]), geo_msk, app_msk
        else:
            geo_feat, app_feat = self.forward(patches_center.long(), return_msk=False)

            return geo_feat.view(num_patches, num_neighbors, self.geo_data.shape[-1]), app_feat.view(num_patches, num_neighbors, self.app_data.shape[-1])


    def extract_patches_center(self, mode="dense", patch_size=None, thickness=None):
        if mode == 'dense':
            patches_center = self.gen_grid_coords()
        elif mode == 'dense_wo_padding':
            assert patch_size and patch_size > 0, f"patch_size greater than 0 expected, got: {patch_size}"
            X = torch.arange(patch_size // 2, self.reso[0] - patch_size // 2, dtype=torch.int32, device=self.device)
            Y = torch.arange(patch_size // 2, self.reso[1] - patch_size // 2, dtype=torch.int32, device=self.device)
            Z = torch.arange(patch_size // 2, self.reso[2] - patch_size // 2, dtype=torch.int32, device=self.device)
            X, Y, Z = torch.meshgrid(X, Y, Z)
            patches_center = torch.stack((X, Y, Z), dim=-1).view(-1, 3)
        elif mode == 'sparse':
            patches_center = (self.geo_links >= 0).nonzero()
        elif mode == 'sparse_w_shell':
            assert thickness and thickness > 0, f"thickness greater than 0 expected, got: {thickness}"
            nei = sample_neighbors(np.linspace(-thickness, thickness, 2 * thickness + 1)).to(self.device)
            patches_center = ((self.geo_links >= 0).nonzero().unsqueeze(1).repeat(1, nei.shape[0], 1) + nei).view(-1, 3).long()
            patches_center = torch.unique(patches_center, dim=0)
        else:
            raise NotImplementedError(f'extract patches mode: {mode} is not implemented')

        self.patches_mode, self.patches_center = mode, patches_center.long()

        return self.patches_center


    def extract_patches(self, mode, patch_size, thickness=None, return_msk=False):
        patches_center = self.extract_patches_center(mode=mode, patch_size=patch_size, thickness=thickness)
        return self.fetch_feat(patches_center, patch_size=patch_size, return_msk=return_msk)


    def combine_patches(self, geo_feat, app_feat, geo_msk=None, app_msk=None, mode="center_voxel"):
        assert self.patches_center.shape[0] == geo_feat.shape[0] == app_feat.shape[0], f"incorrect patches number."
        num_patches = self.patches_center.shape[0]

        if mode == "center_voxel":
            assert geo_feat.shape[1] == app_feat.shape[1] == 1, f"only use one value in center voxel, current is {geo_feat.shape[1]}."
            if geo_msk == None:
                self.geo_links[self.patches_center.unbind(-1)] = torch.arange(geo_feat.shape[0], dtype=torch.int32, device=self.device)
                self.geo_data = geo_feat.squeeze(1).detach()
            else:
                geo_links = torch.cumsum(geo_msk.to(torch.int32), dim=-1).int() - 1
                geo_links[~geo_msk] = -1
                self.geo_links[self.patches_center.unbind(-1)] = geo_links
                self.geo_data = geo_feat[geo_msk].squeeze(1).detach()

            if app_msk == None:
                self.app_links[self.patches_center.unbind(-1)] = torch.arange(app_feat.shape[0], dtype=torch.int32, device=self.device)
                self.app_data = app_feat.squeeze(1).detach()
            else:
                app_links = torch.cumsum(app_msk.to(torch.int32), dim=-1).int() - 1
                app_links[~app_msk] = -1
                self.app_links[self.patches_center.unbind(-1)] = app_links
                self.app_data = app_feat[geo_msk].squeeze(1).detach()

        elif mode == "blending":
            assert geo_feat.shape[1]==app_feat.shape[1] and is_pow3(geo_feat.shape[1]) and is_pow3(app_feat.shape[1]), \
                        f"must be n^3, current is {geo_feat.shape[1]}."
            patch_size = int(np.round(np.power(geo_feat.shape[1], 1/3)))
            padding = 0
            geo_dims, app_dims = geo_feat.shape[-1], app_feat.shape[-1]

            assert self.patches_mode == 'dense_wo_padding', f"blending in combine_patches only support when [patches_mode is dense_wo_padding]."

            def blend_values(feat):
                feat = unfoldNd.foldNd(
                    feat.permute(0, 2, 1).reshape(num_patches, -1).permute(1,0).unsqueeze(0), tuple(self.reso.cpu()), patch_size, dilation=1, padding=padding, stride=1
                )

                input_ones = torch.ones((1, 1, *self.reso), dtype=feat.dtype, device=feat.device)
                divisor = unfoldNd.unfoldNd(input_ones, patch_size, dilation=1, padding=padding, stride=1)
                divisor = unfoldNd.foldNd(divisor, tuple(self.reso.cpu()), patch_size, dilation=1, padding=padding, stride=1)
                divisor[divisor == 0] = 1.0

                return (feat / divisor).view(-1, *self.reso).permute(1, 2, 3, 0)

            geo_feat, app_feat = blend_values(geo_feat).view(-1, geo_dims), blend_values(app_feat).view(-1, app_dims)
            self._update_links_and_data(geo_feat.detach(), app_feat.detach())


    def update_patches(self, geo_feat, app_feat, msk):
        assert msk.sum() == geo_feat.shape[0] == app_feat.shape[0], f"incorrect patches number."
        assert geo_feat.shape[1] == app_feat.shape[1] == 1, f"only use one value in center voxel, current is {geo_feat.shape[1]}."

        self.geo_data[self.geo_links[self.patches_center[msk].unbind(-1)].long()] = geo_feat.squeeze(1).detach()
        self.app_data[self.app_links[self.patches_center[msk].unbind(-1)].long()] = app_feat.squeeze(1).detach()

                
    def to_svox2(self, use_current_feat=True, rescale=False):
        grid = svox2.SparseGrid(reso=tuple(self.reso), radius=self.radius, center=self.center, use_z_order=False, device=self.device)

        if use_current_feat:
            if self.geo_feat_type != "raw" or self.app_feat_type != "raw":
                geo_feat, app_feat = self.extract_patches("dense", 1, return_msk=False)

                if rescale:
                    assert "scaling" in self.geo_feat_type or "01normalization" in sel.geo_feat_type, f"geometry type must be 'scaling' or '01normalization'"
                    if self.geo_feat_type == 'scaling':
                        geo_feat *= self.geo_scale
                    elif self.geo_feat_type == '01normalization':
                        geo_feat = geo_feat * (self.geo_data_max - self.geo_data_min) + self.geo_data_min

                    assert "scaling" in self.app_feat_type or "01normalization" in sel.app_feat_type, f"appearance type must be 'scaling' or '01normalization'"
                    if self.app_feat_type == 'scaling' or self.app_feat_type == 'scaling_by_channel':
                        app_feat *= self.app_scale
                    elif self.app_feat_type == '01normalization' or self.app_feat_type == '01normalization_by_channel':
                        app_feat = app_feat *  (self.app_data_max - self.app_data_min) + self.app_feat_min
                else:
                    if self.verbose:
                        print("geometry & appearance data types are not 'raw', which will result in wrong rendering.")
                grid.density_data = torch.nn.Parameter(geo_feat.squeeze(1).contiguous()).to(self.device)
                grid.sh_data = torch.nn.Parameter(app_feat.squeeze(1).contiguous()).to(self.device)
            else:
                density_data, sh_data, density_msk, sh_msk = self.extract_patches("dense", 1, 0, 0, return_msk=True)
                assert (density_msk != sh_msk).sum() == 0, "incorrect data."
                links = torch.cumsum(density_msk.to(torch.int32), dim=-1).int() - 1
                links[~density_msk] = -1
                grid.links = links.view(*self.reso)
                grid.density_data = torch.nn.Parameter(density_data[density_msk].squeeze(1).contiguous()).to(self.device)
                grid.sh_data = torch.nn.Parameter(sh_data[sh_msk].squeeze(1).contiguous()).to(self.device)

        else: # maily for crop / move the scene
            sample_pts = self.gen_grid_center_points().to(device=self.device, dtype=torch.float32) - self.center + self.grid.center.to(self.device)
            self.grid.to(self.device)
            density_data, sh_data = self.grid(sample_pts)
            self.grid.to('cpu')
            msk = density_data.flatten() != 0.0
            links = torch.cumsum(msk.to(torch.int32), dim=-1).int() - 1
            links[~msk] = -1
            grid.links = links.view(*self.reso)
            grid.density_data = torch.nn.Parameter(density_data[msk].contiguous()).to(self.device)
            grid.sh_data = torch.nn.Parameter(sh_data[msk].contiguous()).to(self.device)
        
        return grid


    def extract_mesh(self, save_name, N_grid=256, sigma_thres=0.0, smooth_sigma=1, floodfill=True):
        radius = self.radius.cpu().numpy()
        x = np.linspace(-radius[0], radius[0], N_grid)
        y = np.linspace(-radius[1], radius[1], N_grid)
        z = np.linspace(-radius[2], radius[2], N_grid)
        xyz = torch.from_numpy(np.stack(np.meshgrid(x, y, z), -1)).float().to(self.device).view(-1, 3) - self.center + self.grid.center.to(self.device)
        self.grid.to(self.device)
        sigma = batchify_query_np(self.grid, xyz, 1024000)[0].reshape(N_grid, N_grid, N_grid)
        self.grid.to('cpu')

        if floodfill: # perform floodfill 
            visit_flag = 1
            def is_valid(v):
                return v > 0 and v != visit_flag
            for i in tqdm(range(sigma.shape[0])):
                valid_ind = np.array((sigma[i] > 0).nonzero()).transpose((1, 0)).tolist()
                while valid_ind:
                    x, y = valid_ind.pop(0)
                    if is_valid(sigma[i, x, y]):
                        sigma[i, x, y] = visit_flag
                        if x - 1 >= 0 and is_valid(sigma[i, x-1, y]):
                            valid_ind.append([x-1, y])
                        if x + 1 < N_grid and is_valid(sigma[i, x+1, y]):
                            valid_ind.append([x+1, y])
                        if y - 1 >= 0 and is_valid(sigma[i, x, y-1]):
                            valid_ind.append([x, y-1])
                        if y + 1 < N_grid and is_valid(sigma[i, x, y+1]):
                            valid_ind.append([x, y+1])

        sigma = np.clip(sigma, 0.0, 1.0)
        pad_size = 4 
        sigma = np.pad(sigma, pad_size)
        if self.verbose:
            print(f'Extracting mesh and save to {save_name}.')
        os.makedirs(osp.dirname(save_name), exist_ok=True)
        if smooth_sigma is not None:
            sigma = mcubes.smooth(sigma, method='gaussian', sigma=smooth_sigma)
        vertices, triangles = mcubes.marching_cubes(sigma, sigma_thres)
        vertices[:,[0, 1]] = vertices[:,[1, 0]]
        triangles[:, [0, 1, 2]] = triangles[:,[0, 2, 1]]
        vertices = (2 * vertices - (N_grid + pad_size)) / N_grid * self.grid.radius.cpu().numpy() + self.grid.center.cpu().numpy()
        mcubes.export_obj(vertices, triangles, save_name)


    def visualization(self, 
                      save_path, 
                      trajectory_path, 
                      depth_cmap: str = "rainbow",
                      fps: int = 8):
        PlenoxelsWrapper._visualization(self.grid, save_path, trajectory_path, depth_cmap, fps)


    @staticmethod
    def _visualization(grid, save_path, trajectory_path, depth_cmap: str = "rainbow", fps: int = 8):
        os.makedirs(os.path.join(save_path, 'rgb'), exist_ok=True)
        os.makedirs(os.path.join(save_path, 'depth'), exist_ok=True)

        grid.to("cuda:0")
        rgb_imgs = []
        trajectory = json.load(open(trajectory_path))
        for i, cam in enumerate(trajectory["parameters"]):
            c2w = np.linalg.inv(np.array(cam["extrinsic"]).reshape(4, 4).transpose())
            fx, fy = cam["intrinsic"]["intrinsic_matrix"][0], cam["intrinsic"]["intrinsic_matrix"][4]
            w, h = cam["intrinsic"]["width"], cam["intrinsic"]["height"]

            cam = svox2.Camera(torch.from_numpy(c2w).to(grid.density_data.device).float(),
                                fx,
                                fy,
                                w * 0.5,
                                h * 0.5,
                                w, h,
                                ndc_coeffs=(-1.0, -1.0))
            torch.cuda.synchronize()
            image = grid.volume_render_image(cam, use_kernel=True)
            depth = grid.volume_render_depth_image(cam)
            torch.cuda.synchronize()

            image = image.clamp_(0.0, 1.0).detach().cpu().numpy()
            image = (image * 255).astype(np.uint8)
            imageio.imwrite(f'{save_path}/rgb/{i:0>5d}.png', image)
            rgb_imgs += [image]
            
            depth = depth.detach().cpu().numpy()
            depth = (plt.get_cmap(depth_cmap)(depth / depth.max())[..., :3] * 255).astype(np.uint8)
            imageio.imwrite(f'{save_path}/depth/{i:0>5d}.png', depth)
        grid.to("cpu")

        imageio.mimwrite(f'{save_path}/rgb_fps{fps}.mp4', rgb_imgs, fps=fps, macro_block_size=8)


