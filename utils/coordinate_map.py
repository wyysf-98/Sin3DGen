import os
import os.path as osp
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import copy
import imageio
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from functools import reduce
from typing import Union, List, NamedTuple, Optional, Tuple
import torch
from torch import nn

import svox2
from utils.base import batchify_query
from utils.sparse_grid import SparseGrid
from utils.plenoxels_wrapper import PlenoxelsWrapper


class CoordinateMap(SparseGrid):
    def __init__(
        self,
        reso: Union[int, List[int], Tuple[int, int, int]] = 16,
        radius: Union[float, List[float]] = 1.0,
        center: Union[float, List[float]] = [0.0, 0.0, 0.0],
        device: Union[torch.device, str] = "cpu",
    ):
        super(CoordinateMap, self).__init__(reso, radius, center, device=device)
        self.device = device

        self.pos_data = self.gen_grid_center_points().to(device=device, dtype=torch.float32)

    def __repr__(self):
        return (
            f"CoordinateMap, "
            + f"reso={self.reso.detach().cpu().numpy().tolist()}, "
            + f"radius={self.radius.detach().cpu().numpy().tolist()}, "
            + f"center={self.center.detach().cpu().numpy().tolist()}, "
            + f"capacity:{self.capacity}), "
            + f"pos_data:{self.pos_data.shape})"
        )

    def _fetch_links(self, grid):
        assert (~self.is_inside(grid)).sum() == 0
        pos = torch.zeros(
            (grid.size(0), self.pos_data.size(1)),
            device=grid.device,
            dtype=torch.float32,
        )

        links = self.links[grid.unbind(-1)]
        msk = links >= 0
        inds = links[msk].long()
        pos[msk] = self.pos_data[inds]

        return pos, msk


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
            "links":self.links.detach().cpu().numpy(),
            "pos_data":self.pos_data.detach().cpu().numpy(),
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

        for attr in ["reso", "radius", "center", "capacity", "links", "pos_data"]:
            assert attr in z.files 

        grid = cls(
            reso=z.f.reso,
            radius=z.f.radius.tolist(),
            center=z.f.center.tolist(),
            device=device,
        )
        grid.capacity = torch.from_numpy(z.f.capacity).to(device=device)
        grid.links = torch.from_numpy(z.f.links).to(device=device)
        grid.pos_data = torch.from_numpy(z.f.pos_data).to(device=device)

        if grid.links.is_cuda:
            grid._accelerate(grid.links)
        return grid


    def forward(self, points):
        '''
        points: [num_points, 3]
        '''
        grid = self.world2grid(points)

        # get inside mask
        msk = self.is_inside(grid)

        # fetch valid voxels
        _grid = torch.round(grid[msk]).to(torch.long)
        pos, _msk = self._fetch_links(_grid)
        msk[msk==True] = _msk
        
        if msk.sum() > 0:
            # compute the offset delta
            points[msk] = torch.mul(grid[msk] - _grid[_msk], 2 * self.radius / self.reso) + pos[_msk]

        return points, msk


    def jitter(self, noise, noise_type="uniform", snap_to_grid=False, bounds=None):
        print(f'jitter using noise: {noise} and snap to grid: {snap_to_grid}')

        radius = self.radius if bounds is None else bounds
        if noise_type == 'gaussian':
            pos_data = self.pos_data + torch.randn_like(self.pos_data) * radius  * noise
        elif noise_type == 'uniform':
            pos_data = self.pos_data + (2 * torch.rand_like(self.pos_data) - 1) * radius * noise
        else:
            raise ValueError(f'noise type {noise_type} not supported')
        
        pos_data = torch.fmod((pos_data - self.center), radius) + self.center
        if snap_to_grid:
            pos_data = self.grid2world(torch.round(self.world2grid(pos_data)))

        self.pos_data = pos_data


    def upsample(self, reso, radius=None, center=None):
        prev_reso, reso = self.reso, self._create_reso(reso)
        radius = self._create_radius(radius) if radius is not None else self.radius * (reso * prev_reso.max()) / (prev_reso * reso.max()) 
        center = self._create_center(center) if center is not None else self.center
        print(f"upsample from size {prev_reso.cpu().numpy()} to {reso.cpu().numpy()} with new radius: {radius.cpu().numpy()}")

        # sampling grid and upsample
        X, Y, Z = torch.meshgrid(
            ((torch.arange(reso[0], dtype=torch.float32, device=self.device) + 0.5) * 2 - reso[0]) / reso[0] * radius[0] + center[0],
            ((torch.arange(reso[1], dtype=torch.float32, device=self.device) + 0.5) * 2 - reso[1]) / reso[1] * radius[1] + center[1],
            ((torch.arange(reso[2], dtype=torch.float32, device=self.device) + 0.5) * 2 - reso[2]) / reso[2] * radius[2] + center[2],
        )
        points = torch.stack((X, Y, Z), dim=-1).view(-1, 3)
        points, msk = self.forward(points)

        init_links = torch.cumsum(msk.to(torch.int32), dim=-1).int() - 1
        init_links[~msk] = -1
        self.links = init_links.view(*reso)

        self.pos_data = points[msk]

        self.reso, self.radius, self.center = reso, radius, center


    def update(self, patches_center, pos_data):
        self.links = -1 * torch.ones_like(self.links).to(self.device)
        self.links[patches_center.unbind(-1)] = torch.arange(pos_data.shape[0], dtype=torch.int32, device=self.device)

        self.pos_data = pos_data.to(self.device)


    def create_scene(self, exemplar, use_grid_data=False, upsample_ratio=None):
        reso = self.reso * upsample_ratio if upsample_ratio is not None else self.reso * self.get_voxel_size() / exemplar.get_voxel_size()
        reso = torch.ceil(reso).to(torch.int32)

        X, Y, Z = torch.meshgrid(
            ((torch.arange(reso[0], dtype=torch.float32, device=self.device) + 0.5) * 2 - reso[0]) / reso[0] * self.radius[0] + self.center[0],
            ((torch.arange(reso[1], dtype=torch.float32, device=self.device) + 0.5) * 2 - reso[1]) / reso[1] * self.radius[1] + self.center[1],
            ((torch.arange(reso[2], dtype=torch.float32, device=self.device) + 0.5) * 2 - reso[2]) / reso[2] * self.radius[2] + self.center[2],
        )
        points = torch.stack((X, Y, Z), dim=-1).view(-1, 3)

        scene = PlenoxelsWrapper({}, device=self.device)
        scene.capacity = reduce(lambda x, y: x * y, reso)
        scene.reso, scene.radius, scene.center = reso, self.radius, self.center
        scene.geo_feat_type, scene.app_feat_type = exemplar.geo_feat_type, exemplar.app_feat_type
        scene.geo_empty_feat, scene.app_empty_feat = exemplar.geo_empty_feat, exemplar.app_empty_feat

        # query points
        points, msk = self.forward(points)
        # print((~msk).sum())

        # fetch color
        if use_grid_data: # mainly for visualizing synthesis results
            exemplar.grid.to(self.device)
            def query_fn(points):
                density_data, sh_data = exemplar.grid(points)
                _msk = density_data.flatten() != 0.0
                return _msk, density_data[_msk], sh_data[_msk]
            _msk, density_data, sh_data = batchify_query(query_fn, points[msk], chunk_size=1024000)
            exemplar.grid.to('cpu')

            msk[msk==True] = _msk
            links = torch.cumsum(msk.to(torch.int32), dim=-1).int() - 1
            links[~msk] = -1
            scene.grid.links = links.view(*reso)
            scene.grid.density_data = torch.nn.Parameter(density_data.contiguous()).to(self.device)
            scene.grid.sh_data = torch.nn.Parameter(sh_data.contiguous()).to(self.device)
            scene.grid.radius = self.radius.cpu()
            scene.grid.center = self.center.cpu()
            scene.grid._scaling = 0.5 / scene.grid.radius
            scene.grid._offset = 0.5 * (1.0 - scene.grid.center / scene.grid.radius)

        else: # mainly for intermediate variables
            grid = exemplar.to_svox2(use_current_feat=True, rescale=False)
            geo_feat, app_feat = grid(points[msk])
            geo_feat, app_feat = geo_feat.detach(), app_feat.detach()
            del grid
            # print(geo_feat.min(), geo_feat.max(), app_feat.min(), app_feat.max())
            scene.grid = exemplar.grid

            geo_msk = copy.deepcopy(msk)
            _geo_msk = abs(geo_feat.flatten() - scene.geo_empty_feat) > self.EPS
            geo_msk[msk==True] = _geo_msk
            geo_links = torch.cumsum(geo_msk.to(torch.int32), dim=-1).int() - 1
            geo_links[~geo_msk] = -1
            scene.geo_links = geo_links.view(*reso)
            scene.geo_data = geo_feat[_geo_msk]

            app_msk = copy.deepcopy(msk)
            _app_msk = abs(app_feat.sum(-1).flatten() - scene.app_empty_feat) > self.EPS
            app_msk[msk==True] = _app_msk
            app_links = torch.cumsum(app_msk.to(torch.int32), dim=-1).int() - 1
            app_links[~app_msk] = -1
            scene.app_links = app_links.view(*reso)
            scene.app_data = app_feat[_app_msk]

        return scene


    def visualization(self, 
                      save_path, 
                      trajectory_path, 
                      mesh_path: str = None, 
                      sdf_thres: float = 0.0,
                      depth_cmap: str = "rainbow",
                      fps: int = 8):

        def draw_geometry(vis_list, trajectory, save_path, render_option_path=None):
            draw_geometry.index = -1
            draw_geometry.trajectory = o3d.io.read_pinhole_camera_trajectory(trajectory)
            draw_geometry.vis = o3d.visualization.Visualizer()

            os.makedirs(os.path.join(save_path, 'rgb'), exist_ok=True)
            os.makedirs(os.path.join(save_path, 'depth'), exist_ok=True)

            def move_forward(vis):
                ctr = vis.get_view_control()
                glb = draw_geometry
                if glb.index >= 0:
                    image = (np.asarray(vis.capture_screen_float_buffer(False)) * 255).astype(np.uint8)
                    imageio.imwrite(f'{save_path}/rgb/{glb.index:0>5d}.png', image)
                    depth = np.asarray(vis.capture_depth_float_buffer(False))
                    depth = (plt.get_cmap(depth_cmap)(depth / depth.max())[..., :3] * 255).astype(np.uint8)
                    imageio.imwrite(f'{save_path}/depth/{glb.index:0>5d}.png', depth)

                glb.index = glb.index + 1
                if glb.index < len(glb.trajectory.parameters):
                    ctr.convert_from_pinhole_camera_parameters(
                        glb.trajectory.parameters[glb.index])
                else:
                    draw_geometry.vis.\
                            register_animation_callback(None)
                    draw_geometry.vis.close()
                return False

            vis = draw_geometry.vis
            vis.create_window(width=draw_geometry.trajectory.parameters[0].intrinsic.width, \
                              height=draw_geometry.trajectory.parameters[0].intrinsic.height)
            for geo in vis_list:
                vis.add_geometry(geo)
            if render_option_path is not None:
                vis.get_render_option().load_from_json(render_option_path)
            vis.register_animation_callback(move_forward)
            vis.run()
            vis.destroy_window()

            rgb_imgs = []
            for i in range(len(draw_geometry.trajectory.parameters)):
                rgb_imgs += [imageio.imread(f'{save_path}/rgb/{i:0>5d}.png')]
            imageio.mimwrite(f'{save_path}/rgb_fps{fps}.mp4', rgb_imgs, fps=fps, macro_block_size=8)

        if mesh_path is not None:
            import trimesh
            from pysdf import SDF
            mesh = trimesh.load(mesh_path)
            mesh_sdf = SDF(mesh.vertices, mesh.faces)

        vis_list = []
        grid_coords = self.gen_grid_coords()[self.links.flatten() >= 0] 
        points = self.grid2world(grid_coords) 
        e_points, _ = self.forward(copy.deepcopy(points))

        colors = (e_points - e_points.min(0)[0]) / (e_points.max(0)[0] - e_points.min(0)[0])
        points = points.detach().cpu().numpy()
        colors = colors.detach().cpu().numpy()
        radius = self.radius.detach().cpu().numpy()
        reso = self.reso.detach().cpu().numpy()

        if mesh_path is not None:
            sdf = mesh_sdf(points)
            points = points[sdf > sdf_thres]
            colors = colors[sdf > sdf_thres]
            del mesh, mesh_sdf, sdf

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        voxel_size = radius / reso
        if abs(voxel_size[0] - voxel_size[1]) < self.EPS and abs(voxel_size[1] - voxel_size[2]) < self.EPS:
            voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd,voxel_size=2*voxel_size[0])
            vis_list += [voxel_grid]
        else:
            print('voxel is not a cube, only vis point cloud.')
            vis_list += [pcd]

        draw_geometry(vis_list, trajectory_path, save_path)

