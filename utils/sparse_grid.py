import os
import os.path as osp
import copy
import numpy as np
from functools import reduce
from typing import Union, List, NamedTuple, Optional, Tuple
import torch
from torch import nn

import svox2 

class SparseGrid(nn.Module):
    def __init__(
        self,
        reso: Union[int, List[int], Tuple[int, int, int]] = 1,
        radius: Union[float, List[float]] = 1.0,
        center: Union[float, List[float]] = [0.0, 0.0, 0.0],
        attributes: Union[str, List[str]] = "links",
        device: Union[torch.device, str] = "cpu",
    ):
        super().__init__()
        self.device = device

        self.reso = self._create_reso(reso)
        self.radius = self._create_radius(radius)
        self.center = self._create_center(center)
        self.capacity = reduce(lambda x, y: x * y, self.reso)

        self._create_links(attributes)

        self.EPS = 1e-6

    def _create_reso(self, reso):
        if isinstance(reso, int):
            reso = [reso] * 3
        else:
            assert (
                len(reso) == 3
            ), "reso must be an integer or indexable object of 3 ints"
        if isinstance(reso, torch.Tensor):
            reso = reso.to(dtype=torch.int32, device=self.device)
        else:
            reso = torch.tensor(reso, dtype=torch.int32, device=self.device)
        return reso

    def _create_radius(self, radius):
        if isinstance(radius, float) or isinstance(radius, int):
            radius = [radius] * 3
        if isinstance(radius, torch.Tensor):
            radius = radius.to(dtype=torch.float32, device=self.device)
        else:
            radius = torch.tensor(radius, dtype=torch.float32, device=self.device)
        return radius

    def _create_center(self, center):
        if isinstance(center, torch.Tensor):
            center = center.to(dtype=torch.float32, device=self.device)
        else:
            center = torch.tensor(center, dtype=torch.float32, device=self.device)
        return center

    def _create_links(self, attributes):
        if isinstance(attributes, str):
            attributes = [attributes]

        for attr in attributes:
            init_links = torch.arange(self.capacity, device=self.device, dtype=torch.int32)
            self.register_buffer(attr, init_links.view(*self.reso))
            if getattr(self, attr).is_cuda:
                self._accelerate(getattr(self, attr))

    def _accelerate(self, links):
        """
        Accelerate
        """
        _C = svox2.utils._get_c_extension()
        assert (
            _C is not None and links.is_cuda
        ), "CUDA extension is currently required for accelerate"
        _C.accel_dist_prop(links)


    def __repr__(self):
        return (
            f"SparseGrid(based on svox2.SparseGrid), "
            + f"reso={self.reso.detach().cpu().numpy().tolist()}, "
            + f"radius={self.radius.detach().cpu().numpy().tolist()}, "
            + f"center={self.center.detach().cpu().numpy().tolist()}, "
            + f"capacity:{self.capacity})"
        )

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

        for attr in ["reso", "radius", "center", "capacity", "links"]:
            assert attr in z.files 

        grid = cls(
            reso=z.f.reso,
            radius=z.f.radius.tolist(),
            center=z.f.center.tolist(),
            device=device,
        )
        grid.capacity = z.f.capacity
        grid.links = torch.from_numpy(z.f.links).to(device=device)

        if grid.links.is_cuda:
            grid._accelerate()
        return grid

    def is_inside(self, grid):
        if grid.dtype is not torch.long:
            grid = torch.round(grid)
        return ((grid>=0) & torch.le(grid, self.reso - 1)).all(dim=-1)

    def world2grid(self, points):
        """
        World coordinates to grid coordinates.
        :param points: (N, 3)
        :return: (N, 3)
        """
        return ((points - self.center) * self.reso / self.radius + self.reso) / 2 - 0.5

    def grid2world(self, grid_coords):
        """
        Grid coordinates to world coordinates.
        :param grid: (N, 3)
        :return: (N, 3)
        """
        return (2 * (grid_coords + 0.5) - self.reso) * self.radius / self.reso + self.center

    def grid2id(self, grid_coords):
        """
        Grid coordinates to ID.
        :param grid: (N, 3)
        :return: (N)
        """
        # print(grid_coords)
        assert grid_coords.max() < 2e10
        return grid_coords[:, 2] << 20 | grid_coords[:, 1] << 10 | grid_coords[:, 0]
        # return grid_coords[:, 0] * self.reso[0] * self.reso[1] + grid_coords[:, 1] * self.reso[1] + grid_coords[:, 2]

    def id2grid(self, ids):
        """
        ID to Grid coordinates.
        :param id: (N)
        :return: (N, 3)
        """
        grid_coords = torch.empty((ids.shape[0], 3), dtype=torch.int64, device=self.device)
        grid_coords[:, 0] = (ids)&((1 << 10) - 1)
        grid_coords[:, 1] = (ids >> 10)&((1 << 10) - 1)
        grid_coords[:, 2] = (ids >> 20)&((1 << 20) - 1)
        return grid_coords
        # return (2 * (grid_coords + 0.5) - self.reso) * self.radius / self.reso + self.center


    def get_voxel_size(self):
        return 2 * self.radius / self.reso

    def gen_grid_coords(self):
        X, Y, Z = torch.meshgrid(
            torch.arange(self.reso[0], dtype=torch.int64, device=self.device),
            torch.arange(self.reso[1], dtype=torch.int64, device=self.device),
            torch.arange(self.reso[2], dtype=torch.int64, device=self.device),
        )
        grid_coords = torch.stack((X, Y, Z), dim=-1).view(-1, 3)

        return grid_coords

    def gen_grid_center_points(self):
        X, Y, Z = torch.meshgrid(
            ((torch.arange(self.reso[0], dtype=torch.float32, device=self.device) + 0.5) * 2 - self.reso[0]) / self.reso[0] * self.radius[0] + self.center[0],
            ((torch.arange(self.reso[1], dtype=torch.float32, device=self.device) + 0.5) * 2 - self.reso[1]) / self.reso[1] * self.radius[1] + self.center[1],
            ((torch.arange(self.reso[2], dtype=torch.float32, device=self.device) + 0.5) * 2 - self.reso[2]) / self.reso[2] * self.radius[2] + self.center[2],
        )
        points = torch.stack((X, Y, Z), dim=-1).view(-1, 3)

        return points


if __name__ == "__main__":
    device = torch.device("cuda:0")
    sgrid = SparseGrid(reso=[4, 4, 2], center=[0, 0, 0], radius=[1, 1, 0.25], device=device)
    print(sgrid, sgrid.reso)

    sgrid_coords = sgrid.gen_grid_coords()
    sgrid_center_points = sgrid.gen_grid_center_points()
    print(sgrid_coords, sgrid_center_points)

    print((sgrid_coords != sgrid.world2grid(sgrid_center_points)).sum())
    print((sgrid_center_points != sgrid.grid2world(sgrid_coords)).sum())

