import json
import torch
import imageio
import numpy as np
from typing import Optional
import matplotlib.pylab as plt
from plyfile import PlyData, PlyElement

### ==== visualize 3D grid
# ply utils
def write_ply(points, filename, text=False):
    """
    input: Nx3, write points to filename as PLY format.
    """
    points = [(points[i, 0], points[i, 1], points[i, 2]) for i in range(points.shape[0])]
    vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    with open(filename, mode='wb') as f:
        PlyData([el], text=text).write(f)

def write_ply_rgb(points, filename, text=False):
    """ input: Nx6, write points with rgb color to filename as PLY format. """
    points = [(points[i, 0], points[i, 1], points[i, 2], points[i, 3], points[i, 4],
               points[i, 5]) for i in range(points.shape[0])]
    vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red','u1'), ('green', 'u1'), ('blue', 'u1')])
    el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    with open(filename, mode='wb') as f:
        PlyData([el], text=text).write(f)

def write_ply_normal(points, filename, text=False):
    """ input: Nx6, write points with normal to filename as PLY format. """
    points = [(points[i, 0], points[i, 1], points[i, 2], points[i, 3], points[i, 4],
               points[i, 5]) for i in range(points.shape[0])]
    vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('nx','f4'), ('ny', 'f4'), ('nz', 'f4')])
    el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    with open(filename, mode='wb') as f:
        PlyData([el], text=text).write(f)

def plot_grid_slices(grid, save_path):
    assert grid.shape[-1] == 1 or grid.shape[-1] == 3
    if isinstance(grid, torch.Tensor):
        grid = grid.cpu().numpy()
    imgs = []
    grid = (grid - grid.min(0)[0]) / (grid.max(0)[0] - grid.min(0)[0])
    for i in range(grid.shape[1]):
        if grid.shape[-1] == 1:
            s = (plt.get_cmap("rainbow")(grid[:, i, :, 0])[..., :3] * 255).astype(np.uint8)
        else:
            s = (grid[:, i, :] * 255).astype(np.uint8)
        # print(s.shape)
        imgs += [np.flip(np.rot90(s), 1)]
    # for i in range(grid.shape[2]):
    #     if grid.shape[-1] == 1:
    #         s = (plt.get_cmap("rainbow")(grid[:, :, i, 0])[..., :3] * 255).astype(np.uint8)
    #     else:
    #         s = (grid[:, :, i] * 255).astype(np.uint8)
    #     print(s.shape)
    #     imgs += [np.flip(np.rot90(s), 1)]
    imageio.mimwrite(save_path, imgs)
    # imageio.mimwrite(save_path, imgs, macro_block_size = None)

def plot_sample(x, y, z, **kwargs):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(projection='3d')
    
    ax.scatter(x, y, z, **kwargs)
    ax.set_box_aspect((1, 1, 1.05))
    ax.set_xlim3d([-max(abs(x.min()), x.max()), max(abs(x.min()), x.max())])
    ax.set_ylim3d([-max(abs(y.min()), y.max()), max(abs(y.min()), y.max())])
    ax.set_zlim3d([-max(abs(z.min()), z.max()), max(abs(z.min()), z.max())])
    plt.savefig("sample.png")
    plt.close()

### ==== generate trajectory
def dump_trajectory(reso, focal, c2ws, trajectory_path):
    trajectory = {
        "version_major" : 1,
        "version_minor" : 0,
        "class_name" : "PinholeCameraTrajectory",
        "parameters" : [{
            "version_major" : 1,
            "version_minor" : 0,
            "class_name" : "PinholeCameraParameters",
            "intrinsic" : {
                "height" : reso[0],
                "width" : reso[1],
                "intrinsic_matrix" : [
                    focal[0], 0, 0, 
                    0, focal[1], 0,
                    reso[0]/2, reso[1]/2, 1
                ]
            },
            "extrinsic" : np.linalg.inv(ext.transpose()).flatten().tolist()
            } for ext in c2ws
        ]
    }
    json.dump(trajectory, open(trajectory_path, 'w'), indent=4)

def sample_on_sphere(num_samples, dist=1.0, min_elevation=-90, obj_pos=[0, 0, 0], track='Z', up_axis='Y'):
    """ sample camera pose from the sphere with radius of dist, and elevation between min_elevation and +90
    reference: https://www.zhihu.com/question/527922446
    """
    # num_samples = 100
    ratio = (1 - np.sin(np.deg2rad(min_elevation))) / 2
    phi = (np.sqrt(5) - 1.0) / 2.
    ret_pos = []
    for n in range(1, int(num_samples/ratio) + 1):
        z = (2. * n - 1) / int(num_samples/ratio) - 1.
        x = np.cos(2*np.pi*n*phi)*np.sqrt(1-z*z)
        y = np.sin(2*np.pi*n*phi)*np.sqrt(1-z*z)
        ret_pos.append((x, y, z))

    ret_pos = np.array(ret_pos)[-num_samples:] * dist
    # plot_sample(ret_pos[:, 0], ret_pos[:, 1], ret_pos[:, 2])
    # exit()

    c2ws = []
    for pos in ret_pos:
        direction = pos - np.array(obj_pos)
        c2w = np.array(Vector(direction).to_track_quat(track, up_axis).to_matrix().to_4x4())
        c2w[:3, 3] = pos
        # OpenGL -> OpenCV
        c2w = c2w @ np.diag(np.array([1, -1, -1, 1], dtype=np.float32))
        c2ws.append(c2w)
    c2ws = np.array(c2ws)

    return c2ws

def pose_spherical(theta : float, phi : float, radius : float, offset : Optional[np.ndarray]=None,
                   vec_up : Optional[np.ndarray]=None):
    """
    Generate spherical rendering poses, from NeRF. Forgive the code horror
    :return: r (3,), t (3,)
    """

    # Rather ugly pose generation code, derived from NeRF
    def _trans_t(t):
        return np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, t],
                [0, 0, 0, 1],
            ],
            dtype=np.float32,
        )

    def _rot_phi(phi):
        return np.array(
            [
                [1, 0, 0, 0],
                [0, np.cos(phi), -np.sin(phi), 0],
                [0, np.sin(phi), np.cos(phi), 0],
                [0, 0, 0, 1],
            ],
            dtype=np.float32,
        )

    def _rot_theta(th):
        return np.array(
            [
                [np.cos(th), 0, -np.sin(th), 0],
                [0, 1, 0, 0],
                [np.sin(th), 0, np.cos(th), 0],
                [0, 0, 0, 1],
            ],
            dtype=np.float32,
        )

    c2w = _trans_t(radius)
    c2w = _rot_phi(phi / 180.0 * np.pi) @ c2w
    c2w = _rot_theta(theta / 180.0 * np.pi) @ c2w
    c2w = (
        np.array(
            [[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
            dtype=np.float32,
        )
        @ c2w
    )
    if vec_up is not None:
        vec_up = vec_up / np.linalg.norm(vec_up)
        vec_1 = np.array([vec_up[0], -vec_up[2], vec_up[1]])
        vec_2 = np.cross(vec_up, vec_1)

        trans = np.eye(4, 4, dtype=np.float32)
        trans[:3, 0] = vec_1
        trans[:3, 1] = vec_2
        trans[:3, 2] = vec_up
        c2w = trans @ c2w
    # OpenGL -> OpenCV
    c2w = c2w @ np.diag(np.array([1, -1, -1, 1], dtype=np.float32))
    if offset is not None:
        c2w[:3, 3] += offset
    return c2w

def gen_fly_trajectory(n=60, focal=[500, 500], size=[512, 512], elevation=-45, radius=3, center=[0, 0, 0], vec_up=None):    
    return {
        "version_major" : 1,
        "version_minor" : 0,
        "class_name" : "PinholeCameraTrajectory",
        "parameters" : [{
			"version_major" : 1,
			"version_minor" : 0,
            "class_name" : "PinholeCameraParameters",
            "intrinsic" : {
                "width" : size[0],
                "height" : size[1],
                "intrinsic_matrix" : [
                    focal[0], 0, 0, 
                    0, focal[1], 0,
                    size[0] / 2-0.5, size[1] / 2-0.5, 1
                ]
            },
            "extrinsic" : np.linalg.inv(pose_spherical(
                            angle,
                            elevation,
                            radius,
                            center,
                            vec_up=vec_up,
                        )).transpose().flatten().tolist()
            } for angle, elevation, radius in zip(np.linspace(-180, 180, n + 1)[:-1], np.linspace(-60, -5, n + 1)[:-1], np.linspace(radius, 0.5, n + 1))
        ]
    }

def gen_circle_trajectory(n=10, focal=[500, 500], size=[512, 512], elevation=-45, radius=3, center=[0, 0, 0], vec_up=None):
    return {
        "version_major" : 1,
        "version_minor" : 0,
        "class_name" : "PinholeCameraTrajectory",
        "parameters" : [{
			"version_major" : 1,
			"version_minor" : 0,
            "class_name" : "PinholeCameraParameters",
            "intrinsic" : {
                "width" : size[0],
                "height" : size[1],
                "intrinsic_matrix" : [
                    focal[0], 0, 0, 
                    0, focal[1], 0,
                    size[0] / 2-0.5, size[1] / 2-0.5, 1
                ]
            },
            "extrinsic" : np.linalg.inv(pose_spherical(
                            angle,
                            elevation,
                            radius,
                            center,
                            vec_up=vec_up,
                        )).transpose().flatten().tolist()
            } for angle in np.linspace(-180, 180, n + 1)[:-1]
        ]
    }
