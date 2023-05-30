import os
import os.path as osp
import torch
from torch.autograd import Variable
from torch.autograd import Function
import torch.nn as nn

from torch.utils.cpp_extension import load
import warnings
warnings.simplefilter("ignore") 
sampling_cuda = load(name="sampling_cuda",
                    sources=[
                            f"{osp.dirname(os.path.abspath(__file__))}/sampling_gpu.cu",
                            f"{osp.dirname(os.path.abspath(__file__))}/sampling.cpp",
                            f"{osp.dirname(os.path.abspath(__file__))}/sampling_api.cpp",
                            ])

class FurthestPointSampling(Function):
    @staticmethod
    def forward(ctx, xyz: torch.Tensor, npoint: int) -> torch.Tensor:
        """
        Uses iterative furthest point sampling to select a set of npoint features that have the largest
        minimum distance
        :param ctx:
        :param xyz: (B, N, 3) where N > npoint
        :param npoint: int, number of features in the sampled set
        :return:
             output: (B, npoint) tensor containing the set
        """
        assert xyz.is_contiguous()

        B, N, _ = xyz.size()
        output = torch.cuda.IntTensor(B, npoint)
        temp = torch.cuda.FloatTensor(B, N).fill_(1e10)

        sampling_cuda.furthest_point_sampling_wrapper(B, N, npoint, xyz, temp, output)
        return output

    @staticmethod
    def backward(xyz, a=None):
        return None, None


def furthest_point_sample(pts, num_pts=2048, return_idx=False):
    bs,_N,_C = pts.shape
    device = pts.device
    shape_pcs = pts.view(bs, -1, 3)
    with torch.no_grad():
        shape_pc_id1 = torch.arange(bs).unsqueeze(1).repeat(1, num_pts).long().view(-1).to(device)
        shape_pc_id2 = FurthestPointSampling.apply(shape_pcs, num_pts).long().view(-1)
    shape_pcs = shape_pcs[shape_pc_id1, shape_pc_id2].view(bs, num_pts, 3)
    
    if return_idx:
        shape_idxs = shape_pc_id2.view(bs, num_pts)
        return shape_pcs, shape_idxs
    return shape_pcs


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def farthest_pts_sampling_tensor(pts, num_samples, return_sampled_idx=False):
    '''
    :param pts: bn, n, 3
    :param num_samples:
    :return:
    '''
    sampled_pts_idx = farthest_point_sample(pts, num_samples)
    sampled_pts_idx_viewed = sampled_pts_idx.view(sampled_pts_idx.shape[0]*sampled_pts_idx.shape[1]).long()
    batch_idxs = torch.tensor(range(pts.shape[0])).to(sampled_pts_idx.device).long()
    batch_idxs_viewed = batch_idxs[:, None].repeat(1, sampled_pts_idx.shape[1]).view(batch_idxs.shape[0]*sampled_pts_idx.shape[1])
    sampled_pts = pts[batch_idxs_viewed, sampled_pts_idx_viewed, :]
    sampled_pts = sampled_pts.view(pts.shape[0], num_samples, 3)

    if return_sampled_idx == False:
        return sampled_pts
    else:
        return sampled_pts, sampled_pts_idx
