import os
import os.path as osp
import torch
from torch.utils.cpp_extension import load
emd = load(name="emd",
          sources=[f"{osp.dirname(os.path.abspath(__file__))}/earth_mover_distance.cpp",
                   f"{osp.dirname(os.path.abspath(__file__))}/earth_mover_distance.cu"])

class EarthMoverDistanceFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xyz1, xyz2):
        xyz1 = xyz1.contiguous()
        xyz2 = xyz2.contiguous()
        assert xyz1.is_cuda and xyz2.is_cuda, "Only support cuda currently."
        match = emd.approxmatch_forward(xyz1, xyz2)
        cost = emd.matchcost_forward(xyz1, xyz2, match)
        ctx.save_for_backward(xyz1, xyz2, match)
        return cost

    @staticmethod
    def backward(ctx, grad_cost):
        xyz1, xyz2, match = ctx.saved_tensors
        grad_cost = grad_cost.contiguous()
        grad_xyz1, grad_xyz2 = emd.matchcost_backward(grad_cost, xyz1, xyz2, match)
        return grad_xyz1, grad_xyz2


class EarthMoverDistance(torch.nn.Module):        
    def __init__(self):
        super(EarthMoverDistance, self).__init__()

    def forward(self, xyz1, xyz2, transpose=True, reduce_mean=True):
        """Earth Mover Distance (Approx)

        Args:
            xyz1 (torch.Tensor): (B, 3, N)
            xyz2 (torch.Tensor): (B, 3, N)
            transpose (bool): whether to transpose inputs as it might be BCN format.
                Extensions only support BNC format.

        Returns:
            earth_mover_dist (torch.Tensor): (b)

        """
        if xyz1.dim() == 2:
            xyz1 = xyz1.unsqueeze(0)
        if xyz2.dim() == 2:
            xyz2 = xyz2.unsqueeze(0)
        if transpose:
            xyz1 = xyz1.transpose(1, 2)
            xyz2 = xyz2.transpose(1, 2)
        earth_mover_dist = EarthMoverDistanceFunction.apply(xyz1, xyz2) / min(xyz1.shape[1], xyz2.shape[1])

        if reduce_mean:
            earth_mover_dist = torch.mean(earth_mover_dist)
            
        return earth_mover_dist

