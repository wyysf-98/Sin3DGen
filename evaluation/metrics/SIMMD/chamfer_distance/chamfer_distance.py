import os
import os.path as osp
import torch
from torch.utils.cpp_extension import load
cd = load(name="cd",
          sources=[f"{osp.dirname(os.path.abspath(__file__))}/chamfer_distance.cpp",
                   f"{osp.dirname(os.path.abspath(__file__))}/chamfer_distance.cu"])

class ChamferDistanceFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xyz1, xyz2):
        """
        Args:
            xyz1 (torch.Tensor): (B, N, 3)
            xyz2 (torch.Tensor): (B, N, 3)
        """
        batchsize, n, _ = xyz1.size()
        _, m, _ = xyz2.size()
        xyz1 = xyz1.contiguous()
        xyz2 = xyz2.contiguous()
        dist1 = torch.zeros(batchsize, n)
        dist2 = torch.zeros(batchsize, m)

        idx1 = torch.zeros(batchsize, n, dtype=torch.int)
        idx2 = torch.zeros(batchsize, m, dtype=torch.int)

        if not xyz1.is_cuda:
            cd.forward(xyz1, xyz2, dist1, dist2, idx1, idx2)
        else:
            dist1 = dist1.cuda()
            dist2 = dist2.cuda()
            idx1 = idx1.cuda()
            idx2 = idx2.cuda()
            cd.forward_cuda(xyz1, xyz2, dist1, dist2, idx1, idx2)

        ctx.save_for_backward(xyz1, xyz2, idx1, idx2)

        return dist1, dist2

    @staticmethod
    def backward(ctx, graddist1, graddist2):
        xyz1, xyz2, idx1, idx2 = ctx.saved_tensors

        graddist1 = graddist1.contiguous()
        graddist2 = graddist2.contiguous()

        gradxyz1 = torch.zeros(xyz1.size())
        gradxyz2 = torch.zeros(xyz2.size())

        if not graddist1.is_cuda:
            cd.backward(xyz1, xyz2, gradxyz1, gradxyz2, graddist1, graddist2, idx1, idx2)
        else:
            gradxyz1 = gradxyz1.cuda()
            gradxyz2 = gradxyz2.cuda()
            cd.backward_cuda(xyz1, xyz2, gradxyz1, gradxyz2, graddist1, graddist2, idx1, idx2)

        return gradxyz1, gradxyz2


class ChamferDistance(torch.nn.Module):
    def __init__(self):
        super(ChamferDistance, self).__init__()

    def forward(self, xyz1, xyz2, transpose=True, reduce_mean=True, return_raw=False):
        """Chamfer Distance (Approx)

        Args:
            xyz1 (torch.Tensor): (B, 3, N)
            xyz2 (torch.Tensor): (B, 3, N)
            transpose (bool): whether to transpose inputs as it might be BCN format.
                Extensions only support BNC format.

        Returns:
            chamfer_dist (torch.Tensor): (b)

        """

        if xyz1.dim() == 2:
            xyz1 = xyz1.unsqueeze(0)
        if xyz2.dim() == 2:
            xyz2 = xyz2.unsqueeze(0)
        if transpose:
            xyz1 = xyz1.transpose(1, 2)
            xyz2 = xyz2.transpose(1, 2)
        dist1, dist2 = ChamferDistanceFunction.apply(xyz1, xyz2)
        if return_raw:
            if reduce_mean:
                return dist1.mean(), dist2.mean()
            return dist1, dist2

        chamfer_dist = (dist1.mean(dim=1) + dist2.mean(dim=1)) / 2
        if reduce_mean:
            chamfer_dist = chamfer_dist.mean()
            
        return chamfer_dist