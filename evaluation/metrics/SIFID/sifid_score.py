#!/usr/bin/env python3
"""Calculates ***Single Image*** Frechet Inception Distance (SIFID) to evalulate Single-Image-GANs
Code was adapted from:
https://github.com/mseitzer/pytorch-fid.git
Which was adapted from the TensorFlow implementation of:

                                 
 https://github.com/bioinf-jku/TTUR

The FID metric calculates the distance between two distributions of images.
The SIFID calculates the distance between the distribution of deep features of a single real image and a single fake image.
Copyright 2018 Institute of Bioinformatics, JKU Linz
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.                                                               
"""

import os
import os.path as osp
import pathlib
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import cv2
import imageio
import torch
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import torchvision
import numpy as np
from scipy import linalg
from glob import glob
from tqdm import tqdm

from inception import InceptionV3
from resnet50 import ResNet50
from joblib.externals.loky.backend.context import get_context

def calculate_activation_statistics(img, model, dims=64):
    """Calculation of the statistics used by the FID.
    Params:
    -- img         : Image of shape (1, 3, HEIGHT, WIDTH)
    -- model       : Instance of inception model
    -- dims        : Dimensionality of features returned by Inception
    Returns:
    -- mu    : The mean over samples of the activations of the inception model.
    -- sigma : The covariance matrix of the activations of the inception model.
    """
    model.eval()
    act = model(img)[0].squeeze().permute(1, 2, 0).view(-1, dims).cpu().numpy()
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma
    

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'
    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)



class ImageDataset(Dataset):
    def __init__(self, path1, path2, pre_fetch=False):
        self.imgs1 = sorted(glob(osp.join(path1, '*.png')))
        self.imgs2 = sorted(glob(osp.join(path2, '*.png')))
        assert len(self.imgs1) == len(self.imgs2), f"{path1} and {path2}"
        
    def __getitem__(self, i):
        # img1 = torchvision.io.read_image(self.imgs1[i])
        # img2 = torchvision.io.read_image(self.imgs2[i])
        # print(self.imgs1[i], self.imgs2[i])
        # exit()
        assert osp.basename(self.imgs1[i]) == osp.basename(self.imgs2[i])

        img1 = torch.from_numpy(imageio.imread(self.imgs1[i])).permute(2, 0, 1)
        img2 = torch.from_numpy(imageio.imread(self.imgs2[i])).permute(2, 0, 1)
        if img1.max() > 1:
            img1 = img1 / 255.0
        if img2.max() > 1:
            img2 = img2 / 255.0

        return img1.float(), img2.float()

    def __len__(self):
        return len(self.imgs1)

def calculate_sifid_given_paths(path1, path2, dims=64, model='InceptionV3', batch_size=1, cuda=True, n_jobs=4):
    """Calculates the SIFID/SwAV_SIFID of two paths, support InceptionV3 and ResNet50 (SwAV)."""
    if model == 'InceptionV3':
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
        model = InceptionV3([block_idx])
    elif model == 'SwAV':
        block_idx = ResNet50.BLOCK_INDEX_BY_DIM[dims]
        model = ResNet50([block_idx])
    device = torch.device('cuda' if cuda else 'cpu')
    if cuda:
        model.to(device)

    fid_values = []
    dataset = ImageDataset(path1, path2)
    # loader = DataLoader(dataset, batch_size=1, num_workers=6, pin_memory=False, multiprocessing_context=get_context('loky'))
    # loader = DataLoader(dataset, batch_size=1, num_workers=6, pin_memory=True)
    loader = DataLoader(dataset, batch_size=1, num_workers=4, pin_memory=False)
    for img1, img2 in tqdm(loader):
        m1, s1 = calculate_activation_statistics(img1.to(device), model, dims)
        m2, s2 = calculate_activation_statistics(img2.to(device), model, dims)
        fid = calculate_frechet_distance(m1, s1, m2, s2)
        fid_values.append(fid)
    
    return np.mean(fid_values)
