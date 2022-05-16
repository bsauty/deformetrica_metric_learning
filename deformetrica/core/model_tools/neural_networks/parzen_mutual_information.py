"""Implementation of Kernel Density Estimation (KDE) [1].
Kernel density estimation is a nonparameteric density estimation method. It works by
placing kernels K on each point in a "training" dataset D. Then, for a test point x, 
p(x) is estimated as p(x) = 1 / |D| \sum_{x_i \in D} K(u(x, x_i)), where u is some 
function of x, x_i. In order for p(x) to be a valid probability distribution, the kernel
K must also be a valid probability distribution.
References (used throughout the file):
    [1]: https://en.wikipedia.org/wiki/Kernel_density_estimation
"""

import abc
from cgi import test

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = 'cpu'

class Kernel(abc.ABC, nn.Module):
    """Base class which defines the interface for all kernels."""

    def __init__(self, bandwidth=0.04):
        """Initializes a new Kernel.
        Args:
            bandwidth: The kernel's (band)width.
        """
        super().__init__()
        self.bandwidth = bandwidth

    def _diffs(self, test_Xs, train_Xs):
        """Computes difference between each x in test_Xs with all train_Xs."""
        test_Xs = test_Xs.view(*test_Xs.shape[:2], 1, test_Xs.shape[2])
        train_Xs = train_Xs.view(1, 1, *train_Xs.shape)
        return test_Xs - train_Xs

    @abc.abstractmethod
    def forward(self, test_Xs, train_Xs):
        """Computes p(x) for each x in test_Xs given train_Xs."""

    @abc.abstractmethod
    def sample(self, train_Xs):
        """Generates samples from the kernel distribution."""


class GaussianKernel(Kernel):
    """Implementation of the Gaussian kernel."""

    def forward(self, test_Xs, train_Xs):
        diffs = self._diffs(test_Xs, train_Xs)
        var = self.bandwidth ** 2
        exp = torch.exp(-torch.norm(diffs, p=2, dim=3) ** 2 / (2 * var))
        coef = 1 / torch.sqrt(torch.tensor(2 * np.pi * var))
        return (coef * exp).mean(dim=2)

    def sample(self, train_Xs):
        device = train_Xs.device
        noise = torch.randn(train_Xs.shape) * self.bandwidth
        return train_Xs + noise    


class KernelDensityEstimator():
    """The KernelDensityEstimator model."""

    def __init__(self, train_Xs, kernel=None):
        """Initializes a new KernelDensityEstimator.
        Args:
            train_Xs: The "training" data to use when estimating probabilities.
            kernel: The kernel to place on each of the train_Xs.
        """
        super().__init__()
        self.kernel = kernel or GaussianKernel()
        self.train_Xs = train_Xs

    @property
    def device(self):
        return self.train_Xs.device

    # TODO(eugenhotaj): This method consumes O(train_Xs * x) memory. Implement an
    # iterative version instead.
    def forward(self, x):
        return self.kernel(x, self.train_Xs)

class MutualInformation(nn.Module):

    def __init__(self, n_grid=20):
        self.n_grid = n_grid
        return

    def forward(self, batch_1, batch_2):
        batch = torch.cat((batch_1.unsqueeze(0), batch_2.unsqueeze(0)))
        batch = batch.view(batch.shape[1], batch.shape[0], *batch.shape[2:])
        kde = KernelDensityEstimator(batch)

        grid = torch.tensor([[(i/self.n_grid, j/self.n_grid) for i in range(self.n_grid)] for j in range(self.n_grid)])
        grid = grid.to(device)
        #grid = grid.view(grid.shape[0]*grid.shape[1], 2)
        
        p_xy = kde.forward(grid)
        p_x = p_xy.sum(dim=1)
        p_y = p_xy.sum(dim=0)

        mi = 0
        for i in range(self.n_grid):
            for j in range(self.n_grid):
                mi += p_xy[i,j] * torch.log(p_xy[i,j]/(p_x[i]*p_y[j]))

        return mi

def parzen_mutual_information_loss(img_1, img_2, n_grid=20):
        
    batch = torch.cat((img_1.ravel().unsqueeze(0), img_2.ravel().unsqueeze(0)))
    batch = batch.view(batch.shape[1], batch.shape[0], *batch.shape[2:])
    kde = KernelDensityEstimator(batch)
    grid = torch.tensor([[(i/n_grid, j/n_grid) for i in range(n_grid)] for j in range(n_grid)])
    grid = grid.to(device)
    #grid = grid.view(grid.shape[0]*grid.shape[1], 2)
    
    p_xy = kde.forward(grid)
    p_xy = p_xy / p_xy.sum()
    p_x = p_xy.sum(dim=1)
    p_y = p_xy.sum(dim=0)
    mi = 0
    for i in range(n_grid):
        for j in range(n_grid):
            mi += p_xy[i,j] * torch.log(p_xy[i,j]/(p_x[i]*p_y[j]))
    
    return - mi

def main():
    """
    For debugging purposes only
    """
    img_1 = torch.load("/network/lustre/iss02/aramis/datasets/adni/caps/caps_v2021/subjects/sub-ADNI002S0295/ses-M00/deeplearning_prepare_data/image_based/t1_linear/sub-ADNI002S0295_ses-M00_T1w_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_T1w.pt")
    #img_2 = torch.load("/network/lustre/iss02/aramis/datasets/adni/caps/caps_v2021/subjects/sub-ADNI002S0295/ses-M06/deeplearning_prepare_data/image_based/t1_linear/sub-ADNI002S0295_ses-M06_T1w_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_T1w.pt")
    img_2 = torch.load("/network/lustre/iss02/aramis/datasets/adni/caps/caps_v2021/subjects/sub-ADNI126S0865/ses-M00/deeplearning_prepare_data/image_based/t1_linear/sub-ADNI126S0865_ses-M00_T1w_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_T1w.pt")
    img_1, img_2 = img_1[:,4:164:2,8:200:2,8:168:2], img_2[:,4:164:2,8:200:2,8:168:2]
    img_1, img_2 = img_1/img_1.max(), img_2/img_2.max()
    
    img_1 = img_1.to(device)
    img_2 = Variable(img_2.to(device), requires_grad=True)
    time_start = time.time()
    for _ in range(10):
        loss = parzen_mutual_information_loss(img_1, img_2)
        loss.backward()
    print(time.time()-time_start)
    print(loss)
    return loss

if __name__ == '__main__':
    main()