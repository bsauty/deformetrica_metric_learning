import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../../../')

import torch
from torch.autograd import Variable

from pydeformetrica.src.support.utilities.general_settings import Settings


class TorchKernel:

    ####################################################################################################################
    ### Constructor:
    ####################################################################################################################

    def __init__(self):
        self.KernelWidth = None

    ####################################################################################################################
    ### Public methods:
    ####################################################################################################################

    def Convolve(self, x, y, p):
        assert self.KernelWidth != None, "torch kernel width not initialized"  # TODO : is this assert expensive when called 100000 times ?

        sq = self._squared_distances(x, y)
        out = torch.mm(torch.exp(-sq / (self.KernelWidth ** 2)), p)
        return out

    def ConvolveGradient(self, px, x, y=None, py=None):
        # Default values.
        if y is None: y = x
        if py is None: py = px

        # Asserts.
        assert (x.size()[0] == px.size()[0])
        assert (y.size()[0] == py.size()[0])
        assert (px.size()[1] == py.size()[1])
        assert (x.size()[1] == y.size()[1])

        # A=exp(-(x_i - y_j)^2/(ker^2)).
        sq = self._squared_distances(x, y)
        A = torch.exp(-sq / (self.KernelWidth ** 2))

        # B=2*(x_i - y_j)*exp(-(x_i - y_j)^2/(ker^2))/(ker^2).
        B = self._differences(x, y) * A

        return (- 2 * torch.sum(px * (torch.matmul(B, py)), 2) / (self.KernelWidth ** 2)).t()

        # # Hamiltonian
        # H = torch.dot(p.view(-1), self.Convolve(x,p,y).view(-1))
        # # return torch.autograd.grad(H, p, create_graph=True)[0]
        # out = torch.autograd.grad(H, p)[0]
        # return out

    ####################################################################################################################
    ### Private methods:
    ####################################################################################################################

    def _squared_distances(self, x, y):
        """
        Returns the matrix of $(x_i - y_j)^2$.
        Output is of size (1, M, N).
        """
        x_col = x.unsqueeze(1)  # (M,D) -> (M,1,D)
        y_lin = y.unsqueeze(0)  # (N,D) -> (1,N,D)
        return torch.sum((x_col - y_lin) ** 2, 2)

    def _differences(self, x, y):
        """
        Returns the matrix of $(x_i - y_j)$.
        Output is of size (D, M, N).
        """
        x_col = x.t().unsqueeze(2)  # (M,D) -> (D,M,1)
        y_lin = y.t().unsqueeze(1)  # (N,D) -> (D,1,N)
        return x_col - y_lin
