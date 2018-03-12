import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../../../')

import torch

# TODO : store a kernelwidthsquared attribute to save a multiplication...

class ExactKernel:
    ####################################################################################################################
    ### Constructor:
    ####################################################################################################################

    def __init__(self):
        self.kernel_type = 'exact'
        self.kernel_width = None

    ####################################################################################################################
    ### Public methods:
    ####################################################################################################################

    def convolve(self, x, y, p, mode=None):
        assert self.kernel_width != None, "exact kernel width not initialized"

        if mode is None:
            sq = self._squared_distances(x, y)
            out = torch.mm(torch.exp(-sq / (self.kernel_width ** 2)), p)
            return out

        else:
            def gaussian(r2, s):
                return torch.exp(-r2 / (s * s))

            def binet(prs):
                return prs ** 2

            sq = self._squared_distances(x[0], y[0])
            out = torch.mm(gaussian(sq, self.kernel_width) * binet(torch.mm(x[1], torch.t(y[1]))), p)
            return out

    def convolve_gradient(self, px, x, y=None, py=None):
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
        A = torch.exp(-sq / (self.kernel_width ** 2))

        # B=2*(x_i - y_j)*exp(-(x_i - y_j)^2/(ker^2))/(ker^2).
        B = self._differences(x, y) * A

        return (- 2 * torch.sum(px * (torch.matmul(B, py)), 2) / (self.kernel_width ** 2)).t()

        # # Hamiltonian
        # H = torch.dot(p.view(-1), self.Convolve(x,p,y).view(-1))
        # # return torch.autograd.grad(H, p, create_graph=True)[0]
        # out = torch.autograd.grad(H, p)[0]
        # return out

    def get_kernel_matrix(self, x, y=None):
        """
        returns the kernel matrix, A_{ij} = exp(-|x_i-x_j|^2/sigma^2)
        """
        if y is None: y = x
        assert (x.size()[0] == y.size()[0])
        sq = self._squared_distances(x, y)
        return torch.exp(-sq / (self.kernel_width ** 2))

    ####################################################################################################################
    ### Private methods:
    ####################################################################################################################

    def _squared_distances(self, x, y):
        """
        Returns the matrix of $(x_i - y_j)^2$.
        Output is of size (1, M, N).
        """
        return torch.sum((x.unsqueeze(1) - y.unsqueeze(0)) ** 2, 2)

    def _differences(self, x, y):
        """
        Returns the matrix of $(x_i - y_j)$.
        Output is of size (D, M, N).
        """
        x_col = x.t().unsqueeze(2)  # (M,D) -> (D,M,1)
        y_lin = y.t().unsqueeze(1)  # (N,D) -> (D,1,N)
        return x_col - y_lin
