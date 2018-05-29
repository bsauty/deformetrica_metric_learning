import torch
from support.kernels.abstract_kernel import AbstractKernel


def gaussian(r2, s):
    return torch.exp(-r2 / (s * s))


def binet(prs):
    return prs ** 2


class TorchKernel(AbstractKernel):
    ####################################################################################################################
    ### Constructor:
    ####################################################################################################################

    def __init__(self, kernel_width=None):
        self.kernel_type = 'torch'
        self.kernel_width = kernel_width

    ####################################################################################################################
    ### Public methods:
    ####################################################################################################################

    def convolve(self, x, y, p, mode='gaussian'):
        if mode == 'gaussian':
            sq = self._squared_distances(x, y)
            return torch.mm(torch.exp(-sq / (self.kernel_width ** 2)), p)
        elif mode == 'varifold':
            sq = self._squared_distances(x[0], y[0])
            return torch.mm(gaussian(sq, self.kernel_width) * binet(torch.mm(x[1], torch.t(y[1]))), p)
        else:
            raise RuntimeError('Unknown kernel mode.')

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

        # B=(x_i - y_j)*exp(-(x_i - y_j)^2/(ker^2))/(ker^2).
        B = self._differences(x, y) * A

        return (- 2 * torch.sum(px * (torch.matmul(B, py)), 2) / (self.kernel_width ** 2)).t()

        # # Hamiltonian
        # H = torch.dot(p.view(-1), self.Convolve(x,p,y).view(-1))
        # # return torch.autograd.grad(H, p, create_graph=True)[0]
        # out = torch.autograd.grad(H, p)[0]
        # return out

    def _differences(self, x, y):
        """
        Returns the matrix of $(x_i - y_j)$.
        Output is of size (D, M, N).
        """
        x_col = x.t().unsqueeze(2)  # (M,D) -> (D,M,1)
        y_lin = y.t().unsqueeze(1)  # (N,D) -> (D,1,N)
        return x_col - y_lin
