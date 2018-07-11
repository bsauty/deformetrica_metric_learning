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

    def __init__(self, kernel_width=None, tensor_scalar_type=None, device='auto', **kwargs):
        super().__init__(kernel_width, device)
        self.kernel_type = 'torch'
        self.tensor_scalar_type = tensor_scalar_type

    ####################################################################################################################
    ### Public methods:
    ####################################################################################################################

    def convolve(self, x, y, p, mode='gaussian'):
        if self.device == 'GPU':
            if self.tensor_scalar_type == torch.cuda.FloatTensor:  # Full-cuda case.
                return self._convolve(x, y, p, mode)
            else:
                return self._convolve(x.cuda(), y.cuda(), p.cuda(), mode).cpu()

        elif self.device == 'CPU':
            if self.tensor_scalar_type == torch.cuda.FloatTensor:  # Full-cuda case.
                return self._convolve(x.cpu(), y.cpu(), p.cpu(), mode).cuda()
            else:
                return self._convolve(x, y, p, mode)

        elif self.device == 'auto':
            return self._convolve(x, y, p, mode)

        else:
            raise RuntimeError('Unknown kernel device. Possibles values are "auto", "CPU", or "GPU".')

    def convolve_gradient(self, px, x, y=None, py=None):

        if y is None: y = x
        if py is None: py = px

        if self.device == 'GPU':
            if self.tensor_scalar_type == torch.cuda.FloatTensor:  # Full-cuda case.
                return self._convolve_gradient(px, x, y, py)
            else:
                return self._convolve_gradient(px.cuda(), x.cuda(), y.cuda(), py.cuda()).cpu()

        elif self.device == 'CPU':
            if self.tensor_scalar_type == torch.cuda.FloatTensor:  # Full-cuda case.
                return self._convolve_gradient(px.cpu(), x.cpu(), y.cpu(), py.cpu()).cuda()
            else:
                return self._convolve_gradient(px, x, y, py)

        elif self.device == 'auto':
            return self._convolve_gradient(px, x, y, py)

        else:
            raise RuntimeError('Unknown kernel device. Possibles values are "auto", "CPU", or "GPU".')

    ####################################################################################################################
    ### Auxiliary methods:
    ####################################################################################################################

    def _convolve(self, x, y, p, mode):
        if mode in ['gaussian', 'pointcloud']:
            sq = self._squared_distances(x, y)
            return torch.mm(torch.exp(-sq / (self.kernel_width ** 2)), p)
        elif mode == 'varifold':
            sq = self._squared_distances(x[0], y[0])
            return torch.mm(gaussian(sq, self.kernel_width) * binet(torch.mm(x[1], torch.t(y[1]))), p)
        else:
            raise RuntimeError('Unknown kernel mode.')

    def _convolve_gradient(self, px, x, y, py):

        # A=exp(-(x_i - y_j)^2/(ker^2)).
        sq = self._squared_distances(x, y)
        A = torch.exp(-sq / (self.kernel_width ** 2))

        # B=(x_i - y_j)*exp(-(x_i - y_j)^2/(ker^2))/(ker^2).
        B = self._differences(x, y) * A

        return (- 2 * torch.sum(px * (torch.matmul(B, py)), 2) / (self.kernel_width ** 2)).t()

    def _differences(self, x, y):
        """
        Returns the matrix of $(x_i - y_j)$.
        Output is of size (D, M, N).
        """
        x_col = x.t().unsqueeze(2)  # (M,D) -> (D,M,1)
        y_lin = y.t().unsqueeze(1)  # (N,D) -> (D,1,N)
        return x_col - y_lin
