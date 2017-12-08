import torch
import sys
import os
from torch.autograd import Variable

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../../../')
from pydeformetrica.src.support.utilities.singleton_pattern import Singleton
from pydeformetrica.src.support.utilities.kernel_types import KernelType
from pydeformetrica.src.support.utilities.general_settings import *


class TorchKernel:
    ####################################################################################################################
    ### Constructor:
    ####################################################################################################################

    def __init__(self):
        self.KernelWidth = None

    ####################################################################################################################
    ### Public methods:
    ####################################################################################################################

    def Convolve(self, x, p, y):
        assert self.KernelWidth != None, "torch kernel width not initialized"  # TODO : is this assert expensive when called 100000 times ?
        sq = self._squared_distances(x, y)
        return torch.mm(torch.exp(-sq / (self.KernelWidth ** 2)), p)

    def ConvolveGradient(self, x, p, y):

        assert (y.size()[0] == p.size()[0])
        dim = Settings().Dimension # Shorthand.
        weightDim = p.size()[1]

        gradK = []
        for i in range(x.size()[0]):
            Gi = Variable(torch.zeros(weightDim, dim).type(Settings().TensorType))

            for j in range(y.size()[0]):
                g = self._evaluate_kernel_gradient(x, y, i, j)

                for k in range(weightDim):
                    pjk = p[j, k]

                    for l in range(dim):
                        Gi[k, l] = Gi[k, l] + g[l] * pjk

            gradK.append(Gi)

        result = Variable(torch.zeros(x.size()).type(Settings().TensorType))
        for j in range(x.size()[0]):
            result[j] = torch.mm(torch.t(gradK[j]), p[j].unsqueeze(1))

        return result

        # #TODO: implement the actual formula
        # #Hamiltonian
        # H = torch.dot(p.view(-1), self.Convolve(x,p,y).view(-1))
        # # return torch.autograd.grad(H, p, create_graph=True)[0]
        # out = torch.autograd.grad(H, p)[0]
        # return out

    ####################################################################################################################
    ### Private methods:
    ####################################################################################################################

    def _squared_distances(self, x, y):
        """
        Returns the matrix of $\|x_i-y_j\|^2$.
        Output is of size (1,M,N)
        """
        x_col = x.unsqueeze(1)  # (N,D) -> (N,1,D)
        y_lin = y.unsqueeze(0)  # (M,D) -> (1,M,D)
        return torch.sum((x_col - y_lin) ** 2, 2)

    def _evaluate_kernel_gradient(self, x, y, i, j):
        assert(x.size()[1] == y.size()[1])

        result = Variable(torch.zeros((Settings().Dimension,)).type(Settings().TensorType))
        dist_squared = Variable(torch.zeros((1, 1)).type(Settings().TensorType))

        for k in range(x.size()[1]):
            diff = x[i, k] - y[j, k]
            result[k] = diff
            dist_squared += diff * diff

        return - torch.t(result * 2.0 * torch.exp(- dist_squared / self.KernelWidth) / self.KernelWidth)

