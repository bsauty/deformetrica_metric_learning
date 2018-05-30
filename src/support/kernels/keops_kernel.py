from support.kernels import AbstractKernel
from pykeops.torch import generic_sum
from support.utilities.general_settings import Settings


class KeopsKernel(AbstractKernel):
    def __init__(self, kernel_width=None):
        self.kernel_type = 'keops'
        super().__init__(kernel_width)

        self.gaussian_convolve = generic_sum(
            "Exp(-G*SqDist(X,Y)) * P",
            "O = Vx(" + str(Settings().dimension) + ")",
            "G = Pm(1)",
            "X = Vx(" + str(Settings().dimension) + ")",
            "Y = Vy(" + str(Settings().dimension) + ")",
            "P = Vy(" + str(Settings().dimension) + ")")

        self.varifold_convolve = generic_sum(
            "Exp(-(WeightedSqDist(G, X, Y))) * Pow((Nx, Ny), 2) * P",
            "O = Vx(1)",
            "G = Pm(1)",
            "X = Vx(" + str(Settings().dimension) + ")",
            "Y = Vy(" + str(Settings().dimension) + ")",
            "Nx = Vx(" + str(Settings().dimension) + ")",
            "Ny = Vy(" + str(Settings().dimension) + ")",
            "P = Vy(1)")

        self.gaussian_convolve_gradient_x = generic_sum(
            "(Px, Py) * Exp(-G*SqDist(X,Y)) * (X-Y) * ",
            "O = Vx(" + str(Settings().dimension) + ")",
            "G = Pm(1)",
            "X = Vx(" + str(Settings().dimension) + ")",
            "Y = Vy(" + str(Settings().dimension) + ")",
            "Px = Vx(" + str(Settings().dimension) + ")",
            "Py = Vy(" + str(Settings().dimension) + ")")

    def convolve(self, x, y, p, backend='auto', mode='gaussian'):
        if mode == 'gaussian':
            return self.gaussian_convolve(self.gamma, x, y, p, backend=backend)
        elif mode == 'varifold':
            x, nx = x
            y, ny = y
            return self.varifold_convolve(self.gamma, x, y, nx, ny, p, backend=backend)
        else:
            raise RuntimeError('Unknown kernel mode.')

    def convolve_gradient(self, px, x, y=None, py=None, backend='auto', mode='gaussian'):

        if y is None:
            y = x
        if py is None:
            py = px

        return -2 * self.gamma * self.gaussian_convolve_gradient_x(self.gamma, x, y, px, py, backend=backend)