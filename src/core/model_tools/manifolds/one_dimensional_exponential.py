import os.path
import sys
import numpy as np
import warnings


sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../../../../../')

from pydeformetrica.src.core.model_tools.manifolds.manifold_calculator import ManifoldCalculator

import torch
from torch.autograd import Variable

class OneDimensionalExponential:

    def __init__(self):
        self.number_of_interpolation_points = None
        self.width = None
        self.interpolation_points_torch = None
        self.interpolation_values_torch = None
        self.manifold_calculator = ManifoldCalculator()

        self.number_of_time_points = 10
        self.position_t = []

        self.initial_momenta = None
        self.initial_position = None

        self.is_modified = True

        self.norm_squared = None

    def set_interpolation_values(self, interpolation_values):
        """
        Torch tensors
        """
        self.interpolation_values_torch = interpolation_values
        self.is_modified = True

    def get_initial_position(self):
        return self.initial_position

    def set_initial_position(self, q):
        self.initial_position = q
        self.is_modified = True

    def set_initial_momenta(self, p):
        self.initial_momenta = p
        self.is_modified = True

    def inverse_metric(self, q):
        squared_distances = (self.interpolation_points_torch - q)**2.
        return torch.dot(self.interpolation_values_torch, torch.exp(-1.*squared_distances/self.width))

    def flow(self):
        if self.initial_position is None:
            msg = "In exponential update, I am not flowing because I don't have an initial position"
            warnings.warn(msg)
        elif self.initial_momenta is None:
            msg = "In exponential update, I am not flowing because I don't have an initial momenta"
            warnings.warn(msg)
        else:
            self.position_t = self.manifold_calculator.exponential(
                self.initial_position, self.initial_momenta,
                nb_steps=self.number_of_time_points,
                inverse_metric=self.inverse_metric)

    def update(self):
        """
        Update the state of the object, depending on what's needed.
        This is the only clean way to call flow on the deformation.
        """
        assert self.number_of_time_points > 0
        if self.is_modified:
            self.flow()
            self.update_norm_squared()
            self.is_modified = False


    def update_norm_squared(self):
        # Should be a torch variable always (torch.dot returns variable ?)
        self.norm_squared = self.manifold_calculator.hamiltonian(
            self.initial_position, self.initial_momenta, self.inverse_metric)

    def set_parameters(self, extra_parameters):
        """
        In this case, the parameters are the interpolation values
        """
        assert extra_parameters.size() == self.interpolation_values_torch.size(),\
            "Wrong format of parameters"
        self.interpolation_values_torch = extra_parameters
        self.is_modified = True


    # def plot_points(self, l, times=None):
    #     l_numpy = [elt.data.numpy() for elt in l]
    #     if times is not None:
    #         t = times
    #     else:
    #         t = np.linspace(0., 1., len(l_numpy))
    #     plt.plot(t, l_numpy)
    #
    # def plot_inverse_metric(self):
    #     times = np.linspace(0., 1., 300)
    #     times_torch = Variable(torch.from_numpy(times)).type(torch.DoubleTensor)
    #     metric_values = [inverse_metric_one_dim_manifold(t).data.numpy()[0] for t in times_torch]
    #     square_root_metric_values = [np.sqrt(elt) for elt in metric_values]
    #     # plt.plot(times, metric_values)
    #     plt.plot(times, square_root_metric_values)
    #



    # for i in range(20):
    #     interpolation_points = np.linspace(0., 1., nb_points)
    #     interpolation_values = np.random.binomial(2, 0.5, nb_points)
    #     print(interpolation_values)
    #     interpolation_points_torch = Variable(torch.from_numpy(interpolation_points)).type(torch.DoubleTensor)
    #     interpolation_values_torch = Variable(torch.from_numpy(interpolation_values)).type(torch.DoubleTensor)
    #
    #     q0 = 0.5
    #     v0 = 1.
    #     p0 = 1./inverse_metric_one_dim_manifold(q0).data.numpy()[0] * v0
    #
    #
    #     q = Variable(torch.Tensor([q0]), requires_grad=True).type(torch.DoubleTensor)
    #     p = Variable(torch.Tensor([p0]), requires_grad=False).type(torch.DoubleTensor)
    #
    #     time1 = time.time()
    #     times, traj_q = exponential(q, p, inverse_metric_one_dim_manifold, nb_steps=100)
    #     time2 = time.time()
    #     print("Time:", time2 - time1)
    #
    #     print(traj_q[-1])
    #     plot_points(traj_q, times=times)
    #
    #
    #     # plot_inverse_metric()
    #
    #
    # plt.show()