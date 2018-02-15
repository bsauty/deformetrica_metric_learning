import os.path
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../../../../../')

from pydeformetrica.src.core.model_tools.manifolds.manifold_calculator import ManifoldCalculator

import torch
from torch.autograd import Variable

class OneDimensionalExponential():

    def __init__(self):
        self.nb_interpolation_points = 20
        self.width = 0.1/self.nb_interpolation_points
        interpolation_points = np.linspace(0., 1., self.nb_interpolation_points)
        interpolation_values = np.random.binomial(2, 0.5, nb_points)
        self.interpolation_points_torch = Variable(torch.from_numpy(interpolation_points)).type(torch.DoubleTensor)
        self.interpolation_values_torch = Variable(torch.from_numpy(interpolation_values)).type(torch.DoubleTensor)
        self.manifold_calculator = ManifoldCalculator()

        self.number_of_time_points = 10
        self.position_t = []

        self.initial_momenta = None
        self.initial_position = None

    def inverse_metric(self, q):
        squared_distances = (self.interpolation_points_torch - q)**2.
        return torch.dot(self.interpolation_values_torch, torch.exp(-1.*squared_distances/self.width))

    def flow(self, q, p):
        self.position_t = ManifoldCalculator.exponential(
            q, p, nb_steps=self.number_of_time_points,
            inverse_metric=self.inverse_metric, dp=None)

    def update(self):
        pass

    def update_norm_squared(self):
        pass

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