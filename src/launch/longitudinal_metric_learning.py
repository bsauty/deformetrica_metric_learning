import os.path
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../../../')

import torch
from torch.autograd import Variable
import numpy as np
import warnings

from pydeformetrica.src.support.utilities.general_settings import Settings
from pydeformetrica.src.core.model_tools.manifolds.generic_geodesic import GenericGeodesic
from pydeformetrica.src.core.model_tools.manifolds.one_dimensional_exponential import OneDimensionalExponential
from pydeformetrica.src.core.model_tools.manifolds.exponential_factory import ExponentialFactory

import matplotlib.pyplot as plt

# We would like to try the geodesic object on one_dim_manifold !
exponential_factory = ExponentialFactory()
exponential_factory.set_manifold_type("one_dimensional")

manifold_parameters = {}
manifold_parameters['number_of_interpolation_points'] = 20
manifold_parameters['width'] = 0.1/20
manifold_parameters['interpolation_points_torch'] = Variable(torch.from_numpy(np.linspace(0, 1, 20))
                                                             .type(Settings().tensor_scalar_type))
manifold_parameters['interpolation_values_torch'] = Variable(torch.from_numpy(np.random.binomial(2, 0.5, 20))
                                                             .type(Settings().tensor_scalar_type))
exponential_factory.set_parameters(manifold_parameters)

generic_geodesic = GenericGeodesic(exponential_factory)

generic_geodesic.set_t0(0.)
generic_geodesic.set_tmin(-1.)
generic_geodesic.set_tmax(1.)

generic_geodesic.set_concentration_of_time_points(50)


q0 = 0.5
v0 = 1.
p0 = v0

q = Variable(torch.Tensor([q0]), requires_grad=True).type(torch.DoubleTensor)
p = Variable(torch.Tensor([p0]), requires_grad=False).type(torch.DoubleTensor)

generic_geodesic.set_position_t0(q)
generic_geodesic.set_momenta_t0(p)

generic_geodesic.update()

traj = generic_geodesic._get_geodesic_trajectory()
print(generic_geodesic._get_times())

def plot_points(l, times=None):
    l_numpy = [elt.data.numpy() for elt in l]
    if times is not None:
        t = times
    else:
        t = np.linspace(0., 1., len(l_numpy))
    plt.plot(t, l_numpy)

plot_points(traj)
plt.show()


