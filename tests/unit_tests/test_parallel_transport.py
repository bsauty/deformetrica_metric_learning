import os
import unittest

import torch

from core import default
from core.model_tools.deformations.geodesic import Geodesic
import support.kernels as kernel_factory
from torch.autograd import Variable
from in_out.array_readers_and_writers import *
from unit_tests import unit_tests_data_dir


class ParallelTransportTests(unittest.TestCase):

    def setUp(self):
        self.tensor_scalar_type = default.tensor_scalar_type

    def test_parallel_transport(self):
        """
        test the parallel transport on a chosen example converges towards the truth (checked from C++ deformetrica)
        """
        control_points = read_2D_array(os.path.join(unit_tests_data_dir, "parallel_transport", "control_points.txt"))
        momenta = read_3D_array(os.path.join(unit_tests_data_dir, "parallel_transport", "geodesic_momenta.txt"))
        momenta_to_transport = read_3D_array(os.path.join(unit_tests_data_dir, "parallel_transport", "momenta_to_transport.txt"))
        transported_momenta_truth = read_3D_array(os.path.join(unit_tests_data_dir, "parallel_transport", "ground_truth_transport.txt"))

        # control_points = np.array([[0.1, 2., 0.2]])
        # momenta = np.array([[1., 0., 0.]])
        # momenta_to_transport = np.array([[0.2, 0.3, 0.4]])

        control_points_torch = Variable(torch.from_numpy(control_points).type(self.tensor_scalar_type))
        momenta_torch = Variable(torch.from_numpy(momenta).type(self.tensor_scalar_type))
        momenta_to_transport_torch = Variable(torch.from_numpy(momenta_to_transport).type(self.tensor_scalar_type))

        geodesic = Geodesic(
            dimension=2,
            dense_mode=False,
            tensor_scalar_type=self.tensor_scalar_type,
            deformation_kernel=kernel_factory.factory('torch', 0.01),
            number_of_time_points=default.number_of_time_points,
            t0=0.,
            use_rk2_for_shoot=True,
            concentration_of_time_points=10

        )

        geodesic.tmin = 0.
        geodesic.tmax = 9.
        geodesic.set_momenta_t0(momenta_torch)
        geodesic.set_control_points_t0(control_points_torch)
        geodesic.update()

        # Now we transport!
        transported_momenta = geodesic.parallel_transport(momenta_to_transport_torch, is_orthogonal=False)[-1].detach().numpy()
        self.assertTrue(np.allclose(transported_momenta, transported_momenta_truth, rtol=1e-4, atol=1e-1))


            # print(transported_momenta[-1].numpy())
            # self.assertTrue(np.allclose(transported_momenta_truth, transported_momenta[-1]))

        #self.assertTrue(np.linalg.norm(transported_momenta_truth - parallel_transport_trajectory[-1].data.numpy())/np.linalg.norm(transported_momenta_truth) <= 1e-5)


        # should be 1e-5 (relative) with 10 concentration wrt C++