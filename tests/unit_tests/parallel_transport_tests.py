import os
import sys
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../../../')
from pydeformetrica.src.support.utilities.general_settings import Settings
from pydeformetrica.src.core.model_tools.deformations.geodesic import Geodesic
from pydeformetrica.src.support.kernels.kernel_functions import create_kernel
from pydeformetrica.src.in_out.utils import *
import unittest
import torch
from torch.autograd import Variable


#Tests are done both in 2 and 3d.

class ParallelTransportTests(unittest.TestCase):
    """
    Methods with names starting by "test" will be run
    """
    def setUp(self):
        pass

    def test_parallel_transport(self):
        """
        test the parallel transport on a chosen example converges towards the truth (checked from old rusty C++ deformetrica)
        """
        control_points = read_2D_array(os.path.join(Settings().unit_tests_data_dir, "parallel_transport","control_points.txt"))
        momenta = read_momenta(os.path.join(Settings().unit_tests_data_dir, "parallel_transport","geodesic_momenta.txt"))[0]
        momenta_to_transport = read_momenta(os.path.join(Settings().unit_tests_data_dir, "parallel_transport", "momenta_to_transport.txt"))[0]
        transported_momenta_truth = read_momenta(os.path.join(Settings().unit_tests_data_dir, "parallel_transport", "ground_truth_transport.txt"))[0]

        control_points_torch = Variable(torch.from_numpy(control_points).type(Settings().tensor_scalar_type))
        momenta_torch = Variable(torch.from_numpy(momenta).type(Settings().tensor_scalar_type))
        momenta_to_transport_torch = Variable(torch.from_numpy(momenta_to_transport).type(Settings().tensor_scalar_type))

        geodesic = Geodesic()
        geodesic.concentration_of_time_points = 20
        geodesic.set_kernel(create_kernel("exact", 0.01))

        geodesic.tmin = 0.
        geodesic.tmax = 9.
        geodesic.t0 = 0.
        geodesic.set_momenta_t0(momenta_torch)
        geodesic.set_control_points_t0(control_points_torch)
        geodesic.update()

        # Now we transport!
        parallel_transport_trajectory = geodesic.parallel_transport(momenta_to_transport_torch)

        self.assertTrue(np.linalg.norm(transported_momenta_truth - parallel_transport_trajectory[-1].data.numpy())/np.linalg.norm(transported_momenta_truth) <= 1e-10)
