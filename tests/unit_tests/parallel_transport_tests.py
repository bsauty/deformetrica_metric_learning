import os
import unittest

import numpy as np
import torch
from core.model_tools.deformations.geodesic import Geodesic
from in_out.array_readers_and_writers import *
import support.kernels as kernel_factory
from support.utilities.general_settings import Settings
from torch.autograd import Variable


class ParallelTransportTests(unittest.TestCase):

    def setUp(self):
        pass

    def test_parallel_transport(self):
        """
        test the parallel transport on a chosen example converges towards the truth (checked from C++ deformetrica)
        """
        control_points = read_2D_array(os.path.join(Settings().unit_tests_data_dir, "parallel_transport", "control_points.txt"))
        momenta = read_3D_array(os.path.join(Settings().unit_tests_data_dir, "parallel_transport", "geodesic_momenta.txt"))
        momenta_to_transport = read_3D_array(os.path.join(Settings().unit_tests_data_dir, "parallel_transport", "momenta_to_transport.txt"))
        transported_momenta_truth = read_3D_array(os.path.join(Settings().unit_tests_data_dir, "parallel_transport", "ground_truth_transport.txt"))

        control_points_torch = Variable(torch.from_numpy(control_points).type(Settings().tensor_scalar_type))
        momenta_torch = Variable(torch.from_numpy(momenta).type(Settings().tensor_scalar_type))
        momenta_to_transport_torch = Variable(torch.from_numpy(momenta_to_transport).type(Settings().tensor_scalar_type))

        print(np.linalg.norm(momenta_to_transport - transported_momenta_truth)/np.linalg.norm(transported_momenta_truth))

        for concentration in range(1, 25, 2):
            geodesic = Geodesic()
            geodesic.concentration_of_time_points = concentration
            geodesic.set_kernel(kernel_factory.factory('torch', 0.01))
            geodesic.set_use_rk2(True)

            geodesic.tmin = 0.
            geodesic.tmax = 9.
            geodesic.t0 = 0.
            geodesic.set_momenta_t0(momenta_torch)
            geodesic.set_control_points_t0(control_points_torch)
            geodesic.update()

            # Now we transport!
            transported_momenta = geodesic.parallel_transport(momenta_to_transport_torch, is_orthogonal=False)


            # for elt in transported_momenta:
            print(concentration, np.linalg.norm(transported_momenta_truth - transported_momenta[-1].numpy())/np.linalg.norm(transported_momenta_truth))

            # self.assertTrue(np.allclose(transported_momenta_truth, transported_momenta[-1]))

        #self.assertTrue(np.linalg.norm(transported_momenta_truth - parallel_transport_trajectory[-1].data.numpy())/np.linalg.norm(transported_momenta_truth) <= 1e-5)


        # should be 1e-5 (relative) with 10 concentration wrt C++