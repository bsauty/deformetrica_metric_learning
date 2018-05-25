import os
import unittest

import numpy as np
import torch
from core.model_tools.deformations.geodesic import Geodesic
import support.kernels as kernel_factory
from support.utilities.general_settings import Settings
from torch.autograd import Variable


class ShootingTests(unittest.TestCase):

    def setUp(self):
        pass

    def test_geodesic_shooting(self):
        """
        Test the shooting with a single cp. tests with (tmin=t0=0,tmax=1 ; tmin=-1,tmax=t0=0.; tmin=-1,t0=0,tmax=1)
        """
        control_points = np.array([[0.1, 0.2, 0.2]])
        momenta = np.array([[1., 0.2, 0.]])
        momenta_to_transport = np.array([[0.2, 0.3, 0.4]])

        control_points_torch = Variable(torch.from_numpy(control_points).type(Settings().tensor_scalar_type))
        momenta_torch = Variable(torch.from_numpy(momenta).type(Settings().tensor_scalar_type))

        geodesic = Geodesic()
        geodesic.concentration_of_time_points = 10
        geodesic.set_kernel(kernel_factory.factory('torch', 0.01))
        geodesic.set_use_rk2(True)

        geodesic.tmin = 0.
        geodesic.tmax = 1.
        geodesic.t0 = 0.
        geodesic.set_momenta_t0(momenta_torch)
        geodesic.set_control_points_t0(control_points_torch)
        geodesic.update()

        cp_traj = geodesic._get_control_points_trajectory()
        mom_traj = geodesic._get_momenta_trajectory()
        times_traj = geodesic._get_times()

        self.assertTrue(len(cp_traj) == len(mom_traj))
        self.assertTrue(len(times_traj) == len(cp_traj))

        for (cp, mom, time) in zip(cp_traj, mom_traj, times_traj):
            self.assertTrue(np.allclose(cp.detach().numpy(), control_points + time * momenta))
            self.assertTrue(np.allclose(mom.detach().numpy(), momenta))

        geodesic.tmin = -1.
        geodesic.tmax = 0.
        geodesic.t0 = 0.
        geodesic.set_momenta_t0(momenta_torch)
        geodesic.set_control_points_t0(control_points_torch)
        geodesic.update()

        cp_traj = geodesic._get_control_points_trajectory()
        mom_traj = geodesic._get_momenta_trajectory()
        times_traj = geodesic._get_times()

        self.assertTrue(len(cp_traj) == len(mom_traj))
        self.assertTrue(len(times_traj) == len(cp_traj))

        for (cp, mom, time) in zip(cp_traj, mom_traj, times_traj):
            self.assertTrue(np.allclose(cp.detach().numpy(), control_points + time * momenta))
            self.assertTrue(np.allclose(mom.detach().numpy(), momenta))

        geodesic.tmin = -1.
        geodesic.tmax = 1.
        geodesic.t0 = 0.
        geodesic.set_momenta_t0(momenta_torch)
        geodesic.set_control_points_t0(control_points_torch)
        geodesic.update()

        cp_traj = geodesic._get_control_points_trajectory()
        mom_traj = geodesic._get_momenta_trajectory()
        times_traj = geodesic._get_times()

        self.assertTrue(len(cp_traj) == len(mom_traj))
        self.assertTrue(len(times_traj) == len(cp_traj))

        for (cp, mom, time) in zip(cp_traj, mom_traj, times_traj):
            self.assertTrue(np.allclose(cp.detach().numpy(), control_points + time * momenta))
            self.assertTrue(np.allclose(mom.detach().numpy(), momenta))
