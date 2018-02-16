import os.path
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../../../../../')

import torch
from torch.autograd import Variable
import numpy as np
import warnings

from pydeformetrica.src.in_out.utils import *
from pydeformetrica.src.support.utilities.general_settings import Settings


# Generic object to be used to integrate geodesics on a given manifold.

class ManifoldCalculator:

    def __init__(self):
        pass

    def _dp(self, h, q):
        """
        if dp is not given on the manifold, we get it using automatic differentiation (more expensive of course)
        """
        return torch.autograd.grad(h, q, create_graph=True, retain_graph=True)[0]

    def _euler_step(self, q, p, dt, inverse_metric):
        d_q = dt * torch.matmul(inverse_metric(q), p)
        H = self.hamiltonian(q, p, inverse_metric)
        d_p = -1. * dt * dp(H, q)
        return q + d_q, p + d_p

    def _rk2_step(self, q, p, dt, inverse_metric, dp=None):
        if dp is None:
            # Intermediate step
            h1 = self.hamiltonian(q, p, inverse_metric)
            mid_q = q + 0.5 * dt * torch.matmul(inverse_metric(q), p)
            mid_p = p - 0.5 * dt * self._dp(h1, q)

            # Final step
            h2 = self.hamiltonian(mid_q, mid_p, inverse_metric)
            return q + dt * torch.matmul(inverse_metric(mid_q), mid_p), p - dt * self._dp(h2, mid_q)
        else:
            mid_q = q + 0.5 * dt * torch.matmul(inverse_metric(q), p)
            mid_p = p - 0.5 * dt * dp(q, p)
            return q + dt * torch.matmul(inverse_metric(mid_q), mid_p), p - dt * dp(q, p)

    def hamiltonian(self, q, p, inverse_metric):
        return torch.dot(p, torch.matmul(inverse_metric(q), p)) * 0.5

    def exponential(self, q, p, nb_steps=10, closed_form=None, inverse_metric=None, dp=None):
        """
        Use the given inverse_metric to compute the Hamiltonian equations.
        OR a given closed-form expression for the geodesic.
        """
        if closed_form is None and inverse_metric is None:
            raise ValueError('Inverse metric or closed_form must be provided to the manifold calculator.')
        q.requires_grad = True
        traj_q, traj_p = [], []
        traj_q.append(q)
        traj_p.append(p)
        dt = 1. / float(nb_steps)
        times = np.linspace(dt, 1., nb_steps-1)

        if closed_form is None:
            for _ in times:
                new_q, new_p = self._rk2_step(traj_q[-1], traj_p[-1], dt, inverse_metric, dp)
                traj_q.append(new_q)
                traj_p.append(new_p)

        else:
            for t in times:
                traj_q.append(closed_form(q, p, t))

        return traj_q