import os.path
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../../../../../')

from pydeformetrica.src.core.model_tools.manifolds.exponential_interface import ExponentialInterface

import torch

"""
Exponential on \R for 1/(q**2(1-q)) metric i.e. logistic curves.
"""

class LogisticExponential(ExponentialInterface):

    def __init__(self):
        # Mother class constructor
        ExponentialInterface.__init__(self)
        self.has_closed_form = True

    def inverse_metric(self, q):
        return (q*(1-q))**2

    def closed_form(self, q, v, t):
        return 1./(1 + (1/q - 1) * torch.exp(-1.*v/(q * (1-q)) * t))

