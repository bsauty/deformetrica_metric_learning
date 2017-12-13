import os.path
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../../../')

import torch

from pydeformetrica.src.support.utilities.singleton_pattern import Singleton

@Singleton
class GeneralSettings:

    """
    General settings, shared across the whole code.
    Singleton pattern.

    """

    def __init__(self):
        self.Dimension = 3
        self.OutputDir = "output"

        self.TensorScalarType = torch.DoubleTensor
        self.TensorIntegerType = torch.LongTensor

        pydeformetrica_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        self.UnitTestsDataDir = os.path.join(pydeformetrica_root, "tests", "unit_tests", "data")

def Settings():
    return GeneralSettings.Instance()