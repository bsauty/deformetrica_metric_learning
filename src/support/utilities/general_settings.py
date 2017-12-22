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
        self.dimension = 3
        self.output_dir = "output"

        #Wether or not to use the state file to resume the computation
        self.load_state = False
        #Default path to state file
        self.state_file = os.path.join(self.output_dir, "pydef_state.p")

        self.tensor_scalar_type = torch.DoubleTensor
        self.tensor_integer_type = torch.LongTensor

        pydeformetrica_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        self.unit_tests_data_dir = os.path.join(pydeformetrica_root, "tests", "unit_tests", "data")

def Settings():
    return GeneralSettings.Instance()