import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../../../')
import numpy as np
from pydeformetrica.src.support.utilities.general_settings import Settings
from pydeformetrica.src.in_out.utils import *
import unittest
import tempfile

class ExponentialTest(unittest.TestCase):
    """
    Methods with names starting by "test" will be run
    """
    def setUp(self):
        pass

    def assert_shoot_is_accurate(self):
        pass

    def assert_flow_is_accurate(self):
        pass

    def assert_get_norm_squared_is_accurate(self):
        pass

    def assert_get_norm_squared_raises_exception_when_shooting_is_modified(self):
        pass

    