import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../../../')
import numpy as np
from pydeformetrica.src.support.utilities.general_settings import Settings
from pydeformetrica.src.in_out.utils import *
import unittest
import tempfile

class InOutUtilsTests(unittest.TestCase):
    """
    Methods with names starting by "test" will be run
    """
    def setUp(self):
        pass

    def test_read_momenta(self):
        momenta = read_momenta(os.path.join(Settings().UnitTestsDataDir, "Momenta.txt"))
        self.assertEqual(momenta.shape, (4,72,3))
        self.assertTrue(np.allclose(momenta[0,0], np.array([-0.0313538, -0.00373486, -0.0256917])))
        self.assertTrue(np.allclose(momenta[0, -1], np.array([-0.518624, 1.47211, 0.880905])))
        self.assertTrue(np.allclose(momenta[-1, -1], np.array([2.81286, -0.353167, -2.16408])))

    def test_write_momenta(self):
        momenta = read_momenta(os.path.join(Settings().UnitTestsDataDir, "Momenta.txt"))
        write_momenta(momenta, os.path.join(tempfile.gettempdir(), "test_write_momenta.txt"))
        read = read_momenta(os.path.join(tempfile.gettempdir(), "test_write_momenta.txt"))
        self.assertTrue(np.allclose(momenta, read))