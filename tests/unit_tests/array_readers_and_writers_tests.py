import os
import tempfile
import unittest

import numpy as np
from in_out.array_readers_and_writers import *
from support.utilities.general_settings import Settings


class ArrayReadersAndWritersTests(unittest.TestCase):
    """
    Methods with names starting by "test" will be run
    """
    def setUp(self):
        pass

    def test_read_3D_array(self):
        momenta = read_3D_array(os.path.join(Settings().unit_tests_data_dir, "Momenta.txt"))
        self.assertEqual(momenta.shape, (4,72,3))
        self.assertTrue(np.allclose(momenta[0,0], np.array([-0.0313538, -0.00373486, -0.0256917])))
        self.assertTrue(np.allclose(momenta[0, -1], np.array([-0.518624, 1.47211, 0.880905])))
        self.assertTrue(np.allclose(momenta[-1, -1], np.array([2.81286, -0.353167, -2.16408])))

    def test_write_3D_array(self):
        momenta = read_3D_array(os.path.join(Settings().unit_tests_data_dir, "Momenta.txt"))
        write_3D_array(momenta, os.path.join(tempfile.gettempdir(), "test_write_3D_array.txt"))
        read = read_3D_array(os.path.join(tempfile.gettempdir(), "test_write_3D_array.txt"))
        self.assertTrue(np.allclose(momenta, read))
