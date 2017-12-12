import sys
import os
import unittest
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../../../')

from pydeformetrica.tests.unit_tests.landmark_tests import LandmarkTests

suite = unittest.TestLoader().loadTestsFromTestCase(LandmarkTests)
unittest.TextTestRunner(verbosity=2).run(suite)