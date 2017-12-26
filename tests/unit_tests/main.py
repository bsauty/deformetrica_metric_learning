import sys
import os
import unittest
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../../../')

from pydeformetrica.tests.unit_tests.surface_mesh_tests import SurfaceMeshTests
from pydeformetrica.tests.unit_tests.poly_line_tests import PolyLineTests
from pydeformetrica.tests.unit_tests.point_cloud_tests import PointCloudTests
from pydeformetrica.tests.unit_tests.distance_tests import DistanceTests
from pydeformetrica.tests.unit_tests.in_out_utils_tests import InOutUtilsTests
from pydeformetrica.tests.unit_tests.parallel_transport_tests import ParallelTransportTests


suite = unittest.TestLoader().loadTestsFromTestCase(SurfaceMeshTests)
unittest.TextTestRunner(verbosity=2).run(suite)

suite = unittest.TestLoader().loadTestsFromTestCase(InOutUtilsTests)
unittest.TextTestRunner(verbosity=2).run(suite)

suite = unittest.TestLoader().loadTestsFromTestCase(PolyLineTests)
unittest.TextTestRunner(verbosity=2).run(suite)

suite = unittest.TestLoader().loadTestsFromTestCase(PointCloudTests)
unittest.TextTestRunner(verbosity=2).run(suite)

suite = unittest.TestLoader().loadTestsFromTestCase(DistanceTests)
unittest.TextTestRunner(verbosity=2).run(suite)

suite = unittest.TestLoader().loadTestsFromTestCase(ParallelTransportTests)
unittest.TextTestRunner(verbosity=2).run(suite)
