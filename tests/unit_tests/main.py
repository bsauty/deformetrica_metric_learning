import unittest

# from tests.unit_tests.distance_tests import DistanceTests
# from tests.unit_tests.in_out_utils_tests import InOutUtilsTests
from tests.unit_tests.kernel_factory_tests import KernelFactory, Kernel

# from tests.unit_tests.parallel_transport_tests import ParallelTransportTests
# from tests.unit_tests.point_cloud_tests import PointCloudTests
# from tests.unit_tests.poly_line_tests import PolyLineTests
# from tests.unit_tests.surface_mesh_tests import SurfaceMeshTests


# TEST_MODULES = [DistanceTests, InOutUtilsTests, KernelFactory, Kernel, ParallelTransportTests, PointCloudTests,
#                 PolyLineTests, SurfaceMeshTests]
TEST_MODULES = [KernelFactory, Kernel]


def main():
    import logging
    logger = logging.getLogger(__name__)
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.DEBUG)

    for t in TEST_MODULES:
        unittest.TextTestRunner(verbosity=2).run(unittest.TestLoader().loadTestsFromTestCase(t))


if __name__ == '__main__':
    main()

