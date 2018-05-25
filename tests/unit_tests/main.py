#!/usr/bin/env python
# -*- encoding: utf-8 -*-

# import sys
# print(sys.path)

import unittest

from tests.unit_tests.distance_tests import DistanceTests
from tests.unit_tests.array_readers_and_writers_tests import ArrayReadersAndWritersTests
from unit_tests.kernel_factory_tests import KernelFactory, Kernel
from tests.unit_tests.parallel_transport_tests import ParallelTransportTests
from tests.unit_tests.shooting_tests import ShootingTests
from tests.unit_tests.point_cloud_tests import PointCloudTests
from tests.unit_tests.poly_line_tests import PolyLineTests
from tests.unit_tests.surface_mesh_tests import SurfaceMeshTests

TEST_MODULES = [KernelFactory,ParallelTransportTests, Kernel, DistanceTests, ArrayReadersAndWritersTests, PolyLineTests, PointCloudTests, SurfaceMeshTests, ShootingTests]

def main():
    import logging
    logger = logging.getLogger(__name__)
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.DEBUG)

    for t in TEST_MODULES:
        unittest.TextTestRunner(verbosity=2).run(unittest.TestLoader().loadTestsFromTestCase(t))


if __name__ == '__main__':
    main()