#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import unittest
import sys

from unit_tests.distance_tests import DistanceTests
from unit_tests.array_readers_and_writers_tests import ArrayReadersAndWritersTests
from unit_tests.kernel_factory_tests import KernelFactory, Kernel

from unit_tests.parallel_transport_tests import ParallelTransportTests
from unit_tests.shooting_tests import ShootingTests
from unit_tests.point_cloud_tests import PointCloudTests
from unit_tests.poly_line_tests import PolyLineTests
from unit_tests.surface_mesh_tests import SurfaceMeshTests

TEST_MODULES = [KernelFactory, Kernel, ParallelTransportTests, DistanceTests, ArrayReadersAndWritersTests,
                PolyLineTests, PointCloudTests, SurfaceMeshTests, ShootingTests]

def main():
    import logging
    logger = logging.getLogger(__name__)
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.DEBUG)

    success = True

    for t in TEST_MODULES:
        res = unittest.TextTestRunner(verbosity=2).run(unittest.TestLoader().loadTestsFromTestCase(t))
        success = success and res.wasSuccessful()

    print(success)
    if not success:
        sys.exit('Test failure !')

if __name__ == '__main__':
    main()
