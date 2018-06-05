#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import unittest
import sys

from unit_tests.test_attachments import DistanceTests
from unit_tests.test_array_readers_and_writers import ArrayReadersAndWritersTests
from unit_tests.test_kernel_factory import KernelFactory, Kernel, KeopsVersusCuda
from unit_tests.test_parallel_transport import ParallelTransportTests
from unit_tests.test_shooting import ShootingTests
from unit_tests.test_point_cloud import PointCloudTests
from unit_tests.test_poly_line import PolyLineTests
from unit_tests.test_surface_mesh import SurfaceMeshTests

TEST_MODULES = [KernelFactory, Kernel, KeopsVersusCuda,
                ParallelTransportTests, DistanceTests, ArrayReadersAndWritersTests,
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
