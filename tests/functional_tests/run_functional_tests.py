#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import os
import unittest

from functional_tests.data.deterministic_atlas.skulls.run import DeterministicAtlasSkulls
from functional_tests.data.deterministic_atlas.brain_structures.run import DeterministicAtlasBrainStructures
from functional_tests.data.geodesic_regression.skulls.run import GeodesicRegressionSkulls

TEST_MODULES = [DeterministicAtlasSkulls, DeterministicAtlasBrainStructures, GeodesicRegressionSkulls]


def setup_conda_env():
    path_to_environment_file = os.path.normpath(
            os.path.join(os.path.abspath(__file__), '../../../environment.yml'))
    cmd = 'hostname && . ~/.profile && conda env create -f %s' % path_to_environment_file
    os.system(cmd)


def main():
    import logging
    logger = logging.getLogger(__name__)
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.DEBUG)

    setup_conda_env()

    for t in TEST_MODULES:
        unittest.TextTestRunner(verbosity=2).run(unittest.TestLoader().loadTestsFromTestCase(t))


if __name__ == '__main__':
    main()

