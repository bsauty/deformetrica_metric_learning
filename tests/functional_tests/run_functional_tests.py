#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import os
import unittest

from functional_tests.data.atlas.skulls.run import AtlasSkulls
from functional_tests.data.atlas.brain_structures.run import AtlasBrainStructures
from functional_tests.data.atlas.digits.run import AtlasDigits
from functional_tests.data.regression.skulls.run import RegressionSkulls

TEST_MODULES = [AtlasSkulls, AtlasBrainStructures, AtlasDigits, RegressionSkulls]


def setup_conda_env():
    path_to_environment_file = os.path.normpath(
            os.path.join(os.path.abspath(__file__), '../../../environment.yml'))
    cmd = 'hostname && ' \
          'if [ -f "~/.profile" ]; then . ~/.profile; ' \
          'fi && conda env create -f %s' % path_to_environment_file
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

