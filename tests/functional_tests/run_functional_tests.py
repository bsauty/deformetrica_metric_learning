#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import unittest

from tests.functional_tests.deterministic_atlas.skulls.run import DeterministicAtlasSkulls

TEST_MODULES = [DeterministicAtlasSkulls]


def main():
    import logging
    logger = logging.getLogger(__name__)
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.DEBUG)

    for t in TEST_MODULES:
        unittest.TextTestRunner(verbosity=2).run(unittest.TestLoader().loadTestsFromTestCase(t))


if __name__ == '__main__':
    main()

