import os

from functional_tests.functional_test import FunctionalTest


class RegressionSkulls(FunctionalTest):
    """
    Methods with names starting by "test" will be run.
    """

    def test_configuration_1(self):
        self.run_configuration(os.path.abspath(__file__), 'output__1', 'output_saved__1',
                               'model.xml', 'data_set.xml', 'optimization_parameters.xml')


