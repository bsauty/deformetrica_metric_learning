import os

from functional_tests.functional_test import FunctionalTest


class DeterministicAtlasSkulls(FunctionalTest):
    """
    Methods with names starting by "test" will be run.
    """

    def test_configuration_1(self):
        self.run_configuration(os.path.abspath(__file__), 'output__1', 'output_saved__1',
                               'model.xml', 'data_set.xml', 'optimization_parameters__1.xml')

    def test_configuration_2(self):
        self.run_configuration(os.path.abspath(__file__), 'output__2', 'output_saved__2',
                               'model.xml', 'data_set.xml', 'optimization_parameters__2.xml')

    def test_configuration_3(self):
        self.run_configuration(os.path.abspath(__file__), 'output__3', 'output_saved__3',
                               'model.xml', 'data_set.xml', 'optimization_parameters__3.xml')

