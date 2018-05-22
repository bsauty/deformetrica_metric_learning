import unittest

import os
import shutil
import _pickle as pickle

import numpy as np

class DeterministicAtlasSkulls(unittest.TestCase):
    """
    Methods with names starting by "test" will be run.
    """

    def _assertEqual(self, expected, actual):
        if isinstance(expected, dict):
            self.assertTrue(isinstance(actual, dict))
            expected_keys = list(expected.keys())
            actual_keys = list(actual.keys())
            self.assertEqual(expected_keys, actual_keys)
            for key in expected_keys:
                self._assertEqual(expected[key], actual[key])

        elif isinstance(expected, np.ndarray):
            self.assertTrue(isinstance(actual, np.ndarray))
            self.assertTrue(np.array_equal(expected, actual))

        else:
            self.assertEqual(expected, actual)

    def setUp(self):
        pass

    def test_configuration_1(self):
        # Run.
        path_to_deformetrica = os.path.normpath(
            os.path.join(os.path.abspath(__file__), '../../../../../src/deformetrica.py'))
        path_to_model_xml = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model.xml'))
        path_to_data_set_xml = os.path.normpath(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data_set.xml'))
        path_to_optimization_parameters_xml = os.path.normpath(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), 'optimization_parameters.xml'))
        path_to_output = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output'))
        path_to_pydef_state = os.path.join(path_to_output, 'pydef_state.p')
        path_to_log = os.path.join(path_to_output, 'log.txt')
        if os.path.isdir(path_to_output):
            shutil.rmtree(path_to_output)
        os.mkdir(path_to_output)
        cmd = 'python %s %s %s %s --output-dir=%s > %s' % \
              (path_to_deformetrica, path_to_model_xml, path_to_data_set_xml, path_to_optimization_parameters_xml,
               path_to_output, path_to_log)
        os.system(cmd)

        # Load computed and saved results.
        path_to_output_saved = os.path.normpath(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output_saved'))
        assert os.path.isdir(path_to_output_saved), 'No previously saved results: no point of comparison.'
        path_to_pydef_state_saved = os.path.join(path_to_output_saved, 'pydef_state.p')
        pydef_state = pickle.load(open(path_to_pydef_state, 'rb'))
        pydef_state_saved = pickle.load(open(path_to_pydef_state_saved, 'rb'))

        # Assert equality.
        self._assertEqual(pydef_state_saved, pydef_state)








