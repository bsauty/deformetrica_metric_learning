import os
import unittest
import shutil
import numpy as np
import _pickle as pickle


class FunctionalTest(unittest.TestCase):
    def assertStateEqual(self, expected, actual):
        if isinstance(expected, dict):
            self.assertTrue(isinstance(actual, dict))
            expected_keys = list(expected.keys())
            actual_keys = list(actual.keys())
            self.assertEqual(expected_keys, actual_keys)
            for key in expected_keys:
                self.assertStateEqual(expected[key], actual[key])

        elif isinstance(expected, np.ndarray):
            self.assertTrue(isinstance(actual, np.ndarray))
            self.assertTrue(np.allclose(expected, actual, atol=1e-5))

        else:
            self.assertEqual(expected, actual)

    def run_configuration(self, path_to_test, output_folder, output_saved_folder,
                          model_xml, data_set_xml, optimization_parameters_xml):
        # Run.
        path_to_deformetrica = os.path.normpath(
            os.path.join(path_to_test, '../../../../../src/deformetrica.py'))
        path_to_model_xml = os.path.normpath(os.path.join(os.path.dirname(path_to_test), model_xml))
        path_to_data_set_xml = os.path.normpath(
            os.path.join(os.path.dirname(path_to_test), data_set_xml))
        path_to_optimization_parameters_xml = os.path.normpath(
            os.path.join(os.path.dirname(path_to_test), optimization_parameters_xml))
        path_to_output = os.path.normpath(os.path.join(os.path.dirname(path_to_test), output_folder))
        path_to_pydef_state = os.path.join(path_to_output, 'pydef_state.p')
        path_to_log = os.path.join(path_to_output, 'log.txt')
        if os.path.isdir(path_to_output):
            shutil.rmtree(path_to_output)
        os.mkdir(path_to_output)
        cmd = 'python %s %s %s %s --output=%s > %s' % \
              (path_to_deformetrica, path_to_model_xml, path_to_data_set_xml,
               path_to_optimization_parameters_xml, path_to_output, path_to_log)
        os.system(cmd)

        # Load computed and saved results.
        path_to_output_saved = os.path.normpath(
            os.path.join(os.path.dirname(path_to_test), output_saved_folder))
        assert os.path.isdir(path_to_output_saved), 'No previously saved results: no point of comparison.'
        path_to_pydef_state_saved = os.path.join(path_to_output_saved, 'pydef_state.p')

        # open pickle file and compare
        with open(path_to_pydef_state, 'rb') as pydef_state_file, \
                open(path_to_pydef_state_saved, 'rb') as pydef_state_saved_file:
            pydef_state = pickle.load(pydef_state_file)
            pydef_state_saved = pickle.load(pydef_state_saved_file)

            # Assert equality.
            self.assertStateEqual(pydef_state_saved, pydef_state)