import os
import time

from core import default
from launch.estimate_deterministic_atlas import instantiate_deterministic_atlas_model
from launch.estimate_geodesic_regression import instantiate_geodesic_regression_model


class Deformetrica:
    def __init__(self, output_dir=default.output_dir):
        self.output_dir = output_dir

        # create output dir if it does not already exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def estimate_deterministic_atlas(self, template_specifications, dataset, estimator, estimator_options={}, write_output=True, **kwargs):
        """
        TODO
        :param template_specifications:
        :param dataset:
        :param estimator:
        :param estimator_options:
        :param kwargs:
        :return:
        """
        statistical_model = instantiate_deterministic_atlas_model(dataset, template_specifications, **kwargs)

        # sanitize estimator_options
        if 'output_dir' in estimator_options:
            raise RuntimeError('estimator_options cannot contain output_dir key')

        # instantiate estimator
        estimator = estimator(statistical_model, output_dir=self.output_dir, **estimator_options)

        """
        Launch
        """
        self.__launch_estimator(estimator, write_output)

        return statistical_model

    def estimate_bayesian_atlas(self):
        """
        TODO
        :return:
        """
        raise RuntimeError('not implemented.')

    def estimate_longitudinal_atlas(self):
        """
        TODO
        :return:
        """
        raise RuntimeError('not implemented.')

    def estimate_rigid_atlas(self):
        """
        TODO
        :return:
        """
        raise RuntimeError('not implemented.')

    def estimate_longitudinal_metric_model(self):
        """
        TODO
        :return:
        """
        raise RuntimeError('not implemented.')

    def estimate_longitudinal_metric_registration(self):
        """
        TODO
        :return:
        """
        raise RuntimeError('not implemented.')

    def estimate_longitudinal_registration(self):
        """
        TODO
        :return:
        """
        raise RuntimeError('not implemented.')

    def estimate_geodesic_regression(self, template_specifications, dataset, estimator, estimator_options={}, write_output=True, **kwargs):
        """
        TODO
        :param template_specifications:
        :param dataset:
        :param estimator:
        :param estimator_options:
        :param kwargs:
        :return:
        """
        statistical_model = instantiate_geodesic_regression_model(dataset, template_specifications, **kwargs)

        # sanitize estimator_options
        if 'output_dir' in estimator_options:
            raise RuntimeError('estimator_options cannot contain output_dir key')

        # instantiate estimator
        estimator = estimator(statistical_model, output_dir=self.output_dir, **estimator_options)

        """
        Launch
        """
        self.__launch_estimator(estimator, write_output)

        return statistical_model

    def estimate_deep_pga(self):
        """
        TODO
        :return:
        """
        raise RuntimeError('not implemented.')

    def compute_parallel_transport(self):
        """
        TODO
        :return:
        """
        raise RuntimeError('not implemented.')

    def compute_shooting(self):
        """
        TODO
        :return:
        """
        raise RuntimeError('not implemented.')

    @staticmethod
    def __launch_estimator(estimator, write_output=True):
        """
        Launch the estimator. This will iterate until a stop condition is reached.

        :param estimator:   Estimator that is to be used.
                            eg: :class:`GradientAscent <core.estimators.gradient_ascent.GradientAscent>`, :class:`ScipyOptimize <core.estimators.scipy_optimize.ScipyOptimize>`
        """
        print('[ update method of the ' + estimator.name + ' optimizer ]')
        start_time = time.time()
        estimator.update()
        end_time = time.time()

        if write_output:
            estimator.write()

        print('>> Estimation took: ' + str(time.strftime("%H:%M:%S", time.gmtime(end_time - start_time))))
