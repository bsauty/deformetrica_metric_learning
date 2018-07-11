import logging
import os
import time

import torch

from core import default
from in_out.array_readers_and_writers import read_2D_array, read_3D_array
from launch.compute_parallel_transport import compute_parallel_transport
from launch.compute_shooting import run_shooting
from launch.estimate_bayesian_atlas import instantiate_bayesian_atlas_model
from launch.estimate_deterministic_atlas import instantiate_deterministic_atlas_model
from launch.estimate_geodesic_regression import instantiate_geodesic_regression_model

logger = logging.getLogger(__name__)


class Deformetrica:
    def __init__(self, output_dir=default.output_dir):
        self.output_dir = output_dir

        # create output dir if it does not already exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        logger.debug('Deformetrica.__exit__()')
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def estimate_deterministic_atlas(self, template_specifications, dataset, estimator, estimator_options={}, write_output=True, **kwargs):
        """
        Estimate deterministic atlas
        """
        statistical_model = instantiate_deterministic_atlas_model(dataset, template_specifications, **kwargs)

        # sanitize estimator_options
        if 'output_dir' in estimator_options:
            raise RuntimeError('estimator_options cannot contain output_dir key')

        # instantiate estimator
        estimator = estimator(statistical_model, dataset, output_dir=self.output_dir, **estimator_options)

        """
        Launch
        """
        self.__launch_estimator(estimator, write_output)

        return statistical_model

    def estimate_bayesian_atlas(self, template_specifications, dataset, estimator, estimator_options={}, write_output=True, **kwargs):
        """
        Estimate bayesian atlas
        """
        statistical_model, individual_RER = instantiate_bayesian_atlas_model(dataset, template_specifications, **kwargs)

        # sanitize estimator_options
        if 'output_dir' in estimator_options:
            raise RuntimeError('estimator_options cannot contain output_dir key')

        # instantiate estimator
        estimator = estimator(statistical_model, dataset, output_dir=self.output_dir, individual_RER=individual_RER, **estimator_options)

        """
        Launch
        """
        self.__launch_estimator(estimator, write_output)

        return statistical_model

    def estimate_longitudinal_atlas(self):
        """
        TODO
        :return:
        """
        raise NotImplementedError

    def estimate_rigid_atlas(self):
        """
        TODO
        :return:
        """
        raise NotImplementedError

    def estimate_longitudinal_metric_model(self):
        """
        TODO
        :return:
        """
        raise NotImplementedError

    def estimate_longitudinal_metric_registration(self):
        """
        TODO
        :return:
        """
        raise NotImplementedError

    def estimate_longitudinal_registration(self):
        """
        TODO
        :return:
        """
        raise NotImplementedError

    def estimate_geodesic_regression(self, template_specifications, dataset, estimator, estimator_options={}, write_output=True, **kwargs):
        """
        estimate geodesic regression
        """
        statistical_model = instantiate_geodesic_regression_model(dataset, template_specifications, **kwargs)

        # sanitize estimator_options
        if 'output_dir' in estimator_options:
            raise RuntimeError('estimator_options cannot contain output_dir key')

        # instantiate estimator
        estimator = estimator(statistical_model, dataset, output_dir=self.output_dir, **estimator_options)

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
        raise NotImplementedError

    def compute_parallel_transport(self, template_specifications, dataset,
                                   initial_control_points, initial_momenta, initial_momenta_to_transport,
                                   deformation_kernel,
                                   initial_control_points_to_transport=default.initial_control_points_to_transport,
                                   **kwargs):
        """
        Compute parallel transport
        """
        if initial_control_points is None:
            raise RuntimeError("Please provide initial control points")
        if initial_momenta is None:
            raise RuntimeError("Please provide initial momenta")
        if initial_momenta_to_transport is None:
            raise RuntimeError("Please provide initial momenta to transport")

        control_points = read_2D_array(initial_control_points)
        initial_momenta = read_3D_array(initial_momenta)
        initial_momenta_to_transport = read_3D_array(initial_momenta_to_transport)

        if initial_control_points_to_transport is None:
            logger.warning("initial-control-points-to-transport was not specified, I am assuming they are the same as initial-control-points")
            control_points_to_transport = control_points
            need_to_project_initial_momenta = False
        else:
            control_points_to_transport = read_2D_array(initial_control_points_to_transport)
            need_to_project_initial_momenta = True

        control_points_torch = torch.from_numpy(control_points).type(dataset.tensor_scalar_type)
        initial_momenta_torch = torch.from_numpy(initial_momenta).type(dataset.tensor_scalar_type)
        initial_momenta_to_transport_torch = torch.from_numpy(initial_momenta_to_transport).type(dataset.tensor_scalar_type)

        # We start by projecting the initial momenta if they are not carried at the reference progression control points.
        if need_to_project_initial_momenta:
            control_points_to_transport_torch = torch.from_numpy(control_points_to_transport).type(dataset.tensor_scalar_type)
            velocity = deformation_kernel.convolve(control_points_torch, control_points_to_transport_torch, initial_momenta_to_transport_torch)
            kernel_matrix = deformation_kernel.get_kernel_matrix(control_points_torch)
            cholesky_kernel_matrix = torch.potrf(kernel_matrix)
            # cholesky_kernel_matrix = torch.Tensor(np.linalg.cholesky(kernel_matrix.data.numpy()).type_as(kernel_matrix))#Dirty fix if pytorch fails.
            projected_momenta = torch.potrs(velocity, cholesky_kernel_matrix).squeeze().contiguous()

        else:
            projected_momenta = initial_momenta_to_transport_torch

        compute_parallel_transport(control_points_torch, initial_momenta_torch, projected_momenta, template_specifications,
                                   deformation_kernel=deformation_kernel,
                                   dimension=dataset.dimension,
                                   tensor_scalar_type=dataset.tensor_scalar_type,
                                   output_dir=self.output_dir, **kwargs)

    def compute_shooting(self, template_specifications, dataset, deformation_kernel, **kwargs):
        """
        Compute shooting
        """
        # sanitize estimator_options
        if 'output_dir' in kwargs:
            raise RuntimeError('estimator_options cannot contain output_dir key')

        run_shooting(template_specifications, dataset, deformation_kernel, output_dir=self.output_dir, **kwargs)

    @staticmethod
    def __launch_estimator(estimator, write_output=True):
        """
        Launch the estimator. This will iterate until a stop condition is reached.

        :param estimator:   Estimator that is to be used.
                            eg: :class:`GradientAscent <core.estimators.gradient_ascent.GradientAscent>`, :class:`ScipyOptimize <core.estimators.scipy_optimize.ScipyOptimize>`
        """
        start_time = time.time()
        logger.info('Estimator started ' + estimator.name + ' at ' + str(start_time))
        estimator.update()
        end_time = time.time()

        if write_output:
            estimator.write()

        logger.info('Estimation took: ' + str(time.strftime("%H:%M:%S", time.gmtime(end_time - start_time))))
