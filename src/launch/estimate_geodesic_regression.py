import math
import os
import time
import warnings

from core.estimators.gradient_ascent import GradientAscent
from core.estimators.scipy_optimize import ScipyOptimize
from core.models.geodesic_regression import GeodesicRegression
from in_out.array_readers_and_writers import *
from in_out.dataset_functions import create_dataset
import support.kernel as kernel_factory


def instantiate_geodesic_regression_model(xml_parameters, dataset=None, ignore_noise_variance=False):
    model = GeodesicRegression()

    # Deformation object -----------------------------------------------------------------------------------------------
    model.geodesic.set_kernel(kernel_factory.factory(xml_parameters.deformation_kernel_type, xml_parameters.deformation_kernel_width))
    model.geodesic.concentration_of_time_points = xml_parameters.concentration_of_time_points
    model.geodesic.t0 = xml_parameters.t0
    model.geodesic.set_use_rk2(xml_parameters.use_rk2)

    # Initial fixed effects --------------------------------------------------------------------------------------------
    # Template.
    model.freeze_template = xml_parameters.freeze_template  # this should happen before the init of the template and the cps
    model.initialize_template_attributes(xml_parameters.template_specifications)
    model.use_sobolev_gradient = xml_parameters.use_sobolev_gradient
    model.smoothing_kernel_width = xml_parameters.deformation_kernel_width * xml_parameters.sobolev_kernel_width_ratio

    # Control points.
    model.freeze_control_points = xml_parameters.freeze_control_points
    if xml_parameters.initial_control_points is not None:
        control_points = read_2D_array(xml_parameters.initial_control_points)
        print(">> Reading " + str(len(control_points)) + " initial control points from file "
              + xml_parameters.initial_control_points)
        model.set_control_points(control_points)
    else:
        model.initial_cp_spacing = xml_parameters.initial_cp_spacing

    # Momenta.
    if xml_parameters.initial_momenta is not None:
        momenta = read_3D_array(xml_parameters.initial_momenta)
        print('>> Reading initial momenta from file: ' + xml_parameters.initial_momenta)
        model.set_momenta(momenta)

    # Final initialization steps by the model object itself ------------------------------------------------------------
    model.update()

    # Special case of the noise variance hyperparameter ----------------------------------------------------------------
    # Compute residuals if needed.
    if not ignore_noise_variance and np.min(model.objects_noise_variance) < 0:

        template_data_torch, control_points_torch, momenta_torch = model._fixed_effects_to_torch_tensors(False)
        target_times = dataset.times[0]
        target_objects = dataset.deformable_objects[0]

        model.geodesic.set_tmin(min(target_times))
        model.geodesic.set_tmax(max(target_times))
        model.geodesic.set_template_data_t0(template_data_torch)
        model.geodesic.set_control_points_t0(control_points_torch)
        model.geodesic.set_momenta_t0(momenta_torch)
        model.geodesic.update()

        residuals = np.zeros((model.number_of_objects,))
        for (time, target) in zip(target_times, target_objects):
            deformed_points = model.geodesic.get_template_data(time)
            residuals += model.multi_object_attachment.compute_distances(
                deformed_points, model.template, target).data.numpy()

        # Initialize the noise variance hyperparameter.
        for k, obj in enumerate(xml_parameters.template_specifications.keys()):
            if model.objects_noise_variance[k] < 0:
                nv = 0.01 * residuals[k] / float(model.number_of_subjects)
                model.objects_noise_variance[k] = nv
                print('>> Automatically chosen noise std: %.4f [ %s ]' % (math.sqrt(nv), obj))

    # Return the initialized model.
    return model


def estimate_geodesic_regression(xml_parameters):
    print('')
    print('[ estimate_geodesic_regression function ]')
    print('')

    """
    Create the dataset object.
    """

    dataset = create_dataset(xml_parameters.dataset_filenames, xml_parameters.visit_ages,
                             xml_parameters.subject_ids, xml_parameters.template_specifications)

    assert dataset.is_time_series(), "Cannot run a geodesic regression on a non-time_series dataset."

    """
    Create the model object.
    """

    model = instantiate_geodesic_regression_model(xml_parameters, dataset)

    """
    Create the estimator object.
    """

    if xml_parameters.optimization_method_type == 'GradientAscent'.lower():
        estimator = GradientAscent()
        estimator.initial_step_size = xml_parameters.initial_step_size
        estimator.scale_initial_step_size = xml_parameters.scale_initial_step_size
        estimator.line_search_shrink = xml_parameters.line_search_shrink
        estimator.line_search_expand = xml_parameters.line_search_expand

    elif xml_parameters.optimization_method_type == 'ScipyLBFGS'.lower():
        estimator = ScipyOptimize()
        estimator.memory_length = xml_parameters.memory_length
        if not model.freeze_template and model.use_sobolev_gradient and estimator.memory_length > 1:
            print('>> Using a Sobolev gradient for the template data with the ScipyLBFGS estimator memory length '
                  'being larger than 1. Beware: that can be tricky.')
            # estimator.memory_length = 1
            # msg = 'Impossible to use a Sobolev gradient for the template data with the ScipyLBFGS estimator memory ' \
            #       'length being larger than 1. Overriding the "memory_length" option, now set to "1".'
            # warnings.warn(msg)

    else:
        estimator = GradientAscent()
        estimator.initial_step_size = xml_parameters.initial_step_size
        estimator.scale_initial_step_size = xml_parameters.scale_initial_step_size
        estimator.line_search_shrink = xml_parameters.line_search_shrink
        estimator.line_search_expand = xml_parameters.line_search_expand

        msg = 'Unknown optimization-method-type: \"' + xml_parameters.optimization_method_type \
              + '\". Defaulting to GradientAscent.'
        warnings.warn(msg)

    estimator.max_iterations = xml_parameters.max_iterations
    estimator.max_line_search_iterations = xml_parameters.max_line_search_iterations
    estimator.convergence_tolerance = xml_parameters.convergence_tolerance

    estimator.print_every_n_iters = xml_parameters.print_every_n_iters
    estimator.save_every_n_iters = xml_parameters.save_every_n_iters

    estimator.dataset = dataset
    estimator.statistical_model = model

    """
    Launch.
    """

    if not os.path.exists(Settings().output_dir):
        os.makedirs(Settings().output_dir)

    model.name = 'GeodesicRegression'
    print('')
    print('[ update method of the ' + estimator.name + ' optimizer ]')

    start_time = time.time()
    estimator.update()
    estimator.write()
    end_time = time.time()
    print('>> Estimation took: ' + str(time.strftime("%H:%M:%S", time.gmtime(end_time - start_time))))

    return model
