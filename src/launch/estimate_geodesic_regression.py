import math

from core import default
from core.models.geodesic_regression import GeodesicRegression
from in_out.array_readers_and_writers import *


def instantiate_geodesic_regression_model(dataset, template_specifications,
                                          deformation_kernel_type=default.deformation_kernel_type,
                                          deformation_kernel_width=default.deformation_kernel_width,
                                          shoot_kernel_type=None,
                                          concentration_of_time_points=default.concentration_of_time_points,
                                          t0=None,
                                          use_rk2_for_shoot=default.use_rk2_for_shoot,
                                          use_rk2_for_flow=default.use_rk2_for_flow,
                                          initial_cp_spacing=default.initial_cp_spacing,
                                          freeze_template=default.freeze_template,
                                          freeze_control_points=default.freeze_control_points,
                                          use_sobolev_gradient=default.use_sobolev_gradient,
                                          smoothing_kernel_width=default.smoothing_kernel_width,
                                          initial_control_points=default.initial_control_points,
                                          initial_momenta=default.initial_momenta,
                                          ignore_noise_variance=False, dense_mode=default.dense_mode,
                                          number_of_threads=default.number_of_threads,
                                          **kwargs
                                          ):
    model = GeodesicRegression(
        template_specifications, dataset.dimension, (dataset.tensor_scalar_type, dataset.tensor_integer_type),
        deformation_kernel_type=deformation_kernel_type, deformation_kernel_width=deformation_kernel_width,
        shoot_kernel_type=shoot_kernel_type,
        concentration_of_time_points=concentration_of_time_points, t0=t0,
        use_rk2_for_shoot=use_rk2_for_shoot, use_rk2_for_flow=use_rk2_for_flow,
        initial_cp_spacing=initial_cp_spacing,
        freeze_template=freeze_template,
        freeze_control_points=freeze_control_points,
        use_sobolev_gradient=use_sobolev_gradient,
        smoothing_kernel_width=smoothing_kernel_width,
        dense_mode=dense_mode,
        number_of_threads=number_of_threads)

    # Control points.
    # model.freeze_control_points = freeze_control_points
    if initial_control_points is not None:
        control_points = read_2D_array(initial_control_points)
        print(">> Reading " + str(len(control_points)) + " initial control points from file " + initial_control_points)
        model.set_control_points(control_points)
    else:
        model.initial_cp_spacing = initial_cp_spacing

    # Momenta.
    if initial_momenta is not None:
        momenta = read_3D_array(initial_momenta)
        print('>> Reading initial momenta from file: ' + initial_momenta)
        model.set_momenta(momenta)

    # Final initialization steps by the model object itself ------------------------------------------------------------
    model.update()

    # Special case of the noise variance hyperparameter ----------------------------------------------------------------
    # Compute residuals if needed.
    if not ignore_noise_variance and np.min(model.objects_noise_variance) < 0:

        template_data_torch, template_points_torch, control_points_torch, momenta_torch \
            = model._fixed_effects_to_torch_tensors(False)
        target_times = dataset.times[0]
        target_objects = dataset.deformable_objects[0]

        model.geodesic.set_tmin(min(target_times))
        model.geodesic.set_tmax(max(target_times))
        model.geodesic.set_template_points_t0(template_points_torch)
        model.geodesic.set_control_points_t0(control_points_torch)
        model.geodesic.set_momenta_t0(momenta_torch)
        model.geodesic.update()

        residuals = np.zeros((model.number_of_objects,))
        for (time, target) in zip(target_times, target_objects):
            deformed_points = model.geodesic.get_template_points(time)
            deformed_data = model.template.get_deformed_data(deformed_points, template_data_torch)
            residuals += model.multi_object_attachment.compute_distances(
                deformed_data, model.template, target).data.numpy()

        # Initialize the noise variance hyperparameter.
        for k, obj in enumerate(template_specifications.keys()):
            if model.objects_noise_variance[k] < 0:
                nv = 0.01 * residuals[k] / float(len(target_times))
                model.objects_noise_variance[k] = nv
                print('>> Automatically chosen noise std: %.4f [ %s ]' % (math.sqrt(nv), obj))

    # Return the initialized model.
    return model
