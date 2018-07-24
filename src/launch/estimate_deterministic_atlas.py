import math

from core import default
from core.models.deterministic_atlas import DeterministicAtlas
from in_out.array_readers_and_writers import *


def instantiate_deterministic_atlas_model(dataset, template_specifications,
                                          deformation_kernel_type=default.deformation_kernel_type,
                                          deformation_kernel_width=default.deformation_kernel_width,
                                          shoot_kernel_type=None,
                                          number_of_time_points=default.number_of_time_points,
                                          use_rk2_for_shoot=default.use_rk2_for_shoot,
                                          use_rk2_for_flow=default.use_rk2_for_flow,
                                          freeze_template=default.freeze_template,
                                          freeze_control_points=default.freeze_control_points,
                                          use_sobolev_gradient=default.use_sobolev_gradient,
                                          smoothing_kernel_width=default.smoothing_kernel_width,
                                          initial_control_points=default.initial_control_points,
                                          initial_cp_spacing=default.initial_cp_spacing,
                                          initial_momenta=default.initial_momenta,
                                          ignore_noise_variance=False, dense_mode=default.dense_mode,
                                          number_of_threads=default.number_of_threads,
                                          **kwargs):

    model = DeterministicAtlas(
        template_specifications, dataset.dimension, (dataset.tensor_scalar_type, dataset.tensor_integer_type),
        deformation_kernel_type=deformation_kernel_type, deformation_kernel_width=deformation_kernel_width,
        shoot_kernel_type=shoot_kernel_type,
        number_of_time_points=number_of_time_points,
        use_rk2_for_shoot=use_rk2_for_shoot, use_rk2_for_flow=use_rk2_for_flow,
        freeze_template=freeze_template, freeze_control_points=freeze_control_points,
        use_sobolev_gradient=use_sobolev_gradient, smoothing_kernel_width=smoothing_kernel_width,
        dense_mode=dense_mode,
        number_of_threads=number_of_threads)

    # Control points.
    if initial_control_points is not None:
        control_points = read_2D_array(initial_control_points)
        print(">> Reading " + str(len(control_points)) + " initial control points from file " + initial_control_points)
        model.set_control_points(control_points)
    else:
        model.initial_cp_spacing = initial_cp_spacing

    # Momenta.
    if initial_momenta is not None:
        momenta = read_3D_array(initial_momenta)
        print('>> Reading %d initial momenta from file: %s' % (momenta.shape[0], initial_momenta))
        model.set_momenta(momenta)
        model.number_of_subjects = momenta.shape[0]
    else:
        model.number_of_subjects = len(dataset.dataset_filenames)

    # Final initialization steps by the model object itself ------------------------------------------------------------
    model.update()

    # Special case of the noise variance hyperparameter ----------------------------------------------------------------
    # Compute residuals if needed.
    if not ignore_noise_variance and np.min(model.objects_noise_variance) < 0:
        template_data_torch, template_points_torch, control_points_torch, momenta_torch \
            = model._fixed_effects_to_torch_tensors(False)
        targets = dataset.deformable_objects
        targets = [target[0] for target in targets]

        residuals_torch = []
        model.exponential.set_initial_template_points(template_points_torch)
        model.exponential.set_initial_control_points(control_points_torch)
        for i, target in enumerate(targets):
            model.exponential.set_initial_momenta(momenta_torch[i])
            model.exponential.update()
            deformed_points = model.exponential.get_template_points()
            deformed_data = model.template.get_deformed_data(deformed_points, template_data_torch)
            residuals_torch.append(model.multi_object_attachment.compute_distances(
                deformed_data, model.template, target))

        residuals = np.zeros((model.number_of_objects,))
        for i in range(len(residuals_torch)):
            residuals += residuals_torch[i].data.numpy()

        # Initialize the noise variance hyperparameter.
        for k, obj in enumerate(template_specifications.keys()):
            if model.objects_noise_variance[k] < 0:
                nv = 0.01 * residuals[k] / float(model.number_of_subjects)
                model.objects_noise_variance[k] = nv
                print('>> Automatically chosen noise std: %.4f [ %s ]' % (math.sqrt(nv), obj))

    # Return the initialized model.
    return model
