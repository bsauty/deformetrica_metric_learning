from core import default
from core.models.bayesian_atlas import BayesianAtlas
from in_out.array_readers_and_writers import *


def instantiate_bayesian_atlas_model(dataset, template_specifications,
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
                                     dense_mode=default.dense_mode,
                                     number_of_threads=default.number_of_threads,
                                     covariance_momenta_prior_normalized_dof=default.covariance_momenta_prior_normalized_dof,
                                     **kwargs):
    model = BayesianAtlas(
        template_specifications, dataset.dimension, (dataset.tensor_scalar_type, dataset.tensor_integer_type),
        deformation_kernel_type=deformation_kernel_type, deformation_kernel_width=deformation_kernel_width,
        shoot_kernel_type=shoot_kernel_type,
        number_of_time_points=number_of_time_points,
        use_rk2_for_shoot=use_rk2_for_shoot, use_rk2_for_flow=use_rk2_for_flow,
        freeze_template=freeze_template, freeze_control_points=freeze_control_points,
        use_sobolev_gradient=use_sobolev_gradient, smoothing_kernel_width=smoothing_kernel_width,
        dense_mode=dense_mode,
        number_of_threads=number_of_threads)

    if initial_control_points is not None:
        control_points = read_2D_array(initial_control_points)
        model.set_control_points(control_points)
    else:
        model.initial_cp_spacing = initial_cp_spacing

    # Prior on the covariance momenta (inverse Wishart: degrees of freedom parameter).
    model.priors[
        'covariance_momenta'].degrees_of_freedom = dataset.number_of_subjects * covariance_momenta_prior_normalized_dof

    # Prior on the noise variance (inverse Wishart: degrees of freedom parameter).
    for k, object in enumerate(template_specifications.values()):
        model.priors['noise_variance'].degrees_of_freedom.append(dataset.number_of_subjects
                                                                 * object['noise_variance_prior_normalized_dof']
                                                                 * model.objects_noise_dimension[k])

    model.update()

    # Initial random effects realizations.
    individual_RER = {}
    cp = model.get_control_points()
    if initial_momenta is not None:
        individual_RER['momenta'] = read_3D_array(initial_momenta)
    else:
        individual_RER['momenta'] = np.zeros((dataset.number_of_subjects, cp.shape[0], cp.shape[1]))

    """
    Prior on the noise variance (inverse Wishart: scale scalars parameters).
    """

    td, tp, cp = model._fixed_effects_to_torch_tensors(False)
    mom = model._individual_RER_to_torch_tensors(individual_RER, False)

    residuals_per_object = sum(model._compute_residuals(dataset, td, tp, cp, mom))
    for k, object in enumerate(template_specifications.values()):
        if object['noise_variance_prior_scale_std'] is None:
            model.priors['noise_variance'].scale_scalars.append(
                0.01 * residuals_per_object[k].detach().cpu().numpy()
                / model.priors['noise_variance'].degrees_of_freedom[k])
        else:
            model.priors['noise_variance'].scale_scalars.append(object['noise_variance_prior_scale_std'] ** 2)
    model.update()

    # Return the initialized model.
    return model, individual_RER
