from core import default
from core.models.longitudinal_atlas import LongitudinalAtlas
from in_out.array_readers_and_writers import *


def instantiate_longitudinal_atlas_model(dataset, template_specifications, t0,
                                         deformation_kernel=default.deformation_kernel,
                                         ignore_noise_variance=False,
                                         dense_mode=default.dense_mode,
                                         concentration_of_time_points=default.concentration_of_time_points,
                                         number_of_time_points=default.number_of_time_points,
                                         use_rk2_for_shoot=default.use_rk2_for_shoot, use_rk2_for_flow=default.use_rk2_for_flow,
                                         freeze_template=False, freeze_control_points=False, freeze_momenta=False, freeze_modulation_matrix=False, freeze_reference_time=False,
                                         freeze_time_shift_variance=False, freeze_log_acceleration_variance=False, freeze_noise_variance=False,
                                         use_sobolev_gradient=True,
                                         initial_control_points=None, initial_cp_spacing=default.initial_cp_spacing,
                                         initial_momenta=None, initial_modulation_matrix=None, number_of_sources=default.number_of_sources,
                                         initial_time_shift_variance=default.initial_time_shift_variance,
                                         initial_log_acceleration_mean=default.initial_log_acceleration_mean,
                                         initial_log_acceleration_variance=default.initial_log_acceleration_variance,
                                         initial_onset_ages=None, initial_log_accelerations=None, initial_sources=None,
                                         sobolev_kernel_width_ratio=default.sobolev_kernel_width_ratio
                                         ):
    if initial_cp_spacing is None:
        initial_cp_spacing = deformation_kernel.kernel_width

    model = LongitudinalAtlas(dataset, template_specifications, dense_mode, deformation_kernel, concentration_of_time_points, number_of_time_points, t0,
                              use_rk2_for_shoot, use_rk2_for_flow,
                              freeze_template=freeze_template, freeze_control_points=freeze_control_points, freeze_momenta=freeze_momenta,
                              freeze_modulation_matrix=freeze_modulation_matrix, freeze_reference_time=freeze_reference_time, freeze_time_shift_variance=freeze_time_shift_variance,
                              freeze_log_acceleration_variance=freeze_log_acceleration_variance, freeze_noise_variance=freeze_noise_variance,
                              initial_cp_spacing=default.initial_cp_spacing, use_sobolev_gradient=use_sobolev_gradient,
                              smoothing_kernel_width=deformation_kernel.kernel_width * sobolev_kernel_width_ratio,
                              number_of_sources=number_of_sources)

    # Deformation object -----------------------------------------------------------------------------------------------
    # model.spatiotemporal_reference_frame.set_kernel(kernel_factory.factory(xml_parameters.deformation_kernel_type, xml_parameters.deformation_kernel_width))
    # model.spatiotemporal_reference_frame.set_concentration_of_time_points(xml_parameters.concentration_of_time_points)
    # model.spatiotemporal_reference_frame.set_number_of_time_points(xml_parameters.number_of_time_points)
    # model.spatiotemporal_reference_frame.set_use_rk2_for_shoot(xml_parameters.use_rk2_for_shoot)
    # model.spatiotemporal_reference_frame.set_use_rk2_for_flow(xml_parameters.use_rk2_for_flow)

    # Initial fixed effects and associated priors ----------------------------------------------------------------------
    # Template.
    # model.is_frozen['template_data'] = xml_parameters.freeze_template
    model.initialize_template_attributes(template_specifications)
    # model.use_sobolev_gradient = xml_parameters.use_sobolev_gradient
    # model.smoothing_kernel_width = xml_parameters.deformation_kernel_width * xml_parameters.sobolev_kernel_width_ratio
    model.initialize_template_data_variables()

    # Control points.
    # model.is_frozen['control_points'] = xml_parameters.freeze_control_points
    if initial_control_points is not None:
        control_points = read_2D_array(initial_control_points)
        print('>> Reading ' + str(len(control_points)) + ' initial control points from file: ' + initial_control_points)
        model.set_control_points(control_points)
    else:
        model.initial_cp_spacing = initial_cp_spacing
    model.initialize_control_points_variables()

    # Momenta.
    # model.is_frozen['momenta'] = xml_parameters.freeze_momenta
    if not initial_momenta is None:
        momenta = read_3D_array(initial_momenta)
        print('>> Reading ' + str(len(momenta)) + ' initial momenta from file: ' + initial_momenta)
        model.set_momenta(momenta)
    model.initialize_momenta_variables()

    # Modulation matrix.
    # model.is_frozen['modulation_matrix'] = xml_parameters.freeze_modulation_matrix
    if initial_modulation_matrix is not None:
        modulation_matrix = read_2D_array(initial_modulation_matrix)
        if len(modulation_matrix.shape) == 1:
            modulation_matrix = modulation_matrix.reshape(-1, 1)
        print('>> Reading ' + str(modulation_matrix.shape[1]) + '-source initial modulation matrix from file: ' + initial_modulation_matrix)
        model.set_modulation_matrix(modulation_matrix)
    else:
        model.number_of_sources = number_of_sources
    model.initialize_modulation_matrix_variables()

    # Reference time.
    # model.is_frozen['reference_time'] = xml_parameters.freeze_reference_time
    model.set_reference_time(t0)
    model.priors['reference_time'].set_variance(initial_time_shift_variance)
    model.initialize_reference_time_variables()

    # Time-shift variance.
    # model.is_frozen['time_shift_variance'] = xml_parameters.freeze_time_shift_variance
    model.set_time_shift_variance(initial_time_shift_variance)

    # Log-acceleration.
    # model.is_frozen['log_acceleration_variance'] = xml_parameters.freeze_log_acceleration_variance
    model.individual_random_effects['log_acceleration'].set_mean(initial_log_acceleration_mean)
    model.set_log_acceleration_variance(initial_log_acceleration_variance)

    # Initial random effects realizations ------------------------------------------------------------------------------
    number_of_subjects = len(dataset.dataset_filenames)
    total_number_of_observations = sum([len(elt) for elt in dataset.dataset_filenames])

    # Onset ages.
    if initial_onset_ages is not None:
        onset_ages = read_2D_array(initial_onset_ages)
        print('>> Reading initial onset ages from file: ' + initial_onset_ages)
    else:
        onset_ages = np.zeros((number_of_subjects,)) + model.get_reference_time()
        print('>> Initializing all onset ages to the initial reference time: %.2f' % model.get_reference_time())

    # Log-accelerations.
    if initial_log_accelerations is not None:
        log_accelerations = read_2D_array(initial_log_accelerations)
        print('>> Reading initial log-accelerations from file: ' + initial_log_accelerations)
    else:
        log_accelerations = np.zeros((number_of_subjects,))
        print('>> Initializing all log-accelerations to zero.')

    # Sources.
    if initial_sources is not None:
        sources = read_2D_array(initial_sources).reshape((-1, model.number_of_sources))
        print('>> Reading initial sources from file: ' + initial_sources)
    else:
        sources = np.zeros((number_of_subjects, model.number_of_sources))
        print('>> Initializing all sources to zero')

    # Final gathering.
    individual_RER = {}
    individual_RER['sources'] = sources
    individual_RER['onset_age'] = onset_ages
    individual_RER['log_acceleration'] = log_accelerations

    # Special case of the noise variance -------------------------------------------------------------------------------
    # model.is_frozen['noise_variance'] = xml_parameters.freeze_noise_variance
    initial_noise_variance = model.get_noise_variance()

    # Compute residuals if needed.
    if not ignore_noise_variance:

        # Compute initial residuals if needed.
        if np.min(initial_noise_variance) < 0:

            template_data, template_points, control_points, momenta, modulation_matrix = model._fixed_effects_to_torch_tensors(False)
            sources, onset_ages, log_accelerations = model._individual_RER_to_torch_tensors(individual_RER, False)
            absolute_times, tmin, tmax = model._compute_absolute_times(dataset.times, onset_ages, log_accelerations)
            model._update_spatiotemporal_reference_frame(template_points, control_points, momenta, modulation_matrix, tmin, tmax)
            residuals = model._compute_residuals(dataset, template_data, absolute_times, sources)

            residuals_per_object = np.zeros((model.number_of_objects,))
            for i in range(len(residuals)):
                for j in range(len(residuals[i])):
                    residuals_per_object += residuals[i][j].data.numpy()

            # Initialize noise variance fixed effect, and the noise variance prior if needed.
            for k, obj in enumerate(template_specifications.values()):
                dof = total_number_of_observations * obj['noise_variance_prior_normalized_dof'] * model.objects_noise_dimension[k]
                nv = 0.01 * residuals_per_object[k] / dof

                if initial_noise_variance[k] < 0:
                    print('>> Initial noise variance set to %.2f based on the initial mean residual value.' % nv)
                    model.fixed_effects['noise_variance'][k] = nv

        # Initialize the dof if needed.
        if not model.is_frozen['noise_variance']:
            for k, obj in enumerate(template_specifications.values()):
                dof = total_number_of_observations * obj['noise_variance_prior_normalized_dof'] * model.objects_noise_dimension[k]
                model.priors['noise_variance'].degrees_of_freedom.append(dof)

    # Final initialization steps by the model object itself ------------------------------------------------------------
    model.update()

    return model, individual_RER


def estimate_longitudinal_atlas(dataset, statistical_model, estimator):
    print('')
    print('[ estimate_longitudinal_atlas function ]')
    print('')

    """
    Create the dataset object.
    """

    # dataset = create_dataset(xml_parameters.dataset_filenames, xml_parameters.visit_ages,
    #                          xml_parameters.subject_ids, xml_parameters.template_specifications)

    """
    Create the model object.
    """

    # model, individual_RER = instantiate_longitudinal_atlas_model(xml_parameters, dataset)

    """
    Create the estimator object.
    """

    # if xml_parameters.optimization_method_type == 'GradientAscent'.lower():
    #     estimator = GradientAscent()
    #     estimator.initial_step_size = xml_parameters.initial_step_size
    #     estimator.scale_initial_step_size = xml_parameters.scale_initial_step_size
    #     estimator.max_line_search_iterations = xml_parameters.max_line_search_iterations
    #     estimator.line_search_shrink = xml_parameters.line_search_shrink
    #     estimator.line_search_expand = xml_parameters.line_search_expand
    #
    # elif xml_parameters.optimization_method_type == 'ScipyLBFGS'.lower():
    #     estimator = ScipyOptimize()
    #     estimator.max_line_search_iterations = xml_parameters.max_line_search_iterations
    #     estimator.memory_length = xml_parameters.memory_length
    #     if not model.is_frozen['template_data'] and model.use_sobolev_gradient and estimator.memory_length > 1:
    #         print('>> Using a Sobolev gradient for the template data with the ScipyLBFGS estimator memory length '
    #               'being larger than 1. Beware: that can be tricky.')
    #         # estimator.memory_length = 1
    #         # msg = 'Impossible to use a Sobolev gradient for the template data with the ScipyLBFGS estimator memory ' \
    #         #       'length being larger than 1. Overriding the "memory_length" option, now set to "1".'
    #         # warnings.warn(msg)
    #
    # elif xml_parameters.optimization_method_type == 'McmcSaem'.lower():
    #     # Onset age proposal distribution.
    #     onset_age_proposal_distribution = MultiScalarNormalDistribution()
    #     onset_age_proposal_distribution.set_variance_sqrt(xml_parameters.onset_age_proposal_std)
    #     # sampler.individual_proposal_distributions['onset_age'] = onset_age_proposal_distribution
    #
    #     # Log-acceleration proposal distribution.
    #     log_acceleration_proposal_distribution = MultiScalarNormalDistribution()
    #     log_acceleration_proposal_distribution.set_variance_sqrt(xml_parameters.log_acceleration_proposal_std)
    #     # sampler.individual_proposal_distributions['log_acceleration'] = log_acceleration_proposal_distribution
    #
    #     # Sources proposal distribution.
    #     sources_proposal_distribution = MultiScalarNormalDistribution()
    #     sources_proposal_distribution.set_variance_sqrt(xml_parameters.sources_proposal_std)
    #     # sampler.individual_proposal_distributions['sources'] = sources_proposal_distribution
    #
    #     sampler = SrwMhwgSampler(onset_age_proposal_distribution, log_acceleration_proposal_distribution, sources_proposal_distribution)
    #
    #     estimator = McmcSaem(statistical_model, dataset,
    #         max_iterations=default.max_iterations,
    #         print_every_n_iters=print_every_n_iters, save_every_n_iters=default.save_every_n_iters,
    #         sampler=sampler,
    #         sample_every_n_mcmc_iters=sample_every_n_mcmc_iters,
    #         gradient_based_estimator=,
    #         individual_RER={},
    #     )
    #     # estimator.sampler = sampler
    #     # estimator.sample_every_n_mcmc_iters = xml_parameters.sample_every_n_mcmc_iters
    #     # estimator.print_every_n_iters = xml_parameters.print_every_n_iters
    #
    #     # Gradient-based estimator.
    #     # estimator.gradient_based_estimator = ScipyOptimize()
    #     # estimator.gradient_based_estimator.memory_length = 5
    #
    #     estimator.gradient_based_estimator = GradientAscent(
    #         statistical_model, dataset,
    #         optimized_log_likelihood=default.optimized_log_likelihood,
    #         max_iterations=default.max_iterations, convergence_tolerance=default.convergence_tolerance,
    #         print_every_n_iters=default.print_every_n_iters, save_every_n_iters=default.save_every_n_iters,
    #         scale_initial_step_size=default.scale_initial_step_size, initial_step_size=default.initial_step_size,
    #         max_line_search_iterations=default.max_line_search_iterations,
    #         line_search_shrink=default.line_search_shrink,
    #         line_search_expand=default.line_search_expand,
    #         output_dir=default.output_dir,
    #         individual_RER={},
    #     )
    #
    #
    #     estimator.gradient_based_estimator.initial_step_size = xml_parameters.initial_step_size
    #     estimator.gradient_based_estimator.scale_initial_step_size = xml_parameters.scale_initial_step_size
    #     estimator.gradient_based_estimator.line_search_shrink = xml_parameters.line_search_shrink
    #     estimator.gradient_based_estimator.line_search_expand = xml_parameters.line_search_expand
    #
    #     estimator.gradient_based_estimator.statistical_model = model
    #     estimator.gradient_based_estimator.dataset = dataset
    #     estimator.gradient_based_estimator.optimized_log_likelihood = 'class2'
    #     estimator.gradient_based_estimator.max_iterations = 5
    #     estimator.gradient_based_estimator.max_line_search_iterations = xml_parameters.max_line_search_iterations
    #     estimator.gradient_based_estimator.convergence_tolerance = xml_parameters.convergence_tolerance
    #     estimator.gradient_based_estimator.print_every_n_iters = 1
    #     estimator.gradient_based_estimator.save_every_n_iters = 100000
    #
    # else:
    #     estimator = GradientAscent()
    #     estimator.initial_step_size = xml_parameters.initial_step_size
    #     estimator.scale_initial_step_size = xml_parameters.scale_initial_step_size
    #     estimator.max_line_search_iterations = xml_parameters.max_line_search_iterations
    #     estimator.line_search_shrink = xml_parameters.line_search_shrink
    #     estimator.line_search_expand = xml_parameters.line_search_expand
    #
    #     msg = 'Unknown optimization-method-type: \"' + xml_parameters.optimization_method_type \
    #           + '\". Defaulting to GradientAscent.'
    #     warnings.warn(msg)
    #
    # estimator.optimized_log_likelihood = xml_parameters.optimized_log_likelihood
    #
    # estimator.max_iterations = xml_parameters.max_iterations
    # estimator.convergence_tolerance = xml_parameters.convergence_tolerance
    #
    # estimator.print_every_n_iters = xml_parameters.print_every_n_iters
    # estimator.save_every_n_iters = xml_parameters.save_every_n_iters
    #
    # estimator.dataset = dataset
    # estimator.statistical_model = model
    #
    # # Initial random effects realizations
    # estimator.individual_RER = individual_RER

    # """
    # Launch.
    # """
    #
    # if not os.path.exists(Settings().output_dir): os.makedirs(Settings().output_dir)
    #
    # model.name = 'LongitudinalAtlas'
    # print('')
    # print('[ update method of the ' + estimator.name + ' optimizer ]')
    # print('')
    #
    # start_time = time.time()
    # estimator.update()
    # estimator.write()
    # end_time = time.time()
    # print('>> Estimation took: ' + str(time.strftime("%H:%M:%S", time.gmtime(end_time - start_time))))

    # return model
