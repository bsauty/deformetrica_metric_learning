import math

from core import default
from core.models.affine_atlas import AffineAtlas
from in_out.array_readers_and_writers import *


def instantiate_affine_atlas_model(dataset, template_specifications,
                                   freeze_translation_vectors=default.freeze_translation_vectors,
                                   freeze_rotation_angles=default.freeze_rotation_angles,
                                   freeze_scaling_ratios=default.freeze_scaling_ratios,
                                   **kwargs):

    # model = AffineAtlas(dataset, template_specifications,
    #
    #                     dimension=default.dimension,
    #                     tensor_scalar_type=default.tensor_scalar_type,
    #                     tensor_integer_type=default.tensor_integer_type,
    #                     dense_mode=default.dense_mode,
    #                     number_of_threads=default.number_of_threads,
    #
    #                     freeze_translation_vectors=freeze_translation_vectors,
    #                     freeze_rotation_angles=freeze_rotation_angles,
    #                     freeze_scaling_ratios=freeze_scaling_ratios)

    # Initial fixed effects --------------------------------------------------------------------------------------------
    # Hyperparameters.
    # model.number_of_subjects = dataset.number_of_subjects

    # Template.
    # model.initialize_template_attributes(template_specifications)

    # Translation vectors.
    # model.is_frozen['translation_vectors'] = xml_parameters.freeze_translation_vectors
    # if dataset is not None and 'landmark_points' in model.template.get_points().keys() and not model.is_frozen['translation_vectors']:
    #     translations = np.zeros((dataset.number_of_subjects, dataset.dimension))
    #     targets = [target[0] for target in dataset.deformable_objects]
    #     template_mean_point = np.mean(model.template.get_points()['landmark_points'], axis=0)
    #     for i, target in enumerate(targets):
    #         target_mean_point = np.mean(target.get_points()['landmark_points'], axis=0)
    #         translations[i] = target_mean_point - template_mean_point
    #     model.set_translation_vectors(translations)

    # Rotation angles.
    # model.is_frozen['rotation_angles'] = xml_parameters.freeze_rotation_angles

    # Scaling ratios.
    # model.is_frozen['scaling_ratios'] = xml_parameters.freeze_scaling_ratios

    # Final initialization steps by the model object itself ------------------------------------------------------------
    # model.update()

    # Special case of the noise variance hyperparameter ----------------------------------------------------------------
    # Compute residuals if needed.
    # if np.min(model.objects_noise_variance) < 0:
    #
    #     if model.number_of_objects == 1:
    #         model.objects_noise_variance[0] = 1.0
    #
    #     else:
    #         translation_vectors = model.get_translation_vectors()
    #         rotation_angles = model.get_rotation_angles()
    #         scaling_ratios = model.get_scaling_ratios()
    #
    #         template_points = {key: dataset.tensor_scalar_type(value) for key, value in model.template.get_points().items()}
    #         template_data = {key: dataset.tensor_scalar_type(value) for key, value in model.template.get_data().items()}
    #
    #         targets = [target[0] for target in dataset.deformable_objects]
    #         residuals = np.zeros((model.number_of_objects,))
    #
    #         for i, (subject_id, target) in enumerate(zip(dataset.subject_ids, targets)):
    #             translation_vector_i = dataset.tensor_scalar_type(translation_vectors[i])
    #             rotation_angles_i = dataset.tensor_scalar_type(rotation_angles[i])
    #             scaling_ratio_i = dataset.tensor_scalar_type([scaling_ratios[i]])
    #
    #             deformed_points = model._deform(translation_vector_i, rotation_angles_i, scaling_ratio_i, template_points)
    #             deformed_data = model.template.get_deformed_data(deformed_points, template_data)
    #
    #             residuals += model.multi_object_attachment.compute_weighted_distance(
    #                 deformed_data, model.template, target, model.objects_noise_variance).detach().cpu().numpy()
    #
    #         # Initialize the noise variance hyperparameter.
    #         for k, obj in enumerate(template_specifications.keys()):
    #             if model.objects_noise_variance[k] < 0:
    #                 nv = 0.01 * residuals[k] / float(model.number_of_subjects)
    #                 model.objects_noise_variance[k] = nv
    #                 print('>> Automatically chosen noise std: %.4f [ %s ]' % (math.sqrt(nv), obj))

    # Return the initialized model.
    return model

# TODO remove
# def estimate_affine_atlas(xml_parameters):
#     print('')
#     print('[ estimate_affine_atlas function ]')
#     print('')
#
#     """
#     Create the dataset object.
#     """
#
#     dataset = create_dataset(xml_parameters.dataset_filenames, xml_parameters.visit_ages,
#                              xml_parameters.subject_ids, xml_parameters.template_specifications)
#
#     assert (dataset.is_cross_sectional()), "Cannot estimate an atlas from a non-cross-sectional dataset."
#
#     """
#     Create the model object.
#     """
#
#     model = instantiate_affine_atlas_model(xml_parameters, dataset)
#
#     """
#     Create the estimator object.
#     """
#
#     if xml_parameters.optimization_method_type.lower() == 'GradientAscent'.lower():
#         estimator = GradientAscent()
#         estimator.initial_step_size = xml_parameters.initial_step_size
#         estimator.scale_initial_step_size = xml_parameters.scale_initial_step_size
#         estimator.line_search_shrink = xml_parameters.line_search_shrink
#         estimator.line_search_expand = xml_parameters.line_search_expand
#
#     elif xml_parameters.optimization_method_type.lower() == 'ScipyLBFGS'.lower():
#         estimator = ScipyOptimize()
#         estimator.memory_length = xml_parameters.memory_length
#
#     elif xml_parameters.optimization_method_type == 'ScipyPowell'.lower():
#         estimator = ScipyOptimize()
#         estimator.method = 'Powell'
#
#     elif xml_parameters.optimization_method_type == 'GridSearch'.lower():
#         estimator = ScipyOptimize()
#         estimator.method = 'GridSearch'
#
#     else:
#         estimator = GradientAscent()
#         estimator.initial_step_size = xml_parameters.initial_step_size
#         estimator.scale_initial_step_size = xml_parameters.scale_initial_step_size
#         estimator.max_line_search_iterations = xml_parameters.max_line_search_iterations
#         estimator.line_search_shrink = xml_parameters.line_search_shrink
#         estimator.line_search_expand = xml_parameters.line_search_expand
#
#         msg = 'Unknown optimization-method-type: \"' + xml_parameters.optimization_method_type \
#               + '\". Defaulting to GradientAscent.'
#         warnings.warn(msg)
#
#     estimator.max_iterations = xml_parameters.max_iterations
#     estimator.max_line_search_iterations = xml_parameters.max_line_search_iterations
#     estimator.convergence_tolerance = xml_parameters.convergence_tolerance
#
#     estimator.print_every_n_iters = xml_parameters.print_every_n_iters
#     estimator.save_every_n_iters = xml_parameters.save_every_n_iters
#
#     estimator.dataset = dataset
#     estimator.statistical_model = model
#
#     """
#     Launch.
#     """
#
#     if not os.path.exists(Settings().output_dir):
#         os.makedirs(Settings().output_dir)
#
#     model.name = 'AffineAtlas'
#
#     print('')
#     print('[ update method of the ' + estimator.name + ' optimizer ]')
#
#     start_time = time.time()
#
#     estimator.update()
#     estimator.write()
#     end_time = time.time()
#     print('>> Estimation took: ' + str(time.strftime("%H:%M:%S", time.gmtime(end_time - start_time))))
#
#     return model
