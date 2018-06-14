import math
import os
import time
import warnings
import torch

from core.estimators.gradient_ascent import GradientAscent
from core.estimators.scipy_optimize import ScipyOptimize
from core.models.rigid_atlas import RigidAtlas
from in_out.array_readers_and_writers import *
from in_out.dataset_functions import create_dataset


def instantiate_rigid_atlas_model(xml_parameters, dataset):
    model = RigidAtlas()

    # Initial fixed effects --------------------------------------------------------------------------------------------
    # Hyperparameters.
    model.number_of_subjects = dataset.number_of_subjects

    # Template.
    model.initialize_template_attributes(xml_parameters.template_specifications)

    # Translations.
    if dataset is not None and 'landmark_points' in model.template.get_points().keys():
        translations = np.zeros((dataset.number_of_subjects, Settings().dimension))
        targets = [target[0] for target in dataset.deformable_objects]
        template_mean_point = np.mean(model.template.get_points()['landmark_points'], axis=0)
        for i, target in enumerate(targets):
            target_mean_point = np.mean(target.get_points()['landmark_points'], axis=0)
            translations[i] = target_mean_point - template_mean_point
        model.set_translations(translations)

    # Final initialization steps by the model object itself ------------------------------------------------------------
    model.update()

    # Special case of the noise variance hyperparameter ----------------------------------------------------------------
    # Compute residuals if needed.
    if np.min(model.objects_noise_variance) < 0:

        if model.number_of_objects == 1:
            model.objects_noise_variance[0] = 1.0

        else:
            translations = model.get_translations()
            rotations = model.get_rotations()

            template_points = {key: Settings().tensor_scalar_type(value)
                               for key, value in model.template.get_points().items()}
            template_data = {key: Settings().tensor_scalar_type(value)
                             for key, value in model.template.get_data().items()}

            targets = [target[0] for target in dataset.deformable_objects]
            residuals = np.zeros((model.number_of_objects,))

            for i, (subject_id, target) in enumerate(zip(dataset.subject_ids, targets)):
                translation = Settings().tensor_scalar_type(translations[i])
                rotation = Settings().tensor_scalar_type(rotations[i])
                rotation_matrix = model._compute_rotation_matrix(rotation)
                deformed_points = {key: torch.mm(rotation_matrix, value) + translation
                                   for key, value in template_points.items()}
                deformed_data = model.template.get_deformed_data(deformed_points, template_data)
                residuals += model.multi_object_attachment.compute_weighted_distance(
                    deformed_data, model.template, target, model.objects_noise_variance).detach().cpu().numpy()

            # Initialize the noise variance hyperparameter.
            for k, obj in enumerate(xml_parameters.template_specifications.keys()):
                if model.objects_noise_variance[k] < 0:
                    nv = 0.01 * residuals[k] / float(model.number_of_subjects)
                    model.objects_noise_variance[k] = nv
                    print('>> Automatically chosen noise std: %.4f [ %s ]' % (math.sqrt(nv), obj))

    # Return the initialized model.
    return model


def estimate_rigid_atlas(xml_parameters):
    print('')
    print('[ estimate_rigid_atlas function ]')
    print('')

    """
    Create the dataset object.
    """

    dataset = create_dataset(xml_parameters.dataset_filenames, xml_parameters.visit_ages,
                             xml_parameters.subject_ids, xml_parameters.template_specifications)

    assert (dataset.is_cross_sectional()), "Cannot estimate an atlas from a non-cross-sectional dataset."

    """
    Create the model object.
    """

    model = instantiate_rigid_atlas_model(xml_parameters, dataset)

    """
    Create the estimator object.
    """

    if xml_parameters.optimization_method_type.lower() == 'GradientAscent'.lower():
        estimator = GradientAscent()
        estimator.initial_step_size = xml_parameters.initial_step_size
        estimator.scale_initial_step_size = xml_parameters.scale_initial_step_size
        estimator.line_search_shrink = xml_parameters.line_search_shrink
        estimator.line_search_expand = xml_parameters.line_search_expand

    elif xml_parameters.optimization_method_type.lower() == 'ScipyLBFGS'.lower():
        estimator = ScipyOptimize()
        estimator.memory_length = xml_parameters.memory_length

    elif xml_parameters.optimization_method_type == 'ScipyPowell'.lower():
        estimator = ScipyOptimize()
        estimator.method = 'Powell'

    elif xml_parameters.optimization_method_type == 'GridSearch'.lower():
        estimator = ScipyOptimize()
        estimator.method = 'GridSearch'

    else:
        estimator = GradientAscent()
        estimator.initial_step_size = xml_parameters.initial_step_size
        estimator.scale_initial_step_size = xml_parameters.scale_initial_step_size
        estimator.max_line_search_iterations = xml_parameters.max_line_search_iterations
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

    model.name = 'RigidAtlas'

    print('')
    print('[ update method of the ' + estimator.name + ' optimizer ]')

    start_time = time.time()

    estimator.update()
    estimator.write()
    end_time = time.time()
    print('>> Estimation took: ' + str(time.strftime("%H:%M:%S", time.gmtime(end_time - start_time))))

    return model
