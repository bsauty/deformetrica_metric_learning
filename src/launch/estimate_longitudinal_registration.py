import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../../../')

import warnings
import time
import shutil

from pydeformetrica.src.core.models.longitudinal_atlas import LongitudinalAtlas
from pydeformetrica.src.launch.estimate_longitudinal_atlas import instantiate_longitudinal_atlas_model
from pydeformetrica.src.core.estimators.scipy_optimize import ScipyOptimize
from pydeformetrica.src.core.estimators.gradient_ascent import GradientAscent
from pydeformetrica.src.core.estimators.mcmc_saem import McmcSaem
from pydeformetrica.src.core.estimator_tools.samplers.srw_mhwg_sampler import SrwMhwgSampler
from pydeformetrica.src.support.utilities.general_settings import Settings
from pydeformetrica.src.support.kernels.kernel_functions import create_kernel
from pydeformetrica.src.support.probability_distributions.multi_scalar_normal_distribution import \
    MultiScalarNormalDistribution
from pydeformetrica.src.in_out.dataset_functions import create_dataset
from src.in_out.utils import *


def estimate_longitudinal_registration(xml_parameters):
    print('')
    print('[ estimate_longitudinal_registration function ]')
    print('')

    """
    Prepare the loop over each subject.
    """

    full_dataset_filenames = xml_parameters.dataset_filenames
    full_visit_ages = xml_parameters.visit_ages
    number_of_subjects = len(full_dataset_filenames)

    for i in range(number_of_subjects):

        """
        Create the dataset object.
        """

        xml_parameters.dataset_filenames = [full_dataset_filenames[i]]
        xml_parameters.visit_ages = [full_visit_ages[i]]

        dataset = create_dataset(xml_parameters.dataset_filenames, xml_parameters.visit_ages,
                                 xml_parameters.subject_ids, xml_parameters.template_specifications)

        """
        Create a dedicated output folder for the current subject, adapt the global settings.
        """

        registration_output_path = 'longitudinal_registration_subject_' + dataset.subject_ids[0]
        if os.path.isdir(registration_output_path):
            shutil.rmtree(registration_output_path)
            os.mkdir(registration_output_path)

        Settings().output_dir = registration_output_path
        Settings().state_file = os.path.join(registration_output_path, 'pydef_state.p')

        """
        Create the model object.
        """

        model, individual_RER = instantiate_longitudinal_atlas_model(xml_parameters, dataset)

        """
        Create the estimator object.
        """

        if xml_parameters.optimization_method_type == 'GradientAscent'.lower():
            estimator = GradientAscent()
            estimator.initial_step_size = xml_parameters.initial_step_size
            estimator.max_line_search_iterations = xml_parameters.max_line_search_iterations
            estimator.line_search_shrink = xml_parameters.line_search_shrink
            estimator.line_search_expand = xml_parameters.line_search_expand

        elif xml_parameters.optimization_method_type == 'ScipyLBFGS'.lower():
            estimator = ScipyOptimize()
            estimator.max_line_search_iterations = xml_parameters.max_line_search_iterations
            estimator.memory_length = xml_parameters.memory_length
            if not model.is_frozen['template_data'] and model.use_sobolev_gradient and estimator.memory_length > 1:
                print('>> Using a Sobolev gradient for the template data with the ScipyLBFGS estimator memory length '
                      'being larger than 1. Beware: that can be tricky.')

        elif xml_parameters.optimization_method_type == 'McmcSaem'.lower():
            sampler = SrwMhwgSampler()

            momenta_proposal_distribution = MultiScalarNormalDistribution()
            # initial_control_points = model.get_control_points()
            # momenta_proposal_distribution.set_mean(np.zeros(initial_control_points.size,))
            momenta_proposal_distribution.set_variance_sqrt(xml_parameters.momenta_proposal_std)
            sampler.individual_proposal_distributions['momenta'] = momenta_proposal_distribution

            estimator = McmcSaem()
            estimator.sampler = sampler

        else:
            estimator = GradientAscent()
            estimator.initial_step_size = xml_parameters.initial_step_size
            estimator.max_line_search_iterations = xml_parameters.max_line_search_iterations
            estimator.line_search_shrink = xml_parameters.line_search_shrink
            estimator.line_search_expand = xml_parameters.line_search_expand

            msg = 'Unknown optimization-method-type: \"' + xml_parameters.optimization_method_type \
                  + '\". Defaulting to GradientAscent.'
            warnings.warn(msg)

        estimator.max_iterations = xml_parameters.max_iterations
        estimator.convergence_tolerance = xml_parameters.convergence_tolerance

        estimator.print_every_n_iters = xml_parameters.print_every_n_iters
        estimator.save_every_n_iters = xml_parameters.save_every_n_iters

        estimator.dataset = dataset
        estimator.statistical_model = model

        # Initial random effects realizations
        estimator.individual_RER = individual_RER

        """
        Launch.
        """

        if not os.path.exists(Settings().output_dir): os.makedirs(Settings().output_dir)

        model.name = 'LongitudinalRegistration'
        print('')
        print('[ update method of the ' + estimator.name + ' optimizer ]')

        start_time = time.time()
        estimator.update()
        end_time = time.time()
        print('>> Estimation took: ' + str(time.strftime("%H:%M:%S", time.gmtime(end_time - start_time))))

    """
    Gather all the individual registration results.
    """
