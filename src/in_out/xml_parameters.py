import logging
import math
import os
import warnings
import xml.etree.ElementTree as et
from sys import platform

import torch

from core import default

logger = logging.getLogger(__name__)


class XmlParameters:
    """
    XmlParameters object class.
    Parses input xmls and stores the given parameters.

    """

    ####################################################################################################################
    ### Constructor:
    ####################################################################################################################

    def __init__(self):
        self.model_type = default.model_type
        self.template_specifications = default.template_specifications
        self.deformation_kernel_width = 0
        self.deformation_kernel_type = 'torch'
        self.number_of_time_points = default.number_of_time_points
        self.concentration_of_time_points = default.concentration_of_time_points
        self.number_of_sources = default.number_of_sources
        self.use_rk2_for_shoot = default.use_rk2_for_shoot
        self.use_rk2_for_flow = default.use_rk2_for_flow
        self.t0 = None
        self.tmin = default.tmin
        self.tmax = default.tmax
        self.initial_cp_spacing = default.initial_cp_spacing
        self.dimension = default.dimension
        self.covariance_momenta_prior_normalized_dof = default.covariance_momenta_prior_normalized_dof

        self.dataset_filenames = default.dataset_filenames
        self.visit_ages = default.visit_ages
        self.subject_ids = default.subject_ids

        self.optimization_method_type = default.optimization_method_type
        self.optimized_log_likelihood = default.optimized_log_likelihood
        self.number_of_threads = default.number_of_threads
        self.max_iterations = default.max_iterations
        self.max_line_search_iterations = default.max_line_search_iterations
        self.save_every_n_iters = default.save_every_n_iters
        self.print_every_n_iters = default.print_every_n_iters
        self.sample_every_n_mcmc_iters = default.sample_every_n_mcmc_iters
        self.use_sobolev_gradient = default.use_sobolev_gradient
        self.sobolev_kernel_width_ratio = default.sobolev_kernel_width_ratio
        self.initial_step_size = default.initial_step_size
        self.line_search_shrink = default.line_search_shrink
        self.line_search_expand = default.line_search_expand
        self.convergence_tolerance = default.convergence_tolerance
        self.memory_length = default.memory_length
        self.scale_initial_step_size = default.scale_initial_step_size
        self.downsampling_factor = default.downsampling_factor

        self.dense_mode = default.dense_mode

        self.use_cuda = default.use_cuda
        self._cuda_is_used = default._cuda_is_used   # true if at least one operation will use CUDA.
        self._keops_is_used = default._keops_is_used  # true if at least one keops kernel operation will take place.

        self.state_file = None

        self.freeze_template = default.freeze_template
        self.freeze_control_points = default.freeze_control_points
        self.freeze_momenta = default.freeze_momenta
        self.freeze_modulation_matrix = default.freeze_modulation_matrix
        self.freeze_reference_time = default.freeze_reference_time
        self.freeze_time_shift_variance = default.freeze_time_shift_variance
        self.freeze_log_acceleration_variance = default.freeze_log_acceleration_variance
        self.freeze_noise_variance = default.freeze_noise_variance

        self.freeze_translation_vectors = False
        self.freeze_rotation_angles = False
        self.freeze_scaling_ratios = False

        # For metric learning atlas
        self.freeze_metric_parameters = default.freeze_metric_parameters
        self.freeze_p0 = default.freeze_p0
        self.freeze_v0 = default.freeze_v0

        self.initial_control_points = default.initial_control_points
        self.initial_momenta = default.initial_momenta
        self.initial_modulation_matrix = default.initial_modulation_matrix
        self.initial_time_shift_variance = default.initial_time_shift_variance
        self.initial_log_acceleration_mean = default.initial_log_acceleration_mean
        self.initial_log_acceleration_variance = default.initial_log_acceleration_variance
        self.initial_onset_ages = default.initial_onset_ages
        self.initial_log_accelerations = default.initial_log_accelerations
        self.initial_sources = default.initial_sources
        self.initial_sources_mean = default.initial_sources_mean
        self.initial_sources_std = default.initial_sources_std

        self.initial_control_points_to_transport = default.initial_control_points_to_transport

        self.momenta_proposal_std = default.momenta_proposal_std
        self.onset_age_proposal_std = default.onset_age_proposal_std
        self.log_acceleration_proposal_std = default.log_acceleration_proposal_std
        self.sources_proposal_std = default.sources_proposal_std
        self.gradient_based_estimator = default.gradient_based_estimator  # Not connected to anything yet.

        # For scalar inputs:
        self.group_file = default.group_file
        self.observations_file = default.observations_file
        self.timepoints_file = default.timepoints_file
        self.v0 = default.v0
        self.p0 = default.p0
        self.metric_parameters_file = default.metric_parameters_file
        self.interpolation_points_file = default.interpolation_points_file
        self.initial_noise_variance = default.initial_noise_variance
        self.exponential_type = default.exponential_type
        self.number_of_metric_parameters = default.number_of_metric_parameters  # number of parameters in metric learning.
        self.number_of_interpolation_points = default.number_of_interpolation_points
        self.latent_space_dimension = default.latent_space_dimension  # For deep metric learning

        self.normalize_image_intensity = default.normalize_image_intensity
        self.initialization_heuristic = default.initialization_heuristic

    ####################################################################################################################
    ### Public methods:
    ####################################################################################################################

    # Read the parameters from the three PyDeformetrica input xmls, and some further parameters initialization.
    def read_all_xmls(self, model_xml_path, dataset_xml_path, optimization_parameters_xml_path, output_dir):
        self._read_model_xml(model_xml_path)
        self._read_dataset_xml(dataset_xml_path)
        self._read_optimization_parameters_xml(optimization_parameters_xml_path)
        self._further_initialization(output_dir)

    ####################################################################################################################
    ### Private methods:
    ####################################################################################################################

    # Read the parameters from the model xml.
    def _read_model_xml(self, model_xml_path):

        model_xml_level0 = et.parse(model_xml_path).getroot()

        for model_xml_level1 in model_xml_level0:

            if model_xml_level1.tag.lower() == 'model-type':
                self.model_type = model_xml_level1.text.lower()

            elif model_xml_level1.tag.lower() == 'dimension':
                self.dimension = int(model_xml_level1.text)

            elif model_xml_level1.tag.lower() == 'initial-control-points':
                self.initial_control_points = os.path.normpath(
                    os.path.join(os.path.dirname(model_xml_path), model_xml_level1.text))

            elif model_xml_level1.tag.lower() == 'initial-momenta':
                self.initial_momenta = os.path.normpath(
                    os.path.join(os.path.dirname(model_xml_path), model_xml_level1.text))

            elif model_xml_level1.tag.lower() == 'initial-modulation-matrix':
                self.initial_modulation_matrix = os.path.normpath(
                    os.path.join(os.path.dirname(model_xml_path), model_xml_level1.text))

            elif model_xml_level1.tag.lower() == 'initial-time-shift-std':
                self.initial_time_shift_variance = float(model_xml_level1.text) ** 2

            elif model_xml_level1.tag.lower() == 'initial-log-acceleration-std':
                self.initial_log_acceleration_variance = float(model_xml_level1.text) ** 2

            elif model_xml_level1.tag.lower() == 'initial-log-acceleration-mean':
                self.initial_log_acceleration_mean = float(model_xml_level1.text)

            elif model_xml_level1.tag.lower() == 'initial-onset-ages':
                self.initial_onset_ages = os.path.normpath(
                    os.path.join(os.path.dirname(model_xml_path), model_xml_level1.text))

            elif model_xml_level1.tag.lower() == 'initial-log-accelerations':
                self.initial_log_accelerations = os.path.normpath(
                    os.path.join(os.path.dirname(model_xml_path), model_xml_level1.text))

            elif model_xml_level1.tag.lower() == 'initial-sources':
                self.initial_sources = os.path.normpath(
                    os.path.join(os.path.dirname(model_xml_path), model_xml_level1.text))

            elif model_xml_level1.tag.lower() == 'initial-sources-mean':
                self.initial_sources_mean = model_xml_level1.text

            elif model_xml_level1.tag.lower() == 'initial-sources-std':
                self.initial_sources_std = model_xml_level1.text

            elif model_xml_level1.tag.lower() == 'initial-momenta-to-transport':
                self.initial_momenta_to_transport = os.path.normpath(
                    os.path.join(os.path.dirname(model_xml_path), model_xml_level1.text))

            elif model_xml_level1.tag.lower() == 'initial-control-points-to-transport':
                self.initial_control_points_to_transport = os.path.normpath(
                    os.path.join(os.path.dirname(model_xml_path), model_xml_level1.text))

            elif model_xml_level1.tag.lower() == 'initial-noise-std':
                self.initial_noise_variance = float(model_xml_level1.text) ** 2

            elif model_xml_level1.tag.lower() == 'latent-space-dimension':
                self.latent_space_dimension = int(model_xml_level1.text)

            elif model_xml_level1.tag.lower() == 'template':
                for model_xml_level2 in model_xml_level1:

                    if model_xml_level2.tag.lower() == 'dense-mode':
                        self.dense_mode = self._on_off_to_bool(model_xml_level2.text)

                    elif model_xml_level2.tag.lower() == 'object':

                        template_object = self._initialize_template_object_xml_parameters()
                        for model_xml_level3 in model_xml_level2:
                            if model_xml_level3.tag.lower() == 'deformable-object-type':
                                template_object['deformable_object_type'] = model_xml_level3.text.lower()
                            elif model_xml_level3.tag.lower() == 'attachment-type':
                                template_object['attachment_type'] = model_xml_level3.text.lower()
                            elif model_xml_level3.tag.lower() == 'kernel-width':
                                template_object['kernel_width'] = float(model_xml_level3.text)
                            elif model_xml_level3.tag.lower() == 'kernel-type':
                                template_object['kernel_type'] = model_xml_level3.text.lower()
                                if model_xml_level3.text.lower() == 'keops'.lower():
                                    if platform in ['darwin']:
                                        logger.warning(
                                            'The "keops" kernel is unavailable for Mac OS X platforms. '
                                            'Overriding with "torch" kernel. Beware: the memory consumption might '
                                            'explode for high-dimensional data.')
                                        template_object['kernel_type'] = 'torch'
                                    else:
                                        self._keops_is_used = True
                            elif model_xml_level3.tag.lower() == 'noise-std':
                                template_object['noise_std'] = float(model_xml_level3.text)
                            elif model_xml_level3.tag.lower() == 'filename':
                                template_object['filename'] = os.path.normpath(
                                    os.path.join(os.path.dirname(model_xml_path), model_xml_level3.text))
                            elif model_xml_level3.tag.lower() == 'noise-variance-prior-scale-std':
                                template_object['noise_variance_prior_scale_std'] = float(model_xml_level3.text)
                            elif model_xml_level3.tag.lower() == 'noise-variance-prior-normalized-dof':
                                template_object['noise_variance_prior_normalized_dof'] = float(model_xml_level3.text)
                            else:
                                msg = 'Unknown entry while parsing the template > ' + model_xml_level2.attrib['id'] + \
                                      ' object section of the model xml: ' + model_xml_level3.tag
                                warnings.warn(msg)

                        self.template_specifications[model_xml_level2.attrib['id']] = template_object
                    else:
                        msg = 'Unknown entry while parsing the template section of the model xml: ' \
                              + model_xml_level2.tag
                        warnings.warn(msg)

            elif model_xml_level1.tag.lower() == 'deformation-parameters':
                for model_xml_level2 in model_xml_level1:
                    if model_xml_level2.tag.lower() == 'kernel-width':
                        self.deformation_kernel_width = float(model_xml_level2.text)
                    elif model_xml_level2.tag.lower() == 'exponential-type':
                        self.exponential_type = model_xml_level2.text
                    elif model_xml_level2.tag.lower() == 'kernel-type':
                        self.deformation_kernel_type = model_xml_level2.text.lower()
                        if model_xml_level2.text.lower() == 'keops'.lower():
                            if platform in ['darwin']:
                                logger.warning(
                                    'The "keops" kernel is unavailable for Mac OS X platforms. '
                                    'Overriding with "torch" kernel. Beware: the memory consumption might '
                                    'explode for high-dimensional data.')
                                self.deformation_kernel_type = 'torch'
                            else:
                                self._keops_is_used = True
                    elif model_xml_level2.tag.lower() == 'number-of-timepoints':
                        self.number_of_time_points = int(model_xml_level2.text)
                    elif model_xml_level2.tag.lower() == 'number-of-interpolation-points':
                        self.number_of_interpolation_points = int(model_xml_level2.text)
                    elif model_xml_level2.tag.lower() == 'concentration-of-timepoints':
                        self.concentration_of_time_points = int(model_xml_level2.text)
                    elif model_xml_level2.tag.lower() == 'number-of-sources':
                        self.number_of_sources = int(model_xml_level2.text)
                    elif model_xml_level2.tag.lower() == 't0':
                        self.t0 = float(model_xml_level2.text)
                    elif model_xml_level2.tag.lower() == 'tmin':
                        self.tmin = float(model_xml_level2.text)
                    elif model_xml_level2.tag.lower() == 'tmax':
                        self.tmax = float(model_xml_level2.text)
                    elif model_xml_level2.tag.lower() == 'p0':
                        self.p0 = model_xml_level2.text
                    elif model_xml_level2.tag.lower() == 'v0':
                        self.v0 = model_xml_level2.text
                    elif model_xml_level2.tag.lower() == 'metric-parameters-file':  # for metric learning
                        self.metric_parameters_file = model_xml_level2.text
                    elif model_xml_level2.tag.lower() == 'interpolation-points-file':  # for metric learning
                        self.interpolation_points_file = model_xml_level2.text
                    elif model_xml_level2.tag.lower() == 'covariance-momenta-prior-normalized-dof':
                        self.covariance_momenta_prior_normalized_dof = float(model_xml_level2.text)
                    else:
                        msg = 'Unknown entry while parsing the deformation-parameters section of the model xml: ' \
                              + model_xml_level2.tag
                        warnings.warn(msg)
            #
            # elif model_xml_level1.tag.lower() == 'use-exp-parallelization':
            #     self.use_exp_parallelization = self._on_off_to_bool(model_xml_level1.text)

            else:
                msg = 'Unknown entry while parsing root of the model xml: ' + model_xml_level1.tag
                warnings.warn(msg)

    # Read the parameters from the dataset xml.
    def _read_dataset_xml(self, dataset_xml_path):
        if dataset_xml_path is not None:

            dataset_xml_level0 = et.parse(dataset_xml_path).getroot()
            data_set_xml_dirname = os.path.dirname(dataset_xml_path)

            dataset_filenames = []
            visit_ages = []
            subject_ids = []
            for dataset_xml_level1 in dataset_xml_level0:
                if dataset_xml_level1.tag.lower() == 'subject':
                    subject_ids.append(dataset_xml_level1.attrib['id'])

                    subject_filenames = []
                    subject_ages = []
                    for dataset_xml_level2 in dataset_xml_level1:
                        if dataset_xml_level2.tag.lower() == 'visit':

                            visit_filenames = {}
                            for dataset_xml_level3 in dataset_xml_level2:
                                if dataset_xml_level3.tag.lower() == 'filename':
                                    visit_filenames[dataset_xml_level3.attrib['object_id']] = os.path.normpath(
                                        os.path.join(data_set_xml_dirname, dataset_xml_level3.text))
                                elif dataset_xml_level3.tag.lower() == 'age':
                                    subject_ages.append(float(dataset_xml_level3.text))
                            subject_filenames.append(visit_filenames)
                    dataset_filenames.append(subject_filenames)
                    visit_ages.append(subject_ages)

                # For scalar input, following leasp model
                if dataset_xml_level1.tag.lower() == 'group-file':
                    self.group_file = dataset_xml_level1.text

                if dataset_xml_level1.tag.lower() == 'timepoints-file':
                    self.timepoints_file = dataset_xml_level1.text

                if dataset_xml_level1.tag.lower() == 'observations-file':
                    self.observations_file = dataset_xml_level1.text

            self.dataset_filenames = dataset_filenames
            self.visit_ages = visit_ages
            self.subject_ids = subject_ids

    # Read the parameters from the optimization_parameters xml.
    def _read_optimization_parameters_xml(self, optimization_parameters_xml_path):

        optimization_parameters_xml_level0 = et.parse(optimization_parameters_xml_path).getroot()

        for optimization_parameters_xml_level1 in optimization_parameters_xml_level0:
            if optimization_parameters_xml_level1.tag.lower() == 'optimization-method-type':
                self.optimization_method_type = optimization_parameters_xml_level1.text.lower()
            elif optimization_parameters_xml_level1.tag.lower() == 'optimized-log-likelihood':
                self.optimized_log_likelihood = optimization_parameters_xml_level1.text.lower()
            elif optimization_parameters_xml_level1.tag.lower() == 'number-of-threads':
                self.number_of_threads = int(optimization_parameters_xml_level1.text)
            elif optimization_parameters_xml_level1.tag.lower() == 'max-iterations':
                self.max_iterations = int(optimization_parameters_xml_level1.text)
            elif optimization_parameters_xml_level1.tag.lower() == 'convergence-tolerance':
                self.convergence_tolerance = float(optimization_parameters_xml_level1.text)
            elif optimization_parameters_xml_level1.tag.lower() == 'memory-length':
                self.memory_length = int(optimization_parameters_xml_level1.text)
            elif optimization_parameters_xml_level1.tag.lower() == 'downsampling-factor':
                self.downsampling_factor = int(optimization_parameters_xml_level1.text)
            elif optimization_parameters_xml_level1.tag.lower() == 'save-every-n-iters':
                self.save_every_n_iters = int(optimization_parameters_xml_level1.text)
            elif optimization_parameters_xml_level1.tag.lower() == 'print-every-n-iters':
                self.print_every_n_iters = int(optimization_parameters_xml_level1.text)
            elif optimization_parameters_xml_level1.tag.lower() == 'sample-every-n-mcmc-iters':
                self.sample_every_n_mcmc_iters = int(optimization_parameters_xml_level1.text)
            elif optimization_parameters_xml_level1.tag.lower() == 'use-sobolev-gradient':
                self.use_sobolev_gradient = self._on_off_to_bool(optimization_parameters_xml_level1.text)
            elif optimization_parameters_xml_level1.tag.lower() == 'sobolev-kernel-width-ratio':
                self.sobolev_kernel_width_ratio = float(optimization_parameters_xml_level1.text)
            elif optimization_parameters_xml_level1.tag.lower() == 'initial-step-size':
                self.initial_step_size = float(optimization_parameters_xml_level1.text)
            elif optimization_parameters_xml_level1.tag.lower() == 'freeze-template':
                self.freeze_template = self._on_off_to_bool(optimization_parameters_xml_level1.text)
            elif optimization_parameters_xml_level1.tag.lower() == 'freeze-control-points':
                self.freeze_control_points = self._on_off_to_bool(optimization_parameters_xml_level1.text)
            elif optimization_parameters_xml_level1.tag.lower() == 'use-cuda':
                self.use_cuda = self._on_off_to_bool(optimization_parameters_xml_level1.text)
                if self.use_cuda:
                    self._cuda_is_used = True
            elif optimization_parameters_xml_level1.tag.lower() == 'max-line-search-iterations':
                self.max_line_search_iterations = int(optimization_parameters_xml_level1.text)
            elif optimization_parameters_xml_level1.tag.lower() == 'state-file':
                self.state_file = optimization_parameters_xml_level1.text
            elif optimization_parameters_xml_level1.tag.lower() == 'use-rk2-for-shoot':
                self.use_rk2_for_shoot = self._on_off_to_bool(optimization_parameters_xml_level1.text)
            elif optimization_parameters_xml_level1.tag.lower() == 'use-rk2':
                self.use_rk2_for_shoot = self._on_off_to_bool(optimization_parameters_xml_level1.text)
                self.use_rk2_for_flow = self._on_off_to_bool(optimization_parameters_xml_level1.text)
            elif optimization_parameters_xml_level1.tag.lower() == 'momenta-proposal-std':
                self.momenta_proposal_std = float(optimization_parameters_xml_level1.text)
            elif optimization_parameters_xml_level1.tag.lower() == 'onset-age-proposal-std':
                self.onset_age_proposal_std = float(optimization_parameters_xml_level1.text)
            elif optimization_parameters_xml_level1.tag.lower() == 'log-acceleration-proposal-std':
                self.log_acceleration_proposal_std = float(optimization_parameters_xml_level1.text)
            elif optimization_parameters_xml_level1.tag.lower() == 'sources-proposal-std':
                self.sources_proposal_std = float(optimization_parameters_xml_level1.text)
            elif optimization_parameters_xml_level1.tag.lower() == 'scale-initial-step-size':
                self.scale_initial_step_size = self._on_off_to_bool(optimization_parameters_xml_level1.text)
            elif optimization_parameters_xml_level1.tag.lower() == 'initialization-heuristic':
                self.initialization_heuristic = self._on_off_to_bool(optimization_parameters_xml_level1.text)
            elif optimization_parameters_xml_level1.tag.lower() == 'freeze-v0':
                self.freeze_v0 = self._on_off_to_bool(optimization_parameters_xml_level1.text)
            elif optimization_parameters_xml_level1.tag.lower() == 'freeze-p0':
                self.freeze_p0 = self._on_off_to_bool(optimization_parameters_xml_level1.text)
            elif optimization_parameters_xml_level1.tag.lower() == 'freeze-modulation-matrix':
                self.freeze_modulation_matrix = self._on_off_to_bool(optimization_parameters_xml_level1.text)
            elif optimization_parameters_xml_level1.tag.lower() == 'freeze-reference-time':
                self.freeze_reference_time = self._on_off_to_bool(optimization_parameters_xml_level1.text)
            elif optimization_parameters_xml_level1.tag.lower() == 'freeze-time-shift-variance':
                self.freeze_time_shift_variance = self._on_off_to_bool(optimization_parameters_xml_level1.text)
            elif optimization_parameters_xml_level1.tag.lower() == 'freeze-log-acceleration-variance':
                self.freeze_log_acceleration_variance = self._on_off_to_bool(optimization_parameters_xml_level1.text)
            elif optimization_parameters_xml_level1.tag.lower() == 'freeze-reference-time':
                self.freeze_reference_time = self._on_off_to_bool(optimization_parameters_xml_level1.text)
            elif optimization_parameters_xml_level1.tag.lower() == 'freeze-noise-variance':
                self.freeze_noise_variance = self._on_off_to_bool(optimization_parameters_xml_level1.text)
            elif optimization_parameters_xml_level1.tag.lower() == 'freeze-translation-vectors':
                self.freeze_translation_vectors = self._on_off_to_bool(optimization_parameters_xml_level1.text)
            elif optimization_parameters_xml_level1.tag.lower() == 'freeze-rotation-angles':
                self.freeze_rotation_angles = self._on_off_to_bool(optimization_parameters_xml_level1.text)
            elif optimization_parameters_xml_level1.tag.lower() == 'freeze-scaling-ratios':
                self.freeze_scaling_ratios = self._on_off_to_bool(optimization_parameters_xml_level1.text)
            elif optimization_parameters_xml_level1.tag.lower() == 'gradient-based-estimator':
                self.gradient_based_estimator = optimization_parameters_xml_level1.text

            else:
                msg = 'Unknown entry while parsing the optimization_parameters xml: ' \
                      + optimization_parameters_xml_level1.tag
                warnings.warn(msg)

    # Default xml parameters for any template object.
    def _initialize_template_object_xml_parameters(self):
        template_object = {}
        template_object['deformable_object_type'] = 'undefined'
        template_object['kernel_type'] = 'undefined'
        template_object['kernel_width'] = 0.0
        template_object['noise_std'] = -1
        template_object['filename'] = 'undefined'
        template_object['noise_variance_prior_scale_std'] = None
        template_object['noise_variance_prior_normalized_dof'] = 0.01
        return template_object

    def _on_off_to_bool(self, s):
        if s.lower() == "on":
            return True
        elif s.lower() == "off":
            return False
        else:
            raise RuntimeError("Please give a valid flag (on, off)")

    # Based on the raw read parameters, further initialization of some remaining ones.
    def _further_initialization(self, output_dir):

        if self.dense_mode:
            print('>> Dense mode activated. No distinction will be made between template and control points.')
            assert len(self.template_specifications) == 1, \
                'Only a single object can be considered when using the dense mode.'
            if not self.freeze_control_points:
                self.freeze_control_points = True
                msg = 'With active dense mode, the freeze_template (currently %s) and freeze_control_points ' \
                      '(currently %s) flags are redundant. Defaulting to freeze_control_poi∂nts = True.' \
                      % (str(self.freeze_template), str(self.freeze_control_points))
                warnings.warn(msg)
            if self.initial_control_points is not None:
                self.initial_control_points = None
                msg = 'With active dense mode, specifying initial_control_points is useless. Ignoring this xml entry.'
                warnings.warn(msg)

        if self.initial_cp_spacing is None and self.initial_control_points is None and not self.dense_mode:
            print('>> No initial CP spacing given: using diffeo kernel width of ' + str(self.deformation_kernel_width))
            self.initial_cp_spacing = self.deformation_kernel_width

        self.tensor_scalar_type = default.tensor_scalar_type
        # We also set the type to FloatTensor if keops is used.
        if self._keops_is_used:
            assert platform not in ['darwin'], 'The "keops" kernel is not available with the Mac OS X platform.'

            print(">> KEOPS is used at least in one operation, all operations will be done with FLOAT precision.")
            self.tensor_scalar_type = torch.FloatTensor

            if torch.cuda.is_available():
                print('>> CUDA is available: the KEOPS backend will automatically be set to "gpu".')
                self._cuda_is_used = True
            else:
                print('>> CUDA seems to be unavailable: the KEOPS backend will automatically be set to "cpu".')

        # Setting tensor types according to CUDA availability and user choices.
        if self._cuda_is_used:

            if not torch.cuda.is_available():
                msg = 'CUDA seems to be unavailable. All computations will be carried out on CPU.'
                warnings.warn(msg)

            else:
                print(">> CUDA is used at least in one operation, all operations will be done with FLOAT precision.")
                if self.use_cuda:
                    print(">> All tensors will be CUDA tensors.")
                    self.tensor_scalar_type = torch.cuda.FloatTensor
                    self.tensor_integer_type = torch.cuda.LongTensor
                else:
                    print(">> Setting tensor type to float.")
                    self.tensor_scalar_type = torch.FloatTensor

            # Special case of the multiprocessing for the deterministic atlas.
            if self.number_of_threads > 1:
                # if self.model_type == 'DeterministicAtlas'.lower():
                #     # self.number_of_threads = 1
                #     # msg = 'It is not possible at the moment to estimate a deterministic atlas with both CUDA ' \
                #     #       'acceleration and multithreading. Overriding the "number-of-threads" option, now set to 1.'
                #     # warnings.warn(msg)

                if self.model_type in ['BayesianAtlas'.lower(), 'Regression'.lower(), 'Shooting'.lower()]:
                    self.number_of_threads = 1
                    msg = 'It is not possible at the moment to estimate a "%s" model with multithreading. ' \
                          'Overriding the "number-of-threads" option, now set to 1.' % self.model_type
                    warnings.warn(msg)


        # If longitudinal model and t0 is not initialized, initializes it.
        if (self.model_type == 'regression' or self.model_type == 'LongitudinalAtlas'.lower()
            or self.model_type == 'LongitudinalRegistration'.lower()) \
                and (self.t0 is None or self.initial_time_shift_variance is None):
            total_number_of_visits = 0
            mean_visit_age = 0.0
            var_visit_age = 0.0
            for i in range(len(self.visit_ages)):
                for j in range(len(self.visit_ages[i])):
                    total_number_of_visits += 1
                    mean_visit_age += self.visit_ages[i][j]
                    var_visit_age += self.visit_ages[i][j] ** 2

            if total_number_of_visits > 0:
                mean_visit_age /= float(total_number_of_visits)
                var_visit_age = (var_visit_age / float(total_number_of_visits) - mean_visit_age ** 2)

                if self.t0 is None:
                    print('>> Initial t0 set to the mean visit age: %.2f' % mean_visit_age)
                    self.t0 = mean_visit_age
                else:
                    print('>> Initial t0 set by the user to %.2f ; note that the mean visit age is %.2f'
                          % (self.t0, mean_visit_age))

                if not self.model_type == 'regression':
                    if self.initial_time_shift_variance is None:
                        print('>> Initial time-shift std set to the empirical std of the visit ages: %.2f'
                              % math.sqrt(var_visit_age))
                        self.initial_time_shift_variance = var_visit_age
                    else:
                        print(('>> Initial time-shift std set by the user to %.2f ; note that the empirical std of '
                               'the visit ages is %.2f') % (self.initial_time_shift_variance, math.sqrt(var_visit_age)))

        # Setting the number of threads in general settings
        if self.number_of_threads > 1:
            print(">> I will use", self.number_of_threads,
                  "threads, and I set OMP_NUM_THREADS and torch_num_threads to 1.")
            os.environ['OMP_NUM_THREADS'] = "1"
            torch.set_num_threads(1)
        else:
            print('>> Setting OMP_NUM_THREADS and torch_num_threads to 4.')
            os.environ['OMP_NUM_THREADS'] = "4"
            torch.set_num_threads(4)

        try:
            torch.multiprocessing.set_start_method("spawn")
        except RuntimeError as error:
            print('>> Warning: ' + str(error) + ' [ in xml_parameters ]. Ignoring.')

        self._initialize_state_file(output_dir)

        # Freeze the fixed effects in case of a registration.
        if self.model_type == 'Registration'.lower():
            self.freeze_template = True

        elif self.model_type == 'LongitudinalRegistration'.lower():
            self.freeze_template = True
            self.freeze_control_points = True
            self.freeze_momenta = True
            self.freeze_modulation_matrix = True
            self.freeze_reference_time = True
            self.freeze_time_shift_variance = True
            self.freeze_log_acceleration_variance = True
            self.freeze_noise_variance = True

        # Initialize the number of sources if needed.
        if self.model_type == 'LongitudinalAtlas'.lower() \
                and self.initial_modulation_matrix is None and self.number_of_sources is None:
            self.number_of_sources = 4
            print('>> No initial modulation matrix given, neither a number of sources. '
                  'The latter will be ARBITRARILY defaulted to 4.')

        if self.dimension <= 1:
            print("Setting the number of sources to 0 because the dimension is 1.")
            self.number_of_sources = 0

        # Initialize the initial_log_acceleration_variance if needed.
        if (self.model_type == 'LongitudinalAtlas'.lower() or self.model_type == 'LongitudinalRegistration'.lower()) \
                and self.initial_log_acceleration_variance is None:
            print('>> The initial log-acceleration std fixed effect is ARBITRARILY set to 0.5')
            log_acceleration_std = 0.5
            self.initial_log_acceleration_variance = (log_acceleration_std ** 2)

        # Image grid downsampling factor.
        if not self.downsampling_factor == 1:
            image_object_specs = [(key, value) for key, value in self.template_specifications.items()
                                  if value['deformable_object_type'].lower() == 'image']
            if len(image_object_specs) > 2:
                raise RuntimeError('Only a single image object can be used.')
            elif len(image_object_specs) == 1:
                print('>> Setting the image grid downsampling factor to: %d.' % self.downsampling_factor)
                self.template_specifications[image_object_specs[0][0]]['downsampling_factor'] = self.downsampling_factor
            else:
                msg = 'The "downsampling_factor" parameter is useful only for image data, ' \
                      'but none is considered here. Ignoring.'
                warnings.warn(msg)

    def _initialize_state_file(self, output_dir):
        """
        If a state file was given, assert the file exists and set Settings() so that the estimators will try to resume the computations
        If a state file was not given, We automatically create one
        """
        if self.state_file is None:
            self.state_file = os.path.join(output_dir, "pydef_state.p")
        else:
            if os.path.exists(self.state_file):
                self.load_state = True
                print(">> Will attempt to resume computation from file", self.state_file)
            else:
                msg = "A state file was given, but it does not exist. I will save the new state on this file nonetheless."
                warnings.warn(msg)
        print(">> State will be saved in file", self.state_file)
