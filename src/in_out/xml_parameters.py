import xml.etree.ElementTree as et
import warnings
import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../../../')
from pydeformetrica.src.support.utilities.general_settings import Settings

class XmlParameters:

    """
    XmlParameters object class.
    Parses input xmls and stores the given parameters.

    """

    ####################################################################################################################
    ### Constructor:
    ####################################################################################################################

    def __init__(self):
        self.model_type = 'undefined'
        self.template_specifications = {}
        self.deformation_kernel_width = 0
        self.deformation_kernel_type = 'undefined'
        self.number_of_time_points = 11
        self.transported_trajectory_number_of_time_points = 11
        self.use_rk2 = True
        self.t0 = None
        self.tmin = float('inf')
        self.tmax = - float('inf')
        self.initial_cp_spacing = -1
        self.dimension = 3
        self.covariance_momenta_prior_normalized_dof = 0.001

        self.dataset_filenames = []
        self.visit_ages = []
        self.subject_ids = []

        self.optimization_method_type = 'undefined'
        self.number_of_threads = 1
        self.max_iterations = 100
        self.max_line_search_iterations = 10
        self.save_every_n_iters = 100
        self.print_every_n_iters = 1
        self.use_sobolev_gradient = True
        self.sobolev_kernel_width_ratio = 1
        self.initial_step_size = 0.001
        self.line_search_shrink = 0.5
        self.line_search_expand = 1.2
        self.convergence_tolerance = 1e-4
        self.memory_length = 10

        self.state_file = None

        self.freeze_template = False
        self.freeze_control_points = True
        self.use_cuda = False

        self.initial_momenta = None
        self.initial_control_points = None

        self.use_exp_parallelization = True
        self.initial_control_points_to_transport = None

    ####################################################################################################################
    ### Public methods:
    ####################################################################################################################

    # Read the parameters from the three PyDeformetrica input xmls, and some further parameters initialization.
    def read_all_xmls(self, modelXmlPath, datasetXmlPath, optimizationParametersXmlPath):
        self._read_model_xml(modelXmlPath)
        self._read_dataset_xml(datasetXmlPath)
        self._read_optimization_parameters_xml(optimizationParametersXmlPath)
        self._further_initialization()


    ####################################################################################################################
    ### Private methods:
    ####################################################################################################################

    # Read the parameters from the model xml.
    def _read_model_xml(self, modelXmlPath):

        model_xml_level0 = et.parse(modelXmlPath).getroot()

        for model_xml_level1 in model_xml_level0:

            if model_xml_level1.tag.lower() == 'model-type':
                self.model_type = model_xml_level1.text.lower()


            elif model_xml_level1.tag.lower() == 'dimension':
                self.dimension = int(model_xml_level1.text)

            elif model_xml_level1.tag.lower() == 'initial-momenta':
                self.initial_momenta = model_xml_level1.text

            elif model_xml_level1.tag.lower() == 'initial-control-points':
                self.initial_control_points = model_xml_level1.text

            elif model_xml_level1.tag.lower() == 'initial-momenta-to-transport':
                self.initial_momenta_to_transport = model_xml_level1.text

            elif model_xml_level1.tag.lower() == 'initial-control-points-to-transport':
                self.initial_control_points_to_transport = model_xml_level1.text

            elif model_xml_level1.tag.lower() == 'template':
                for model_xml_level2 in model_xml_level1:

                    template_object = self._initialize_template_object_xml_parameters()
                    for model_xml_level3 in model_xml_level2:
                        if model_xml_level3.tag.lower() == 'deformable-object-type':
                            template_object['deformable_object_type'] = model_xml_level3.text.lower()
                        elif model_xml_level3.tag.lower() == 'attachment-type':
                            template_object['AttachmentType'] = model_xml_level3.text.lower()
                        elif model_xml_level3.tag.lower() == 'kernel-width':
                            template_object['kernel_width'] = float(model_xml_level3.text)
                        elif model_xml_level3.tag.lower() == 'kernel-type':
                            template_object['kernel_type'] = model_xml_level3.text.lower()
                        elif model_xml_level3.tag.lower() == 'noise-std':
                            template_object['noise_std'] = float(model_xml_level3.text)
                        elif model_xml_level3.tag.lower() == 'filename':
                            template_object['filename'] = model_xml_level3.text
                        elif model_xml_level3.tag.lower() == 'noise-variance-prior-scale-std':
                            template_object['noise_variance_prior_scale_std'] = float(model_xml_level3.text)
                        elif model_xml_level3.tag.lower() == 'noise-variance-prior-normalized-dof':
                            template_object['noise_variance_prior_normalized_dof'] = float(model_xml_level3.text)
                        else:
                            msg = 'Unknown entry while parsing the template > ' + model_xml_level2.attrib['id'] + \
                                  ' object section of the model xml: ' + model_xml_level3.tag
                            warnings.warn(msg)
                        self.template_specifications[model_xml_level2.attrib['id']] = template_object

            elif model_xml_level1.tag.lower() == 'deformation-parameters':
                for model_xml_level2 in model_xml_level1:
                    if model_xml_level2.tag.lower() == 'kernel-width':
                        self.deformation_kernel_width = float(model_xml_level2.text)
                    elif model_xml_level2.tag.lower() == 'kernel-type':
                        self.deformation_kernel_type = model_xml_level2.text.lower()
                    elif model_xml_level2.tag.lower() == 'number-of-timepoints':
                        self.number_of_time_points = int(model_xml_level2.text)
                    elif model_xml_level2.tag.lower() == 't0':
                        self.t0 = float(model_xml_level2.text)
                    elif model_xml_level2.tag.lower() == 'tmin':
                        self.tmin = float(model_xml_level2.text)
                    elif model_xml_level2.tag.lower() == 'tmax':
                        self.tmax = float(model_xml_level2.text)
                    elif model_xml_level2.tag.lower() == 'covariance-momenta-prior-normalized-dof':
                        self.covariance_momenta_prior_normalized_dof = float(model_xml_level2.text)
                    else:
                        msg = 'Unknown entry while parsing the deformation-parameters section of the model xml: ' \
                              + model_xml_level2.tag
                        warnings.warn(msg)

            elif model_xml_level1.tag.lower() == 'use-exp-parallelization':
                self.use_exp_parallelization = self._on_off_to_bool(model_xml_level1.text)

            elif model_xml_level1.tag.lower() == 'transported-trajectory-number-of-timepoints':#For parallel transport script.
                self.transported_trajectory_number_of_time_points = int(model_xml_level1.text)

            elif model_xml_level1.tag.lower() == 'transported-trajectory-tmin':#For parallel transport script.
                self.transported_trajectory_tmin = float(model_xml_level1.text)

            elif model_xml_level1.tag.lower() == 'transported-trajectory-tmax':#For parallel transport script.
                self.transported_trajectory_tmax = float(model_xml_level1.text)



            else:
                msg = 'Unknown entry while parsing root of the model xml: ' + model_xml_level1.tag
                warnings.warn(msg)

    # Read the parameters from the dataset xml.
    def _read_dataset_xml(self, dataset_xml_path):
        if dataset_xml_path is not None:

            dataset_xml_level0 = et.parse(dataset_xml_path).getroot()

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
                                    visit_filenames[dataset_xml_level3.attrib['object_id']] = dataset_xml_level3.text
                                elif dataset_xml_level3.tag.lower() == 'age':
                                    subject_ages.append(float(dataset_xml_level3.text))
                            subject_filenames.append(visit_filenames)
                    dataset_filenames.append(subject_filenames)
                    visit_ages.append(subject_ages)
            self.dataset_filenames = dataset_filenames
            self.visit_ages = visit_ages
            self.subject_ids = subject_ids

    # Read the parameters from the optimization_parameters xml.
    def _read_optimization_parameters_xml(self, optimizationParametersXmlPath):

        optimization_parameters_xml_level0 = et.parse(optimizationParametersXmlPath).getroot()

        for optimization_parameters_xml_level1 in optimization_parameters_xml_level0:
            if optimization_parameters_xml_level1.tag.lower() == 'optimization-method-type':
                self.optimization_method_type = optimization_parameters_xml_level1.text.lower()
            elif optimization_parameters_xml_level1.tag.lower() == 'number-of-threads':
                self.number_of_threads = int(optimization_parameters_xml_level1.text)
            elif optimization_parameters_xml_level1.tag.lower() == 'max-iterations':
                self.max_iterations = int(optimization_parameters_xml_level1.text)
            elif optimization_parameters_xml_level1.tag.lower() == 'convergence-tolerance':
                self.convergence_tolerance = float(optimization_parameters_xml_level1.text)
            elif optimization_parameters_xml_level1.tag.lower() == 'memory-length':
                self.memory_length = int(optimization_parameters_xml_level1.text)
            elif optimization_parameters_xml_level1.tag.lower() == 'save-every-n-iters':
                self.save_every_n_iters = int(optimization_parameters_xml_level1.text)
            elif optimization_parameters_xml_level1.tag.lower() == 'use-sobolev-gradient':
                self.use_sobolev_gradient = self._on_off_to_bool(optimization_parameters_xml_level1.text)
            elif optimization_parameters_xml_level1.tag.lower() == 'sobolev-kernel-width-ratio':
                self.sobolev_kernel_width_ratio = float(optimization_parameters_xml_level1.text)
            elif optimization_parameters_xml_level1.tag.lower() == 'initial-step-size':
                self.initial_step_size = float(optimization_parameters_xml_level1.text)
            elif optimization_parameters_xml_level1.tag.lower() == 'freeze-template':
                self.freeze_template = self._on_off_to_bool(optimization_parameters_xml_level1.text)
            elif optimization_parameters_xml_level1.tag.lower() == 'freeze-cp':
                self.freeze_control_points = self._on_off_to_bool(optimization_parameters_xml_level1.text)
            elif optimization_parameters_xml_level1.tag.lower() == 'use-cuda':
                self.use_cuda = self._on_off_to_bool(optimization_parameters_xml_level1.text)
            elif optimization_parameters_xml_level1.tag.lower() == 'max-line-search-iterations':
                self.max_line_search_iterations = int(optimization_parameters_xml_level1.text)
            elif optimization_parameters_xml_level1.tag.lower() == 'use-exp-parallelization':
                self.use_exp_parallelization = self._on_off_to_bool(optimization_parameters_xml_level1.text)
            elif optimization_parameters_xml_level1.tag.lower() == 'state-file':
                self.state_file = optimization_parameters_xml_level1.text
            elif optimization_parameters_xml_level1.tag.lower() == 'use-rk2':
                self.use_rk2 = self._on_off_to_bool(optimization_parameters_xml_level1.text)
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
        template_object['noise_std'] = 1.0
        template_object['filename'] = 'undefined'
        template_object['noise_variance_prior_scale_std'] = None
        template_object['noise_variance_prior_normalized_dof'] = 0.01
        return template_object

    def _on_off_to_bool(self, s):
        if s.lower() == "on": return True
        elif s.lower() == "off": return False
        else: raise RuntimeError("Please give a valid flag (on, off)")

    # Based on the raw read parameters, further initialization of some remaining ones.
    def _further_initialization(self):
        if self.initial_cp_spacing < 0:
            print('>> No initial CP spacing given: using diffeo kernel width of ' + str(self.deformation_kernel_width))
            self.initial_cp_spacing = self.deformation_kernel_width

        # Setting tensor types according to cuda availability.
        if self.use_cuda:
            if not(torch.cuda.is_available()):
                msg = 'Cuda seems to be unavailable. Overriding the use-cuda option.'
                warnings.warn(msg)
            else:
                Settings().tensor_scalar_type = torch.cuda.FloatTensor
                Settings().tensor_integer_type = torch.cuda.LongTensor

        # Setting the dimension.
        Settings().dimension = self.dimension

        # If longitudinal model and t0 is not initialized, initializes it.
        if self.model_type == 'regression' and self.t0 is None:
            total_number_of_visits = 0
            mean_visit_age = 0
            for i in range(len(self.visit_ages)):
                for j in range(len(self.visit_ages[i])):
                    total_number_of_visits += 1
                    mean_visit_age += self.visit_ages[i][j]
            mean_visit_age /= float(total_number_of_visits)
            self.t0 = mean_visit_age

        # Setting the number of threads in general settings
        Settings().number_of_threads = self.number_of_threads
        if self.number_of_threads > 1:
            print(">>> I will use", self.number_of_threads, " threads")

        self._initialize_state_file()


    def _initialize_state_file(self):
        """
        If a state file was given, assert the file exists and set Settings() so that the estimators will try to resume the computations
        If a state file was not given, We automatically create one
        """
        if self.state_file is None:
            self.state_file = os.path.join(Settings().output_dir, "pydef_state.p")
        else:
            Settings().state_file = self.state_file
            if os.path.exists(self.state_file):
                Settings().load_state = True
                print("Will attempt to resume computation from file", self.state_file)
            else:
                msg = "A state file was given, but it does not exist. I will save the new state on this file nonetheless."
                warnings.warn(msg)
        print("State will be saved in file", self.state_file)



