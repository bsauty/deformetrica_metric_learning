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
        self.number_of_time_points = 10
        self.t0 = None
        self.initial_cp_spacing = -1
        self.dimension = 3

        self.dataset_filenames = []
        self.visit_ages = []
        self.subject_ids = []

        self.optimization_method_type = 'undefined'
        self.number_of_threads = 1
        self.max_iterations = 100
        self.max_line_search_iterations = 10
        self.save_every_n_iters = 100
        self.print_every_n_iters = 1
        self.smoothing_kernel_width_ratio = 1
        self.initial_step_size = 0.001
        self.line_search_shrink = 0.5
        self.line_search_expand = 1.5
        self.convergence_tolerance = 1e-4

        self.freeze_template = False
        self.freeze_control_points = False
        self.use_cuda = False

        self.initial_momenta = None
        self.initial_control_points = None

    def in_off_to_bool(self, s):
        if s.lower() == "on":
            return True
        elif s.lower() == "off":
            return False
        assert False, "Please give a valid flag (on, off)"

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
                self.model_type = model_xml_level1.text

            elif model_xml_level1.tag.lower() == 'dimension':
                self.dimension = int(model_xml_level1.text)

            elif model_xml_level1.tag.lower() == 'initial-momenta':
                self.initial_momenta = model_xml_level1.text

            elif model_xml_level1.tag.lower() == 'initial-control-points':
                self.initial_control_points = model_xml_level1.text

            elif model_xml_level1.tag.lower() == 'template':
                for model_xml_level2 in model_xml_level1:

                    template_object = self._initialize_template_object_xml_parameters()
                    for model_xml_level3 in model_xml_level2:
                        if model_xml_level3.tag.lower() == 'deformable-object-type':
                            template_object['DeformableObjectType'] = model_xml_level3.text.lower()
                        elif model_xml_level3.tag.lower() == 'kernel-width':
                            template_object['KernelWidth'] = float(model_xml_level3.text)
                        elif model_xml_level3.tag.lower() == 'kernel-type':
                            template_object['KernelType'] = model_xml_level3.text.lower()
                        elif model_xml_level3.tag.lower() == 'noise-std':
                            template_object['NoiseStd'] = float(model_xml_level3.text)
                        elif model_xml_level3.tag.lower() == 'filename':
                            template_object['Filename'] = model_xml_level3.text
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
                    else:
                        msg = 'Unknown entry while parsing the deformation-parameters section of the model xml: ' \
                              + model_xml_level2.tag
                        warnings.warn(msg)

            else:
                msg = 'Unknown entry while parsing root of the model xml: ' + model_xml_level1.tag
                warnings.warn(msg)

    # Read the parameters from the dataset xml.
    def _read_dataset_xml(self, datasetXmlPath):
        if datasetXmlPath is not None:

            datasetXml_level0 = et.parse(datasetXmlPath).getroot()

            datasetFilenames = []
            visitAges = []
            subjectIds = []
            for datasetXml_level1 in datasetXml_level0:
                if datasetXml_level1.tag.lower() == 'subject':
                    subjectIds.append(datasetXml_level1.attrib['id'])

                    subjectFilenames = []
                    subjectAges = []
                    for datasetXml_level2 in datasetXml_level1:
                        if datasetXml_level2.tag.lower() == 'visit':

                            visitFilenames = {}
                            for datasetXml_level3 in datasetXml_level2:
                                if datasetXml_level3.tag.lower() == 'filename':
                                    visitFilenames[datasetXml_level3.attrib['object_id']] = datasetXml_level3.text
                                elif datasetXml_level3.tag.lower() == 'age':
                                    subjectAges.append(float(datasetXml_level3.text))
                            subjectFilenames.append(visitFilenames)
                    datasetFilenames.append(subjectFilenames)
                    visitAges.append(subjectAges)
            self.DatasetFilenames = datasetFilenames
            self.VisitAges = visitAges
            self.SubjectIds = subjectIds

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
            elif optimization_parameters_xml_level1.tag.lower() == 'save-every-n-iters':
                self.save_every_n_iters = int(optimization_parameters_xml_level1.text)
            elif optimization_parameters_xml_level1.tag.lower() == 'smoothing-kernel-width-ratio':
                self.smoothing_kernel_width_ratio = float(optimization_parameters_xml_level1.text)
            elif optimization_parameters_xml_level1.tag.lower() == 'initial-step-size':
                self.initial_step_size = float(optimization_parameters_xml_level1.text)
            elif optimization_parameters_xml_level1.tag.lower() == 'freeze-template':
                self.freeze_template = self.in_off_to_bool(optimization_parameters_xml_level1.text)
            elif optimization_parameters_xml_level1.tag.lower() == 'freeze-cp':
                self.freeze_control_points = self.in_off_to_bool(optimization_parameters_xml_level1.text)
            elif optimization_parameters_xml_level1.tag.lower() == 'use-cuda':
                self.use_cuda = self.in_off_to_bool(optimization_parameters_xml_level1.text)
            else:
                msg = 'Unknown entry while parsing the optimization_parameters xml: ' \
                      + optimization_parameters_xml_level1.tag
                warnings.warn(msg)

    # Default xml parameters for any template object.
    def _initialize_template_object_xml_parameters(self):
        template_object = {}
        template_object['DeformableObjectType'] = 'undefined'
        template_object['KernelType'] = 'undefined'
        template_object['KernelWidth'] = 0.0
        template_object['NoiseStd'] = 1.0
        template_object['Filename'] = 'undefined'
        return template_object

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

        #Settting the dimension.
        Settings().Dimension = self.dimension




