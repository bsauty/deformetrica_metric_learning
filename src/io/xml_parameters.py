import xml.etree.ElementTree as et
import warnings

class XmlParameters:

    """
    XmlParameters object class.
    Parses input xmls and stores the given parameters.

    """

    ################################################################################
    ### Constructor:
    ################################################################################

    def __init__(self):
        self.ModelType = 'undefined'
        self.TemplateSpecifications = {}
        self.DeformationKernelWidth = 0
        self.DeformationKernelType = 'exact'
        self.NumberOfTimePoints = 10
        self.InitialCpSpacing = -1

        self.DatasetFilenames = []
        self.VisitAges = []
        self.SubjectIds = []

        self.OptimizationMehodType = 'gradientascent'
        self.NumberOfThreads = 1
        self.MaxIterations = 100
        self.SaveEveryNIters = 100
        self.SmoothingKernelWidthRatio = 1
        self.InitialStepSize = 0.001

    ################################################################################
    ### Public methods:
    ################################################################################

    # Read the parameters from the three PyDeformetrica input xmls, and some further parameters initialization.
    def ReadAllXmls(self, modelXmlPath, datasetXmlPath, optimizationParametersXmlPath):
        self.ReadModelXml(modelXmlPath)
        self.ReadDatasetXml(datasetXmlPath)
        self.ReadOptimizationParametersXml(optimizationParametersXmlPath)
        self.FurtherInitialization()

    # Read the parameters from the model xml.
    def ReadModelXml(self, modelXmlPath):

        modelXml_level0 = et.parse(modelXmlPath).getroot()

        for modelXml_level1 in modelXml_level0:

            if modelXml_level1.tag.lower() == 'model-type':
                self.ModelType = modelXml_level1.text

            elif modelXml_level1.tag.lower() == 'template':
                for modelXml_level2 in modelXml_level1:

                    templateObject = self.InitializeTemplateObjectXmlParameters()
                    for modelXml_level3 in modelXml_level2:
                        if modelXml_level3.tag.lower() == 'deformable-object-type':
                            templateObject['DeformableObjectType'] = modelXml_level3.text.lower()
                        elif modelXml_level3.tag.lower() == 'kernel-width':
                            templateObject['KernelWidth'] = float(modelXml_level3.text)
                        elif modelXml_level3.tag.lower() == 'kernel-type':
                            templateObject['KernelType'] = modelXml_level3.text.lower()
                        elif modelXml_level3.tag.lower() == 'noise-std':
                            templateObject['NoiseStd'] = float(modelXml_level3.text)
                        elif modelXml_level3.tag.lower() == 'filename':
                            templateObject['Filename'] = modelXml_level3.text
                        else:
                            msg = 'Unknown entry while parsing the template > ' + modelXml_level2.attrib['id'] + \
                                  ' object section of the model xml: ' + modelXml_level3.tag
                            warnings.warn(msg)
                        self.TemplateSpecifications[modelXml_level2.attrib['id']] = templateObject

            elif modelXml_level1.tag.lower() == 'deformation-parameters':
                for modelXml_level2 in modelXml_level1:
                    if modelXml_level2.tag.lower() == 'kernel-width':
                        self.DeformationKernelWidth = float(modelXml_level2.text)
                    elif modelXml_level2.tag.lower() == 'kernel-type':
                        self.DeformationKernelType = modelXml_level2.text.lower()
                    elif modelXml_level2.tag.lower() == 'number-of-timepoints':
                        self.NumberOfTimePoints = int(modelXml_level2.text)
                    else:
                        msg = 'Unknown entry while parsing the deformation-parameters section of the model xml: ' \
                              + modelXml_level2.tag
                        warnings.warn(msg)

            else:
                msg = 'Unknown entry while parsing root of the model xml: ' + modelXml_level1.tag
                warnings.warn(msg)

    # Read the parameters from the dataset xml.
    def ReadDatasetXml(self, datasetXmlPath):

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
    def ReadOptimizationParametersXml(self, optimizationParametersXmlPath):

        optimizationParametersXml_level0 = et.parse(optimizationParametersXmlPath).getroot()

        for optimizationParametersXml_level1 in optimizationParametersXml_level0:
            if optimizationParametersXml_level1.tag.lower() == 'optimization-method-type':
                self.OptimizationMehodType = optimizationParametersXml_level1.text.lower()
            elif optimizationParametersXml_level1.tag.lower() == 'number-of-threads':
                self.NumberOfThreads = int(optimizationParametersXml_level1.text)
            elif optimizationParametersXml_level1.tag.lower() == 'max-iterations':
                self.MaxIterations = int(optimizationParametersXml_level1.text)
            elif optimizationParametersXml_level1.tag.lower() == 'save-every-n-iters':
                self.SaveEveryNIters = int(optimizationParametersXml_level1.text)
            elif optimizationParametersXml_level1.tag.lower() == 'smoothing-kernel-width-ratio':
                self.SmoothingKernelWidthRatio = float(optimizationParametersXml_level1.text)
            elif optimizationParametersXml_level1.tag.lower() == 'initial-step-size':
                self.InitialStepSize = float(optimizationParametersXml_level1.text)
            else:
                msg = 'Unknown entry while parsing the optimization_parameters xml: ' \
                      + optimizationParametersXml_level1.tag
                warnings.warn(msg)


    ################################################################################
    ### Private methods:
    ################################################################################

    # Default xml parameters for any template object.
    def InitializeTemplateObjectXmlParameters(self):
        templateObject = {}
        templateObject['DeformableObjectType'] = 'undefined'
        templateObject['KernelType'] = 'exact'
        templateObject['KernelWidth'] = 0.0
        templateObject['NoiseStd'] = 1.0
        templateObject['Filename'] = 'undefined'
        return templateObject

    # Based on the raw read parameters, further initialization of some remaining ones.
    def FurtherInitialization(self):
        if self.InitialCpSpacing < 0:
            print('>> No initial CP spacing given: using diffeo kernel width of ' + str(self.DeformationKernelWidth))
            self.InitialCpSpacing = self.DeformationKernelWidth
