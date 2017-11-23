import xml.etree.ElementTree as et


class XmlParameters:

    """
    XmlParameters object class.
    Parses input xmls and stores the given parameters.

    """

    def ReadAllXmls(self, modelXmlPath, datasetXmlPath, optimizationParametersXmlPath):
        self.ReadModelXml(modelXmlPath)
        self.ReadDatasetXml(datasetXmlPath)
        self.ReadOptimizationParametersXml(optimizationParametersXmlPath)

    def ReadModelXml(self, modelXmlPath):

        modelXml_level0 = et.parse(modelXmlPath).getroot()

        for modelXml_level1 in modelXml_level0:

            if modelXml_level1.tag == 'model-type':
                self.ModelType = modelXml_level1.text

            elif modelXml_level1.tag == 'template':
                templateObjects = {}
                for modelXml_level2 in modelXml_level1:
                    templateObject = {}
                    for modelXml_level3 in modelXml_level2:
                        if modelXml_level3.tag == 'deformable-object-type':
                            templateObject['DeformableObjectType'] = modelXml_level3.text
                        elif modelXml_level3.tag == 'kernel-width':
                            templateObject['KernelWidth'] = float(modelXml_level3.text)
                        elif modelXml_level3.tag == 'kernel-type':
                            templateObject['KernelType'] = modelXml_level3.text
                        elif modelXml_level3.tag == 'noise-std':
                            templateObject['NoiseStd'] = float(modelXml_level3.text)
                        elif modelXml_level3.tag == 'filename':
                            templateObject['Filename'] = modelXml_level3.text
                    templateObjects[modelXml_level2.attrib['id']] = templateObject
                self.TemplateObjectsSpecification = templateObjects

            elif modelXml_level1.tag == 'deformation-parameters':
                for modelXml_level2 in modelXml_level1:
                    if modelXml_level2.tag == 'kernel-width':
                        self.DeformationKernelWidth = float(modelXml_level2.text)
                    elif modelXml_level2.tag == 'kernel-type':
                        self.DeformationKernelType = modelXml_level2.text
                    elif modelXml_level2.tag == 'number-of-timepoints':
                        self.NumberOfTimePoints = int(modelXml_level2.text)


    def ReadDatasetXml(self, datasetXmlPath):

        datasetXml_level0 = et.parse(datasetXmlPath).getroot()

        datasetFilenames = []
        datasetAges = []
        datasetIds = []
        for datasetXml_level1 in datasetXml_level0:
            if datasetXml_level1.tag == 'subject':
                datasetIds.append(datasetXml_level1.attrib['id'])

                subjectFilenames = []
                subjectAges = []
                for datasetXml_level2 in datasetXml_level1:
                    if datasetXml_level2.tag == 'visit':

                        visitFilenames = []
                        for datasetXml_level3 in datasetXml_level2:
                            if datasetXml_level3.tag == 'filename':
                                visitFilenames.append(datasetXml_level3.text)
                            elif datasetXml_level3.tag == 'age':
                                subjectAges.append(float(datasetXml_level3.text))
                        subjectFilenames.append(visitFilenames)
                datasetFilenames.append(subjectFilenames)
                datasetAges.append(subjectAges)
        self.DatasetFilenames = datasetFilenames
        self.DatasetAges = datasetAges
        self.DatasetIds = datasetIds


    def ReadOptimizationParametersXml(self, optimizationParametersXmlPath):

        optimizationParametersXml_level0 = et.parse(optimizationParametersXmlPath).getroot()

        for optimizationParametersXml_level1 in optimizationParametersXml_level0:
            if optimizationParametersXml_level1.tag == 'optimization-method-type':
                self.OptimizationMehodType = optimizationParametersXml_level1.text
            elif optimizationParametersXml_level1.tag == 'number-of-threads':
                self.NumberOfThreads = optimizationParametersXml_level1.text





