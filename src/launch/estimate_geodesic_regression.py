import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../../../')

print(sys.path)

import torch
import warnings
import time

from pydeformetrica.src.core.models.deterministic_atlas import DeterministicAtlas
from pydeformetrica.src.core.estimators.torch_optimize import TorchOptimize
from pydeformetrica.src.core.estimators.scipy_optimize import ScipyOptimize
from pydeformetrica.src.core.estimators.gradient_ascent import GradientAscent
from pydeformetrica.src.in_out.xml_parameters import XmlParameters
from pydeformetrica.src.support.utilities.general_settings import Settings
from pydeformetrica.src.in_out.dataset_creator import DatasetCreator

"""
Basic info printing.

"""

print('')
print('##############################')
print('##### PyDeformetrica 1.0 #####')
print('##############################')
print('')

print('[ estimate_deterministic_atlas function ]')
print('')

"""
Read command line, read xml files, set general settings.

"""

assert len(sys.argv) >= 4, "Usage: " + sys.argv[0] + " <model.xml> <data_set.xml> <optimization_parameters.xml>"
modelXmlPath = sys.argv[1]
datasetXmlPath = sys.argv[2]
optimizationParametersXmlPath = sys.argv[3]

xmlParameters = XmlParameters()
xmlParameters.read_all_xmls(modelXmlPath, datasetXmlPath, optimizationParametersXmlPath)

Settings().Dimension = xmlParameters.Dimension

"""
Create the dataset object.

"""

datasetCreator = DatasetCreator()
dataset = datasetCreator.CreateDataset(xmlParameters.DatasetFilenames, xmlParameters.VisitAges,
                                       xmlParameters.SubjectIds, xmlParameters.TemplateSpecifications)

"""
Create the model object.

"""

model = DeterministicAtlas()

model.Diffeomorphism.KernelType = xmlParameters.DeformationKernelType
model.Diffeomorphism.SetKernelWidth(xmlParameters.DeformationKernelWidth)
model.Diffeomorphism.NumberOfTimePoints = xmlParameters.NumberOfTimePoints
model.Diffeomorphism.T0 = xmlParameters.T0
model.Diffeomorphism.TN = xmlParameters.TN

model.FreezeTemplate = xmlParameters.FreezeTemplate  # this should happen before the init of the template and the cps
model.FreezeControlPoints = xmlParameters.FreezeControlPoints

model._initialize_template_attributes(xmlParameters.TemplateSpecifications)

model.SmoothingKernelWidth = xmlParameters.DeformationKernelWidth * xmlParameters.SmoothingKernelWidthRatio
model.InitialCpSpacing = xmlParameters.InitialCpSpacing
model.NumberOfSubjects = dataset.NumberOfSubjects

model.update()

"""
Create the estimator object.

"""

if xmlParameters.OptimizationMethodType == 'GradientAscent'.lower():
    estimator = GradientAscent()
    estimator.InitialStepSize = xmlParameters.InitialStepSize
    estimator.MaxLineSearchIterations = xmlParameters.MaxLineSearchIterations
    estimator.LineSearchShrink = xmlParameters.LineSearchShrink
    estimator.LineSearchExpand = xmlParameters.LineSearchExpand
elif xmlParameters.OptimizationMethodType == 'TorchLBFGS'.lower():
    estimator = TorchOptimize()
elif xmlParameters.OptimizationMethodType == 'ScipyLBFGS'.lower():
    estimator = ScipyOptimize()
else:
    estimator = TorchOptimize()
    msg = 'Unknown optimization-method-type: \"' + xmlParameters.OptimizationMethodType \
          + '\". Defaulting to TorchLBFGS.'
    warnings.warn(msg)

estimator.MaxIterations = xmlParameters.MaxIterations
estimator.ConvergenceTolerance = xmlParameters.ConvergenceTolerance

estimator.PrintEveryNIters = xmlParameters.PrintEveryNIters
estimator.SaveEveryNIters = xmlParameters.SaveEveryNIters

estimator.Dataset = dataset
estimator.StatisticalModel = model

"""
Launch.

"""

if not os.path.exists('output'):
    os.makedirs('output')

model.Name = 'DeterministicAtlas'

startTime = time.time()
estimator.update()
endTime = time.time()
print('>> Estimation took: ' + str(time.strftime("%H:%M:%S", time.gmtime(endTime - startTime))))
