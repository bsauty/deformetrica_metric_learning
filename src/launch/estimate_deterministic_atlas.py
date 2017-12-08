import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../../../')

import torch
import warnings

from pydeformetrica.src.core.models.deterministic_atlas import DeterministicAtlas
from pydeformetrica.src.core.estimators.torch_optimize import TorchOptimize
from pydeformetrica.src.core.estimators.gradient_ascent import GradientAscent
from pydeformetrica.src.in_out.xml_parameters import XmlParameters
from pydeformetrica.src.support.utilities.general_settings import *
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
xmlParameters.ReadAllXmls(modelXmlPath, datasetXmlPath, optimizationParametersXmlPath)

Settings().Dimension = xmlParameters.Dimension

if xmlParameters.UseCuda:
    if not(torch.cuda.is_available()):
        msg = 'Cuda seems to be unavailable. Overriding the use-cuda option.'
        warnings.warn(msg)
    else:
        Settings().TorchTensorType = torch.cuda.LongTensor


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

model.FreezeTemplate = xmlParameters.FreezeTemplate #this should happen before the init of the template and the cps
model.FreezeControlPoints = xmlParameters.FreezeControlPoints

model.InitializeTemplateAttributes(xmlParameters.TemplateSpecifications)

model.SmoothingKernelWidth = xmlParameters.DeformationKernelWidth * xmlParameters.SmoothingKernelWidthRatio
model.InitialCpSpacing = xmlParameters.InitialCpSpacing
model.NumberOfSubjects = dataset.NumberOfSubjects

model.Update()

"""
Create the estimator object.

"""

if xmlParameters.OptimizationMethodType == 'GradientAscent'.lower():
    estimator = GradientAscent()
    estimator.InitialStepSize = xmlParameters.InitialStepSize
    estimator.MaxLineSearchIterations = xmlParameters.MaxLineSearchIterations
    estimator.LineSearchShrink = xmlParameters.LineSearchShrink
    estimator.LineSearchExpand = xmlParameters.LineSearchExpand
    estimator.ConvergenceTolerance = xmlParameters.ConvergenceTolerance
elif xmlParameters.OptimizationMethodType == 'TorchLBFGS'.lower():
    estimator = TorchOptimize()
else:
    estimator = TorchOptimize()
    msg = 'Unknown optimization-method-type: \"' + xmlParameters.OptimizationMethodType \
          + '\". Defaulting to TorchLBFGS.'
    warnings.warn(msg)

estimator.MaxIterations = xmlParameters.MaxIterations
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
estimator.Update()
