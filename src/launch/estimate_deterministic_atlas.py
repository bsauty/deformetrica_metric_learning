import os.path
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../../../')
from pydeformetrica.src.core.models.deterministic_atlas import DeterministicAtlas
from pydeformetrica.src.core.estimators.gradient_ascent import GradientAscent
from pydeformetrica.src.core.estimators.torch_optimize import TorchOptimize
from pydeformetrica.src.in_out.xml_parameters import XmlParameters
from pydeformetrica.src.support.utilities.general_settings import GeneralSettings
from pydeformetrica.src.in_out.dataset_creator import DatasetCreator
from pydeformetrica.src.core.model_tools.deformations.diffeomorphism import Diffeomorphism
from pydeformetrica.src.core.model_tools.attachments.multi_object_attachment import ComputeMultiObjectWeightedDistance

import numpy as np
from scipy.optimize import minimize
import torch
from torch.autograd import Variable
import time
from scipy.optimize import minimize
from torch import optim

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
Read command line, read xml files, create dataset object.

"""

assert len(sys.argv) >= 4, "Usage: " + sys.argv[0] + " <model.xml> <data_set.xml> <optimization_parameters.xml>"
modelXmlPath = sys.argv[1]
datasetXmlPath = sys.argv[2]
optimizationParametersXmlPath = sys.argv[3]

xmlParameters = XmlParameters()
xmlParameters.ReadAllXmls(modelXmlPath, datasetXmlPath, optimizationParametersXmlPath)

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

model.FreezeTemplate = xmlParameters.FreezeTemplate#this should happen before the init of the template and the cps
model.FreezeControlPoints = xmlParameters.FreezeControlPoints

model.InitializeTemplateAttributes(xmlParameters.TemplateSpecifications)

model.SmoothingKernelWidth = xmlParameters.DeformationKernelWidth * xmlParameters.SmoothingKernelWidthRatio
model.InitialCpSpacing = xmlParameters.InitialCpSpacing
model.NumberOfSubjects = dataset.NumberOfSubjects

model.Update()

"""
Create the estimator object.

"""

# estimator = GradientAscent()
estimator = TorchOptimize()

estimator.MaxIterations = xmlParameters.MaxIterations
estimator.PrintEveryNIters = xmlParameters.PrintEveryNIters
estimator.SaveEveryNIters = xmlParameters.SaveEveryNIters
estimator.InitialStepSize = xmlParameters.InitialStepSize
estimator.MaxLineSearchIterations = xmlParameters.MaxLineSearchIterations
estimator.LineSearchShrink = xmlParameters.LineSearchShrink
estimator.LineSearchExpand = xmlParameters.LineSearchExpand
estimator.ConvergenceTolerance = xmlParameters.ConvergenceTolerance

estimator.Dataset = dataset
estimator.StatisticalModel = model

"""
Launch.

"""

model.Name = 'DeterministicAtlas'
estimator.Update()
