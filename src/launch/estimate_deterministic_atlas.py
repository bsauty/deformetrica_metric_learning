import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../../../')

import torch
import warnings
import time

from pydeformetrica.src.core.models.deterministic_atlas import DeterministicAtlas
from pydeformetrica.src.core.estimators.torch_optimize import TorchOptimize
from pydeformetrica.src.core.estimators.scipy_optimize import ScipyOptimize
from pydeformetrica.src.core.estimators.gradient_ascent import GradientAscent
from pydeformetrica.src.in_out.xml_parameters import XmlParameters
from pydeformetrica.src.support.utilities.general_settings import *
from pydeformetrica.src.support.kernels.kernel_functions import create_kernel
from pydeformetrica.src.in_out.dataset_functions import create_dataset
from src.in_out.utils import *

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

"""
Create the dataset object.

"""

dataset = create_dataset(xmlParameters.DatasetFilenames, xmlParameters.VisitAges,
                         xmlParameters.SubjectIds, xmlParameters.TemplateSpecifications)

assert (dataset.is_cross_sectionnal(), "Cannot run a deterministic atlas on a non-cross-sectionnal dataset.")

"""
Create the model object.

"""

model = DeterministicAtlas()

model.diffeomorphism.kernel = create_kernel(xmlParameters.DeformationKernelType, xmlParameters.DeformationKernelWidth)
model.diffeomorphism.number_of_time_points = xmlParameters.NumberOfTimePoints

if not xmlParameters.InitialControlPoints is None:
    control_points = read_2D_array(xmlParameters.InitialControlPoints)
    model.set_control_points(control_points)

if not xmlParameters.InitialMomenta is None:
    momenta = read_momenta(xmlParameters.InitialMomenta)
    model.set_momenta(momenta)

model.freeze_template = xmlParameters.FreezeTemplate  # this should happen before the init of the template and the cps
model.freeze_control_points = xmlParameters.FreezeControlPoints

model._initialize_template_attributes(xmlParameters.TemplateSpecifications)

model.smoothing_kernel_width = xmlParameters.DeformationKernelWidth * xmlParameters.SmoothingKernelWidthRatio
model.initial_cp_spacing = xmlParameters.InitialCpSpacing
model.number_of_subjects = dataset.number_of_subjects

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

if not os.path.exists(Settings().OutputDir):
    os.makedirs(Settings().OutputDir)

model.Name = 'DeterministicAtlas'

startTime = time.time()
estimator.update()
endTime = time.time()
print('>> Estimation took: ' + str(time.strftime("%H:%M:%S", time.gmtime(endTime - startTime))))
