import os.path
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../../../')
from pydeformetrica.src.core.models.deterministic_atlas import DeterministicAtlas
from pydeformetrica.src.core.estimators.gradient_ascent import GradientAscent
from pydeformetrica.src.in_out.xml_parameters import XmlParameters
from pydeformetrica.src.support.utilities.general_settings import GeneralSettings
from pydeformetrica.src.in_out.dataset_creator import DatasetCreator
from pydeformetrica.src.core.model_tools.deformations.diffeomorphism import Diffeomorphism
from pydeformetrica.src.core.model_tools.attachments.landmarks_attachments import OrientedSurfaceDistance

import numpy as np

import torch
from torch.autograd import Variable
import time
from scipy.optimize import minimize

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
model.Diffeomorphism.KernelWidth = xmlParameters.DeformationKernelWidth
model.Diffeomorphism.NumberOfTimePoints = xmlParameters.NumberOfTimePoints

model.InitializeTemplateAttributes(xmlParameters.TemplateSpecifications)

model.SmoothingKernelWidth = xmlParameters.DeformationKernelWidth * xmlParameters.SmoothingKernelWidthRatio
model.InitialCpSpacing = xmlParameters.InitialCpSpacing
model.NumberOfSubjects = dataset.NumberOfSubjects

model.Update()

# """
# Create the estimator object.
#
# """
#
# estimator = GradientAscent()
#
# estimator.MaxIterations = xmlParameters.MaxIterations
# estimator.PrintEveryNIters = xmlParameters.PrintEveryNIters
# estimator.SaveEveryNIters = xmlParameters.SaveEveryNIters
# estimator.InitialStepSize = xmlParameters.InitialStepSize
# estimator.MaxLineSearchIterations = xmlParameters.MaxLineSearchIterations
# estimator.LineSearchShrink = xmlParameters.LineSearchShrink
# estimator.LineSearchExpand = xmlParameters.LineSearchExpand
# estimator.ConvergenceTolerance = xmlParameters.ConvergenceTolerance
#
# estimator.Dataset = dataset
# estimator.StatisticalModel = model
#
# """
# Launch.
#
# """
#
# model.Name = 'DeterministicAtlas'
# estimator.Update()

cp = Variable(torch.from_numpy(model.GetControlPoints()), requires_grad=True)
mom = Variable(torch.from_numpy(model.GetMomenta()), requires_grad=True)
templateData = Variable(torch.from_numpy(model.GetTemplateData().RawMatrixList[0]), requires_grad = True)#matrix list
templateObject = model.Template.ObjectList[0]
subjects = dataset.DeformableObjects
subjectsData = [Variable(torch.from_numpy(elt[0][0].GetData())) for elt in subjects]
subjects = [elt[0][0] for elt in subjects]


def cost(templateData, cp, mom):
    #for each subject, get phi(cp[i], mom[i], template)
    #get the attachment and the deformation norm
    #add the two
    penalty = 0.
    attachment = 0.
    for i, elt in enumerate(subjectsData):
        diffeo = Diffeomorphism()
        diffeo.SetKernelWidth(xmlParameters.DeformationKernelWidth)
        diffeo.SetStartPositions(cp)
        diffeo.SetStartMomenta(mom[i])
        diffeo.SetLandmarkPoints(templateData)
        diffeo.Shoot()
        diffeo.Flow()
        deformedPoints = diffeo.GetLandmarkPoints()
        penalty += diffeo.GetNorm()
        attachment += OrientedSurfaceDistance(deformedPoints, elt, templateObject, subjects[i], kernel_width=10.)
    return penalty + model.ObjectsNoiseVariance[0] * attachment

print(time.time())
c = cost(cp, mom, templateData)
print(time.time())
print(torch.autograd.grad(c, mom))
print(time.time())

X = np.array((templateData.data.numpy(), cp.data.numpy(), mom.data.numpy()))
X_shape = X.shape

from scipy.optimize import minimize
def cost_and_derivative_numpy(x_numpy):
    X_numpy = x_numpy.astype('float64').reshape(X_shape)
    td_torch = Variable(torch.from_numpy(X_numpy[0]).type(torch.FloatTensor), requires_grad=True)
    cp_torch = Variable(torch.from_numpy(X_numpy[1]).type(torch.FloatTensor), requires_grad=True)
    mom_torch = Variable(torch.from_numpy(X_numpy[2]).type(torch.FloatTensor), requires_grad=True)

    c = cost(td_torch, cp_torch, mom_torch)
    J_td_numpy = torch.autograd.grad(c, td_torch).flatten().astype('float64')
    J_cp_numpy = torch.autograd.grad(c, cp_torch).flatten().astype('float64')
    J_mom_numpy = torch.autograd.grad(c, mom_torch).flatten().astype('float64')
    J_numpy = (J_td_numpy, J_cp_numpy, J_mom_numpy).flatten()

    return (c.data.numpy(), J_numpy)

print('------------ START SCIPY OPTIMIZE ------------')
tstart = time.time()
res = minimize(cost_and_derivative_numpy, X.flatten(), method='L-BFGS-B', jac=True, options=dict(
    maxiter = 100, ftol = .0001, maxxor=10
))
tend = time.time()
print('------------ END SCIPY OPTIMIZE ------------')
print('Total time: ' + str(tend - tstart))

