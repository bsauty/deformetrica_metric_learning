import os.path
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../../../')
from pydeformetrica.src.core.models.deterministic_atlas import DeterministicAtlas
from pydeformetrica.src.core.estimators.gradient_ascent import GradientAscent
from pydeformetrica.src.in_out.xml_parameters import XmlParameters
from pydeformetrica.src.support.utilities.general_settings import GeneralSettings
from pydeformetrica.src.in_out.dataset_creator import DatasetCreator
from pydeformetrica.src.core.model_tools.deformations.diffeomorphism import Diffeomorphism
from pydeformetrica.src.core.model_tools.attachments.multi_object_attachment import ComputeMultiObjectDistance
import numpy as np

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
templateData = Variable(torch.from_numpy(model.GetTemplateData().Concatenate()), requires_grad = True)
templateObject = model.Template
subjects = dataset.DeformableObjects
subjects = [ [elt for elt in subject[0]] for subject in subjects]
subjectsData = [elt.GetData().Concatenate() for elt in subjects]


def cost(cp, mom, templateData):
    #for each subject, get phi(cp[i], mom[i], template)
    #get the attachment and the deformation norm
    #add the two
    penalty = 0.
    attachment = 0.
    diffeo = Diffeomorphism()
    diffeo.SetKernelWidth(xmlParameters.DeformationKernelWidth)
    diffeo.SetLandmarkPoints(templateData)
    for i, elt in enumerate(subjectsData):
        diffeo.SetStartPositions(cp)
        diffeo.SetStartMomenta(mom[i])
        diffeo.Shoot()
        diffeo.Flow()
        deformedPoints = diffeo.GetLandmarkPoints()
        penalty += diffeo.GetNorm()
        attachment += ComputeMultiObjectDistance(deformedPoints, elt, templateObject, subjects[i], kernel_width=10.)
    return penalty + model.ObjectsNoiseVariance[0] * attachment

# optimizer = optim.Adadelta([cp, mom, templateData], lr=10)
#
# for i in range(10):
#     loss = cost(cp, mom, templateData)
#     optimizer.zero_grad()
#     print(templateData)
#     loss.backward()
#     optimizer.step()
#     print "Iteration: ", i, " Loss: ", loss
#
# #now we need to save
# model.SetTemplateData(templateData.data.numpy())
# model.SetControlPoints(cp.data.numpy())
# model.SetMomenta(mom.data.numpy())
# model.Write()


# print(time.time())
# c = cost(templateData, cp, mom)
# print(time.time())
# print(torch.autograd.grad(c, mom))
# print(time.time())

X = np.array([templateData.data.numpy(), cp.data.numpy(), mom.data.numpy()])
X_shape = X.shape

x_test = X.flatten()
X_test = x_test.reshape(X_shape)

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
    maxiter = 100, ftol = .0001, maxcor=10
))
tend = time.time()
print('------------ END SCIPY OPTIMIZE ------------')
print('Total time: ' + str(tend - tstart))
