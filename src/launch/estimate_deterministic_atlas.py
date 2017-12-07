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
from pydeformetrica.src.core.model_tools.attachments.multi_object_attachment import ComputeMultiObjectDistance
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

# model.Name = 'DeterministicAtlas'
# estimator.Update()


"""
Scripts for fast testing.

"""

cp = model.GetControlPoints()
mom = model.GetMomenta()
templateData = model.GetTemplateData()
templateObject = model.Template
print([len(elt) for elt in templateObject.GetData().RawMatrixList])
print([len(elt) for elt in templateObject.GetData().RawMatrixList])
print([len(elt) for elt in templateObject.GetData().RawMatrixList])
templateData = Variable(torch.from_numpy(templateObject.GetData().Concatenate()), requires_grad=True)
subjects = dataset.DeformableObjects
subjects = [subject[0] for subject in subjects]#because longitudinal
subjectsData = [Variable(torch.from_numpy(elt.GetData().Concatenate())) for elt in subjects]

print("Control points :", cp.size())
print("Momenta:", mom.size())
print("Template Data:", templateData.size())


def cost(templateData, cp, mom):
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
        attachment += ComputeMultiObjectDistance(deformedPoints, elt, templateObject, subjects[i], 10)
    return penalty + model.ObjectsNoiseVariance[0] * attachment

optimizer = optim.Adadelta([templateData, cp, mom], lr=10)

tstart = time.time()
print(tstart)
for i in range(10):
    loss = cost(templateData, cp, mom)
    optimizer.zero_grad()
    loss.backward()
    # optimizer.step()
    print "Iteration: ", i, " Loss: ", loss[0]
tend = time.time()
print("Computation time :", (tend-tstart))

#now we need to save
model.FixedEffects['TemplateData'] = templateData
model.FixedEffects['ControlPoints'] = cp
model.FixedEffects['Momenta'] = mom
model.Write()
#
# aTemp, bTemp = templateData.data.numpy().shape
# aCp, bCp = cp.data.numpy().shape
# aMom, bMom, cMom = mom.data.numpy().shape
#
# X = np.concatenate([templateData.data.numpy().flatten(), cp.data.numpy().flatten(), mom.data.numpy().flatten()])
#
# def cost_and_derivative_numpy(x):
#
#     tempData = x[:aTemp * bTemp].reshape((aTemp, bTemp))
#     cpData = x[aTemp * bTemp:aTemp * bTemp + aCp * bCp].reshape((aCp, bCp))
#     momData = x[aTemp * bTemp + aCp * bCp:].reshape((aMom, bMom, cMom))
#
#     td_torch = Variable(torch.from_numpy(tempData), requires_grad=True)
#     cp_torch = Variable(torch.from_numpy(cpData), requires_grad=True)
#     mom_torch = Variable(torch.from_numpy(momData), requires_grad=True)
#
#     c = cost(td_torch, cp_torch, mom_torch)
#
#     c.backward()
#     J_td_numpy = td_torch.grad.data.numpy()
#     J_cp_numpy = cp_torch.grad.data.numpy()
#     J_mom_numpy = mom_torch.grad.data.numpy()
#
#     c_gradient = np.concatenate([J_td_numpy.flatten(), J_cp_numpy.flatten(), J_mom_numpy.flatten()])
#
#     return (c.data.numpy()[0], c_gradient)
#
# print('------------ START SCIPY OPTIMIZE ------------')
# tstart = time.time()
# print(tstart)
# res = minimize(cost_and_derivative_numpy, X, method='L-BFGS-B', jac=True, options={'maxiter':10, 'ftol':.02, 'maxcor':10, 'disp':True})
# tend = time.time()
# print('------------ END SCIPY OPTIMIZE ------------')
# print('Total time: ' + str(tend - tstart))
