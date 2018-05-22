import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../../../')

import numpy as np
import unittest
from torch.autograd import Variable
import torch

from pydeformetrica.src.support.utilities.general_settings import Settings
from pydeformetrica.src.support.kernels.kernel_functions import create_kernel

if torch.cuda.is_available():
    pass


class CudaKernelTests(unittest.TestCase):
    """
    Methods with names starting by "test" will be run
    """
    def setUp(self):
        pass

    def test_gpu_and_cpu_convolve_are_equal(self):
        # Parameters.
        kernel_width = 10.
        number_of_control_points = 10
        dimension = 3
        tensor_scalar_type = torch.cuda.FloatTensor

        # Set the global settings accordingly.
        Settings().dimension = dimension
        Settings().tensor_scalar_type = tensor_scalar_type

        # Instantiate the needed objects.
        gpu_kernel = create_kernel('keops', kernel_width)
        cpu_kernel = create_kernel('torch', kernel_width)
        random_control_points_1 = Variable(torch.from_numpy(
            np.random.randn(number_of_control_points, dimension)).type(tensor_scalar_type))
        random_control_points_2 = Variable(torch.from_numpy(
            np.random.randn(number_of_control_points, dimension)).type(tensor_scalar_type))
        random_momenta = Variable(torch.from_numpy(
            np.random.randn(number_of_control_points, dimension)).type(tensor_scalar_type))

        # Compute the desired quantities.
        gpu_convolve_11 = gpu_kernel.convolve(
            random_control_points_1, random_control_points_1, random_momenta).data.cpu().numpy()
        cpu_convolve_11 = cpu_kernel.convolve(
            random_control_points_1, random_control_points_1, random_momenta).data.cpu().numpy()
        gpu_convolve_12 = gpu_kernel.convolve(
            random_control_points_1, random_control_points_2, random_momenta).data.cpu().numpy()
        cpu_convolve_12 = cpu_kernel.convolve(
            random_control_points_1, random_control_points_2, random_momenta).data.cpu().numpy()

        # Print.
        # print('>> gpu_convolve_11 = ')
        # print(gpu_convolve_11)
        # print('>> cpu_convolve_11 = ')
        # print(cpu_convolve_11)
        # print('>> np.mean(np.abs(gpu_convolve_11.ravel() - cpu_convolve_11.ravel()) = %f' %
        #       np.mean(np.abs(gpu_convolve_11.ravel() - cpu_convolve_11.ravel())))
        # print('>> gpu_convolve_12 = ')
        # print(gpu_convolve_12)
        # print('>> cpu_convolve_12 = ')
        # print(cpu_convolve_12)
        # print('>> np.mean(np.abs(gpu_convolve_12.ravel() - cpu_convolve_12.ravel()) = %f' %
        #       np.mean(np.abs(gpu_convolve_12.ravel() - cpu_convolve_12.ravel())))

        # Check for equality.
        self.assertTrue(np.mean(np.abs(gpu_convolve_11.ravel() - cpu_convolve_11.ravel())) < 1e-5)
        self.assertTrue(np.mean(np.abs(gpu_convolve_12.ravel() - cpu_convolve_12.ravel())) < 1e-5)

    def test_gpu_and_cpu_convolve_gradient_are_equal(self):
        # Parameters.
        kernel_width = 10.
        number_of_control_points = 10
        dimension = 3
        tensor_scalar_type = torch.cuda.FloatTensor

        # Set the global settings accordingly.
        Settings().dimension = dimension
        Settings().tensor_scalar_type = tensor_scalar_type

        # Instantiate the needed objects.
        gpu_kernel = create_kernel('keops', kernel_width)
        cpu_kernel = create_kernel('torch', kernel_width)
        random_control_points_1 = Variable(torch.from_numpy(
            np.random.randn(number_of_control_points, dimension)).type(tensor_scalar_type), requires_grad=True)
        random_momenta_1 = Variable(torch.from_numpy(
            np.random.randn(number_of_control_points, dimension)).type(tensor_scalar_type))
        random_control_points_2 = Variable(torch.from_numpy(
            np.random.randn(number_of_control_points, dimension)).type(tensor_scalar_type), requires_grad=True)
        random_momenta_2 = Variable(torch.from_numpy(
            np.random.randn(number_of_control_points, dimension)).type(tensor_scalar_type))

        # Compute the desired quantities.
        gpu_convolve_gradient_11 = gpu_kernel.convolve_gradient(
            random_momenta_1, random_control_points_1).data.cpu().numpy()
        cpu_convolve_gradient_11 = cpu_kernel.convolve_gradient(
            random_momenta_1, random_control_points_1).data.cpu().numpy()
        gpu_convolve_gradient_11_bis = gpu_kernel.convolve_gradient(
            random_momenta_1, random_control_points_1, random_control_points_1, random_momenta_1).data.cpu().numpy()
        cpu_convolve_gradient_11_bis = cpu_kernel.convolve_gradient(
            random_momenta_1, random_control_points_1, random_control_points_1, random_momenta_1).data.cpu().numpy()
        gpu_convolve_gradient_12 = gpu_kernel.convolve_gradient(
            random_momenta_1, random_control_points_1, random_control_points_2, random_momenta_2).data.cpu().numpy()
        cpu_convolve_gradient_12 = cpu_kernel.convolve_gradient(
            random_momenta_1, random_control_points_1, random_control_points_2, random_momenta_2).data.cpu().numpy()

        # Print.
        # print('>> gpu_convolve_gradient_11 = ')
        # print(gpu_convolve_gradient_11)
        # print('>> cpu_convolve_gradient_11 = ')
        # print(cpu_convolve_gradient_11)
        # print('>> np.mean(np.abs(gpu_convolve_gradient_11.ravel() - cpu_convolve_gradient_11.ravel())) = %f' %
        #       np.mean(np.abs(gpu_convolve_gradient_11.ravel() - cpu_convolve_gradient_11.ravel())))
        # print('>> gpu_convolve_gradient_11_bis = ')
        # print(gpu_convolve_gradient_11_bis)
        # print('>> cpu_convolve_gradient_11_bis = ')
        # print(cpu_convolve_gradient_11_bis)
        # print('>> np.mean(np.abs(gpu_convolve_gradient_11_bis.ravel() - cpu_convolve_gradient_11_bis.ravel())) = %f' %
        #       np.mean(np.abs(gpu_convolve_gradient_11_bis.ravel() - cpu_convolve_gradient_11_bis.ravel())))
        # print('>> gpu_convolve_gradient_12 = ')
        # print(gpu_convolve_gradient_12)
        # print('>> cpu_convolve_gradient_12 = ')
        # print(cpu_convolve_gradient_12)
        # print('>> np.mean(np.abs(gpu_convolve_gradient_12.ravel() - cpu_convolve_gradient_12.ravel())) = %f' %
        #       np.mean(np.abs(gpu_convolve_gradient_12.ravel() - cpu_convolve_gradient_12.ravel())))

        # Check for equality.
        self.assertTrue(
            np.mean(np.abs(gpu_convolve_gradient_11_bis.ravel() - gpu_convolve_gradient_11_bis.ravel())) < 1e-5)
        self.assertTrue(
            np.mean(np.abs(cpu_convolve_gradient_11_bis.ravel() - cpu_convolve_gradient_11_bis.ravel())) < 1e-5)
        self.assertTrue(np.mean(np.abs(gpu_convolve_gradient_11.ravel() - cpu_convolve_gradient_11.ravel())) < 1e-5)
        self.assertTrue(np.mean(np.abs(gpu_convolve_gradient_12.ravel() - cpu_convolve_gradient_12.ravel())) < 1e-5)

