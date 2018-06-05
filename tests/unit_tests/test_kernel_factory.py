import logging
import unittest

import torch
import numpy as np

import support.kernels as kernel_factory
from support.utilities.general_settings import Settings


class KernelFactory(unittest.TestCase):
    def test_instantiate_abstract_class(self):
        with self.assertRaises(TypeError):
            kernel_factory.AbstractKernel()

    def test_unknown_kernel_string(self):
        with self.assertRaises(TypeError):
            kernel_factory.factory('unknown_type')

    def test_non_cuda_kernel_factory(self):
        for k in [kernel_factory.Type.NO_KERNEL, kernel_factory.Type.TORCH]:
            logging.debug("testing kernel=", k)
            instance = kernel_factory.factory(k, kernel_width=1.)
            self.__isKernelValid(instance)

    @unittest.skipIf(not torch.cuda.is_available(), 'cuda is not available')
    def test_cuda_kernel_factory(self):
        for k in [kernel_factory.Type.KEOPS, kernel_factory.Type.TORCH_CUDA]:
            logging.debug("testing kernel=", k)
            instance = kernel_factory.factory(k, kernel_width=1.)
            self.__isKernelValid(instance)

    def test_non_cuda_kernel_factory_from_string(self):
        for k in ['no_kernel', 'no-kernel', 'torch']:
            logging.debug("testing kernel=", k)
            instance = kernel_factory.factory(k, kernel_width=1.)
            self.__isKernelValid(instance)

    @unittest.skipIf(not torch.cuda.is_available(), 'cuda is not available')
    def test_cuda_kernel_factory_from_string(self):
        for k in ['keops']:
            logging.debug("testing kernel=", k)
            instance = kernel_factory.factory(k, kernel_width=1.)
            self.__isKernelValid(instance)

    def __isKernelValid(self, instance):
        if instance is not None:
            self.assertIsInstance(instance, kernel_factory.AbstractKernel)
            self.assertEqual(instance.kernel_width, 1.)


class KernelTestBase(unittest.TestCase):
    def setUp(self):
        Settings().dimension = 3
        Settings().tensor_scalar_type = torch.FloatTensor

        torch.manual_seed(42)  # for reproducibility
        torch.set_printoptions(precision=30)  # for more precision when printing tensor

        self.x = torch.rand([4, 3]).type(Settings().tensor_scalar_type)
        self.y = torch.rand([4, 3]).type(Settings().tensor_scalar_type)
        self.p = torch.rand([4, 3]).type(Settings().tensor_scalar_type)
        self.expected_convolve_res = torch.tensor([
            [1.098455905914306640625000000000, 0.841387629508972167968750000000, 1.207388281822204589843750000000],
            [1.135044455528259277343750000000, 0.859343230724334716796875000000, 1.387768864631652832031250000000],
            [1.258846044540405273437500000000, 0.927951455116271972656250000000, 1.383145809173583984375000000000],
            [1.334064722061157226562500000000, 0.887639760971069335937500000000, 1.360101222991943359375000000000]])

        self.expected_convolve_gradient_res = torch.tensor([
            [-1.623382568359375000000000000000, -1.212645769119262695312500000000, 1.440739274024963378906250000000],
            [-1.414733767509460449218750000000, 1.848072409629821777343750000000, -0.102501690387725830078125000000],
            [1.248104929924011230468750000000, 0.059575259685516357421875000000, -1.860013246536254882812500000000],
            [1.790011405944824218750000000000, -0.695001959800720214843750000000, 0.521775603294372558593750000000]])

        super().setUp()

    def _assert_tensor_close(self, t1, t2):
        if t1.requires_grad is True:
            t1 = t1.detach()
        if t2.requires_grad is True:
            t2 = t2.detach()

        # print(t1)
        # print(t2)
        # print(t1 - t2)
        self.assertTrue(np.allclose(t1, t2, rtol=1e-5, atol=1e-5),
                        'Tested tensors are not within acceptable tolerance levels')


@unittest.skipIf(not torch.cuda.is_available(), 'cuda is not available')
class Kernel(KernelTestBase):
    def setUp(self):
        self.test_on_device = 'cuda:0'
        self.kernel_instance = kernel_factory.factory(kernel_factory.Type.TORCH_CUDA,
                                                      kernel_width=1., device=self.test_on_device)
        super().setUp()

    def test_torch_cuda_with_move_to_device(self):
        res = self.kernel_instance.convolve(self.x, self.y, self.p)
        self.assertEqual(res.device, torch.device(self.test_on_device))
        # move to CPU
        res = res.to(torch.device('cpu'))
        self._assert_tensor_close(res, self.expected_convolve_res)

        # test convolve gradient method
        res = self.kernel_instance.convolve_gradient(self.x, self.x)
        self.assertEqual(res.device, torch.device(self.test_on_device))
        # move to CPU
        res = res.to(torch.device('cpu'))
        # print(res)
        self._assert_tensor_close(res, self.expected_convolve_gradient_res)

    def test_torch_cuda_without_move_to_device(self):
        res = self.kernel_instance.convolve(self.x, self.y, self.p)
        self.assertEqual(res.device, torch.device(self.test_on_device))
        # move to CPU
        res = res.to(torch.device('cpu'))
        self._assert_tensor_close(res, self.expected_convolve_res)

        # test convolve gradient method
        res = self.kernel_instance.convolve_gradient(self.x, self.x)
        self.assertEqual(res.device, torch.device(self.test_on_device))
        # move to CPU
        res = res.to(torch.device('cpu'))
        # print(res)
        self._assert_tensor_close(res, self.expected_convolve_gradient_res)


class KeopsKernel(KernelTestBase):
    def setUp(self):
        super().setUp()
        self.kernel_instance = kernel_factory.factory(kernel_factory.Type.KEOPS, kernel_width=1.)

    def test_convolve(self):
        res = self.kernel_instance.convolve(self.x, self.y, self.p)
        self._assert_tensor_close(res, self.expected_convolve_res)

    def test_convolve_gradient(self):
        res = self.kernel_instance.convolve_gradient(self.x, self.x)
        self._assert_tensor_close(res, self.expected_convolve_gradient_res)

    @unittest.skipIf(not torch.cuda.is_available(), 'cuda is not available')
    def test_convolve_gpu(self):
        res = self.kernel_instance.convolve(self.x, self.y, self.p, backend='GPU')
        self._assert_tensor_close(res, self.expected_convolve_res)

    @unittest.skipIf(not torch.cuda.is_available(), 'cuda is not available')
    def test_convolve_gradient_gpu(self):
        res = self.kernel_instance.convolve_gradient(self.x, self.x, backend='GPU')
        self._assert_tensor_close(res, self.expected_convolve_gradient_res)


class KeopsVersusCuda(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        pass

    def test_keops_and_torch_gaussian_convolve_are_equal(self):
        # Parameters.
        kernel_width = 10.
        number_of_control_points = 10
        dimension = 3
        # tensor_scalar_type = torch.cuda.FloatTensor
        tensor_scalar_type = torch.FloatTensor

        # Set the global settings accordingly.
        Settings().dimension = dimension
        Settings().tensor_scalar_type = tensor_scalar_type

        # Instantiate the needed objects.
        keops_kernel = kernel_factory.factory(kernel_factory.Type.KEOPS, kernel_width)
        torch_kernel = kernel_factory.factory(kernel_factory.Type.TORCH, kernel_width)
        random_control_points_1 = torch.from_numpy(
            np.random.randn(number_of_control_points, dimension)).type(tensor_scalar_type).requires_grad_()
        random_control_points_2 = torch.from_numpy(
            np.random.randn(number_of_control_points, dimension)).type(tensor_scalar_type).requires_grad_()
        random_momenta_1 = torch.from_numpy(
            np.random.randn(number_of_control_points, dimension)).type(tensor_scalar_type).requires_grad_()
        random_momenta_2 = torch.from_numpy(
            np.random.randn(number_of_control_points, dimension)).type(tensor_scalar_type).requires_grad_()

        # Compute the desired forward quantities.
        keops_convolve_11 = keops_kernel.convolve(random_control_points_1, random_control_points_1, random_momenta_1)
        torch_convolve_11 = torch_kernel.convolve(random_control_points_1, random_control_points_1, random_momenta_1)
        keops_convolve_12 = keops_kernel.convolve(random_control_points_1, random_control_points_2, random_momenta_2)
        torch_convolve_12 = torch_kernel.convolve(random_control_points_1, random_control_points_2, random_momenta_2)

        # Compute the desired backward quantities.
        keops_total_12 = torch.dot(random_momenta_1.view(-1), keops_convolve_12.view(-1))
        torch_total_12 = torch.dot(random_momenta_1.view(-1), torch_convolve_12.view(-1))

        [keops_dcp_1, keops_dcp_2, keops_dmom_1, keops_dmom_2] = torch.autograd.grad(
            keops_total_12, [random_control_points_1, random_control_points_2, random_momenta_1, random_momenta_2])
        [torch_dcp_1, torch_dcp_2, torch_dmom_1, torch_dmom_2] = torch.autograd.grad(
            torch_total_12, [random_control_points_1, random_control_points_2, random_momenta_1, random_momenta_2])

        # Convert back to numpy.
        keops_convolve_11 = keops_convolve_11.detach().cpu().numpy()
        torch_convolve_11 = torch_convolve_11.detach().cpu().numpy()
        keops_convolve_12 = keops_convolve_12.detach().cpu().numpy()
        torch_convolve_12 = torch_convolve_12.detach().cpu().numpy()
        keops_dcp_1 = keops_dcp_1.detach().cpu().numpy()
        keops_dcp_2 = keops_dcp_2.detach().cpu().numpy()
        keops_dmom_1 = keops_dmom_1.detach().cpu().numpy()
        keops_dmom_2 = keops_dmom_2.detach().cpu().numpy()
        torch_dcp_1 = torch_dcp_1.detach().cpu().numpy()
        torch_dcp_2 = torch_dcp_2.detach().cpu().numpy()
        torch_dmom_1 = torch_dmom_1.detach().cpu().numpy()
        torch_dmom_2 = torch_dmom_2.detach().cpu().numpy()

        # Check for equality.
        self.assertTrue(np.allclose(keops_convolve_11, torch_convolve_11, rtol=1e-05, atol=1e-05))
        self.assertTrue(np.allclose(keops_convolve_12, torch_convolve_12, rtol=1e-05, atol=1e-05))
        self.assertTrue(np.allclose(keops_dcp_1, torch_dcp_1, rtol=1e-05, atol=1e-05))
        self.assertTrue(np.allclose(keops_dcp_2, torch_dcp_2, rtol=1e-05, atol=1e-05))
        self.assertTrue(np.allclose(keops_dmom_1, torch_dmom_1, rtol=1e-05, atol=1e-05))
        self.assertTrue(np.allclose(keops_dmom_2, torch_dmom_2, rtol=1e-05, atol=1e-05))

    def test_keops_and_torch_varifold_convolve_are_equal(self):
        # Parameters.
        kernel_width = 10.
        number_of_control_points = 10
        dimension = 3
        # tensor_scalar_type = torch.cuda.FloatTensor
        tensor_scalar_type = torch.FloatTensor

        # Set the global settings accordingly.
        Settings().dimension = dimension
        Settings().tensor_scalar_type = tensor_scalar_type

        # Instantiate the needed objects.
        keops_kernel = kernel_factory.factory(kernel_factory.Type.KEOPS, kernel_width)
        torch_kernel = kernel_factory.factory(kernel_factory.Type.TORCH, kernel_width)
        random_points_1 = torch.from_numpy(
            np.random.randn(number_of_control_points, dimension)).type(tensor_scalar_type).requires_grad_()
        random_points_2 = torch.from_numpy(
            np.random.randn(number_of_control_points, dimension)).type(tensor_scalar_type).requires_grad_()
        random_normals_1 = torch.from_numpy(
            np.random.randn(number_of_control_points, dimension)).type(tensor_scalar_type).requires_grad_()
        random_normals_2 = torch.from_numpy(
            np.random.randn(number_of_control_points, dimension)).type(tensor_scalar_type).requires_grad_()
        random_areas_1 = torch.from_numpy(
            np.random.randn(number_of_control_points, 1)).type(tensor_scalar_type).requires_grad_()
        random_areas_2 = torch.from_numpy(
            np.random.randn(number_of_control_points, 1)).type(tensor_scalar_type).requires_grad_()

        # Compute the desired forward quantities.
        keops_convolve_11 = keops_kernel.convolve(
            (random_points_1, random_normals_1), (random_points_1, random_normals_1), random_areas_1, mode='varifold')
        torch_convolve_11 = torch_kernel.convolve(
            (random_points_1, random_normals_1), (random_points_1, random_normals_1), random_areas_1, mode='varifold')
        keops_convolve_12 = keops_kernel.convolve(
            (random_points_1, random_normals_1), (random_points_2, random_normals_2), random_areas_2, mode='varifold')
        torch_convolve_12 = torch_kernel.convolve(
            (random_points_1, random_normals_1), (random_points_2, random_normals_2), random_areas_2, mode='varifold')

        # Compute the desired backward quantities.
        keops_total_12 = torch.dot(random_areas_1.view(-1), keops_convolve_12.view(-1))
        torch_total_12 = torch.dot(random_areas_1.view(-1), torch_convolve_12.view(-1))

        [keops_dp_1, keops_dp_2, keops_dn_1, keops_dn_2, keops_da_1, keops_da_2] = torch.autograd.grad(
            keops_total_12,
            [random_points_1, random_points_2, random_normals_1, random_normals_2, random_areas_1, random_areas_2])
        [torch_dp_1, torch_dp_2, torch_dn_1, torch_dn_2, torch_da_1, torch_da_2] = torch.autograd.grad(
            torch_total_12,
            [random_points_1, random_points_2, random_normals_1, random_normals_2, random_areas_1, random_areas_2])

        # Convert back to numpy.
        keops_convolve_11 = keops_convolve_11.detach().cpu().numpy()
        torch_convolve_11 = torch_convolve_11.detach().cpu().numpy()
        keops_convolve_12 = keops_convolve_12.detach().cpu().numpy()
        torch_convolve_12 = torch_convolve_12.detach().cpu().numpy()
        keops_dp_1 = keops_dp_1.detach().cpu().numpy()
        keops_dp_2 = keops_dp_2.detach().cpu().numpy()
        keops_dn_1 = keops_dn_1.detach().cpu().numpy()
        keops_dn_2 = keops_dn_2.detach().cpu().numpy()
        keops_da_1 = keops_da_1.detach().cpu().numpy()
        keops_da_2 = keops_da_2.detach().cpu().numpy()
        torch_dp_1 = torch_dp_1.detach().cpu().numpy()
        torch_dp_2 = torch_dp_2.detach().cpu().numpy()
        torch_dn_1 = torch_dn_1.detach().cpu().numpy()
        torch_dn_2 = torch_dn_2.detach().cpu().numpy()
        torch_da_1 = torch_da_1.detach().cpu().numpy()
        torch_da_2 = torch_da_2.detach().cpu().numpy()

        # Check for equality.
        self.assertTrue(np.allclose(keops_convolve_11, torch_convolve_11, rtol=1e-05, atol=1e-05))
        self.assertTrue(np.allclose(keops_convolve_12, torch_convolve_12, rtol=1e-05, atol=1e-05))
        self.assertTrue(np.allclose(keops_dp_1, torch_dp_1, rtol=1e-05, atol=1e-05))
        self.assertTrue(np.allclose(keops_dp_2, torch_dp_2, rtol=1e-05, atol=1e-05))
        self.assertTrue(np.allclose(keops_dn_1, torch_dn_1, rtol=1e-05, atol=1e-05))
        self.assertTrue(np.allclose(keops_dn_2, torch_dn_2, rtol=1e-05, atol=1e-05))
        self.assertTrue(np.allclose(keops_da_1, torch_da_1, rtol=1e-05, atol=1e-05))
        self.assertTrue(np.allclose(keops_da_2, torch_da_2, rtol=1e-05, atol=1e-05))

    def test_keops_and_torch_convolve_gradient_are_equal(self):
        # Parameters.
        kernel_width = 10.
        number_of_control_points = 10
        dimension = 3
        # tensor_scalar_type = torch.cuda.FloatTensor
        tensor_scalar_type = torch.FloatTensor

        # Set the global settings accordingly.
        Settings().dimension = dimension
        Settings().tensor_scalar_type = tensor_scalar_type

        # Instantiate the needed objects.
        keops_kernel = kernel_factory.factory(kernel_factory.Type.KEOPS, kernel_width)
        torch_kernel = kernel_factory.factory(kernel_factory.Type.TORCH, kernel_width)
        random_control_points_1 = torch.from_numpy(
            np.random.randn(number_of_control_points, dimension)).type(tensor_scalar_type).requires_grad_()
        random_control_points_2 = torch.from_numpy(
            np.random.randn(number_of_control_points, dimension)).type(tensor_scalar_type).requires_grad_()
        random_momenta_1 = torch.from_numpy(
            np.random.randn(number_of_control_points, dimension)).type(tensor_scalar_type).requires_grad_()
        random_momenta_2 = torch.from_numpy(
            np.random.randn(number_of_control_points, dimension)).type(tensor_scalar_type).requires_grad_()

        # Compute the desired forward quantities.
        keops_convolve_gradient_11 = keops_kernel.convolve_gradient(
            random_momenta_1, random_control_points_1)
        torch_convolve_gradient_11 = torch_kernel.convolve_gradient(
            random_momenta_1, random_control_points_1)
        keops_convolve_gradient_11_bis = keops_kernel.convolve_gradient(
            random_momenta_1, random_control_points_1, random_control_points_1, random_momenta_1)
        torch_convolve_gradient_11_bis = torch_kernel.convolve_gradient(
            random_momenta_1, random_control_points_1, random_control_points_1, random_momenta_1)
        keops_convolve_gradient_12 = keops_kernel.convolve_gradient(
            random_momenta_1, random_control_points_1, random_control_points_2, random_momenta_2)
        torch_convolve_gradient_12 = torch_kernel.convolve_gradient(
            random_momenta_1, random_control_points_1, random_control_points_2, random_momenta_2)

        # Compute the desired backward quantities.
        keops_total_12 = torch.dot(random_momenta_1.view(-1), keops_convolve_gradient_12.contiguous().view(-1))
        torch_total_12 = torch.dot(random_momenta_1.view(-1), torch_convolve_gradient_12.contiguous().view(-1))

        [keops_dcp_1, keops_dcp_2, keops_dmom_1, keops_dmom_2] = torch.autograd.grad(
            keops_total_12, [random_control_points_1, random_control_points_2, random_momenta_1, random_momenta_2])
        [torch_dcp_1, torch_dcp_2, torch_dmom_1, torch_dmom_2] = torch.autograd.grad(
            torch_total_12, [random_control_points_1, random_control_points_2, random_momenta_1, random_momenta_2])

        # Convert back to numpy.
        keops_convolve_gradient_11 = keops_convolve_gradient_11.detach().cpu().numpy()
        torch_convolve_gradient_11 = torch_convolve_gradient_11.detach().cpu().numpy()
        keops_convolve_gradient_11_bis = keops_convolve_gradient_11_bis.detach().cpu().numpy()
        torch_convolve_gradient_11_bis = torch_convolve_gradient_11_bis.detach().cpu().numpy()
        keops_convolve_gradient_12 = keops_convolve_gradient_12.detach().cpu().numpy()
        torch_convolve_gradient_12 = torch_convolve_gradient_12.detach().cpu().numpy()
        keops_dcp_1 = keops_dcp_1.detach().cpu().numpy()
        keops_dcp_2 = keops_dcp_2.detach().cpu().numpy()
        keops_dmom_1 = keops_dmom_1.detach().cpu().numpy()
        keops_dmom_2 = keops_dmom_2.detach().cpu().numpy()
        torch_dcp_1 = torch_dcp_1.detach().cpu().numpy()
        torch_dcp_2 = torch_dcp_2.detach().cpu().numpy()
        torch_dmom_1 = torch_dmom_1.detach().cpu().numpy()
        torch_dmom_2 = torch_dmom_2.detach().cpu().numpy()

        # Check for equality.
        self.assertTrue(np.allclose(keops_convolve_gradient_11_bis, keops_convolve_gradient_11_bis, rtol=1e-05, atol=1e-05))
        self.assertTrue(np.allclose(torch_convolve_gradient_11_bis, torch_convolve_gradient_11_bis, rtol=1e-05, atol=1e-05))
        self.assertTrue(np.allclose(keops_convolve_gradient_11, torch_convolve_gradient_11, rtol=1e-05, atol=1e-05))
        self.assertTrue(np.allclose(keops_convolve_gradient_12, torch_convolve_gradient_12, rtol=1e-05, atol=1e-05))
        self.assertTrue(np.allclose(keops_dcp_1, torch_dcp_1, rtol=1e-05, atol=1e-05))
        self.assertTrue(np.allclose(keops_dcp_2, torch_dcp_2, rtol=1e-05, atol=1e-05))
        self.assertTrue(np.allclose(keops_dmom_1, torch_dmom_1, rtol=1e-05, atol=1e-05))
        self.assertTrue(np.allclose(keops_dmom_2, torch_dmom_2, rtol=1e-05, atol=1e-05))
