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
        torch.set_printoptions(precision=30)    # for more precision when printing tensor

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
        self.kernel_instance = kernel_factory.factory(kernel_factory.Type.TorchCudaKernel,
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

    # @unittest.skip('TODO')  # TODO: res defers depending on machine/cpu/gpu
    def test_convolve_gradient(self):
        expected_convolve_gradient_res = torch.tensor([
            [-1.623382449150085449218750000000, -1.212645649909973144531250000000, 1.440739274024963378906250000000],
            [-1.414733648300170898437500000000, 1.848072290420532226562500000000, -0.102501690387725830078125000000],
            [1.248104929924011230468750000000, 0.059575233608484268188476562500, -1.860013127326965332031250000000],
            [1.790011286735534667968750000000, -0.695001900196075439453125000000, 0.521775543689727783203125000000]])

        res = self.kernel_instance.convolve_gradient(self.x, self.x)
        self._assert_tensor_close(res, expected_convolve_gradient_res)

    # @unittest.skipIf(not torch.cuda.is_available(), 'cuda is not available')
    @unittest.skip('TODO')  # TODO: res defers depending on machine/cpu/gpu
    def test_convolve_gpu(self):
        res = self.kernel_instance.convolve(self.x, self.y, self.p, backend='GPU')
        self._assert_tensor_close(res, self.expected_convolve_res)

    # @unittest.skipIf(not torch.cuda.is_available(), 'cuda is not available')
    @unittest.skip('TODO')    # TODO: res defers depending on machine/cpu/gpu
    def test_convolve_gradient_gpu(self):
        expected_convolve_gradient_res = torch.tensor([
            [0.575284719467163085937500000000, 1.041690707206726074218750000000, 0.561024427413940429687500000000],
            [0.876295149326324462890625000000, 1.767178058624267578125000000000, 1.534361839294433593750000000000],
            [1.244411468505859375000000000000, 2.220475196838378906250000000000, 1.913318395614624023437500000000],
            [1.194657802581787109375000000000, 2.259799003601074218750000000000, 1.833473324775695800781250000000]])

        res = self.kernel_instance.convolve_gradient(self.x, self.x, backend='GPU')
        self._assert_tensor_close(res, expected_convolve_gradient_res)

