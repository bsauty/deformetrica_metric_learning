import unittest
import torch

import support.kernel as kernel_factory


class KernelFactory(unittest.TestCase):

    def test_instantiate_abstract_class(self):
        with self.assertRaises(TypeError):
            kernel_factory.AbstractKernel()

    def test_unknown_kernel_string(self):
        with self.assertRaises(TypeError):
            kernel_factory.factory('unknown_type')

    def test_all_kernel_factory(self):
        for k in kernel_factory.Type:
            print("testing kernel=", k)
            instance = kernel_factory.factory(k, 0)
            self.__isKernelValid(instance)

    def test_kernel_factory_from_string(self):
        for k in ['no_kernel', 'no-kernel', 'exact', 'cuda_exact', 'cuda exact']:
            print("testing kernel=", k)
            instance = kernel_factory.factory(k, 0)
            self.__isKernelValid(instance)

    def __isKernelValid(self, instance):
        if instance is not None:
            self.assertIsInstance(instance, kernel_factory.AbstractKernel)
            self.assertEqual(instance.kernel_width, 0)


class Kernel(unittest.TestCase):

    def test_myTest(self):
        instance = kernel_factory.factory(kernel_factory.Type.CudaExactTorchKernel, 0)

        x = torch.tensor([10, 1])
        x.fill_(2)

        y = x.clone()
        p = torch.ones([10, 1])

        res = instance.convolve(x, y, p)
        print(res)
