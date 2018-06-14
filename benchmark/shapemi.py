#!/usr/bin/env python
# -*- encoding: utf-8 -*-


"""

ShapeMI at MICCAI 2018
https://shapemi.github.io/


Benchmark CPU vs GPU on small (500 points) and large (5000 points) meshes.

"""


import os
import matplotlib.pyplot as plt
import numpy as np
import support.kernels as kernel_factory
import torch

from in_out.deformable_object_reader import DeformableObjectReader
from core.model_tools.attachments.multi_object_attachment import MultiObjectAttachment
from core.observations.deformable_objects.deformable_multi_object import DeformableMultiObject
from support.utilities.general_settings import Settings

path_to_small_surface_mesh_1 = 'data/landmark/surface_mesh/hippocampus_500_cells_1.vtk'
path_to_small_surface_mesh_2 = 'data/landmark/surface_mesh/hippocampus_500_cells_2.vtk'
path_to_large_surface_mesh_1 = 'data/landmark/surface_mesh/hippocampus_5000_cells_1.vtk'
path_to_large_surface_mesh_2 = 'data/landmark/surface_mesh/hippocampus_5000_cells_2.vtk'


class ProfileAttachments:
    def __init__(self, kernel_type, kernel_width, tensor_scalar_type=torch.FloatTensor):
        Settings().tensor_scalar_type = tensor_scalar_type

        self.multi_object_attachment = MultiObjectAttachment()
        self.kernel = kernel_factory.factory(kernel_type, kernel_width)

        reader = DeformableObjectReader()
        self.small_surface_mesh_1 = DeformableMultiObject()
        self.small_surface_mesh_1.object_list.append(reader.create_object(path_to_small_surface_mesh_1, 'SurfaceMesh'))
        self.small_surface_mesh_1.update()

        self.small_surface_mesh_2 = DeformableMultiObject()
        self.small_surface_mesh_2.object_list.append(reader.create_object(path_to_small_surface_mesh_2, 'SurfaceMesh'))
        self.small_surface_mesh_2.update()

        self.large_surface_mesh_1 = DeformableMultiObject()
        self.large_surface_mesh_1.object_list.append(reader.create_object(path_to_large_surface_mesh_1, 'SurfaceMesh'))
        self.large_surface_mesh_1.update()

        self.large_surface_mesh_2 = DeformableMultiObject()
        self.large_surface_mesh_2.object_list.append(reader.create_object(path_to_large_surface_mesh_2, 'SurfaceMesh'))
        self.large_surface_mesh_2.update()

        self.small_surface_mesh_1_points = {key: Settings().tensor_scalar_type(value) 
                                            for key, value in self.small_surface_mesh_1.get_points().items()}
        self.large_surface_mesh_1_points = {key: Settings().tensor_scalar_type(value)
                                            for key, value in self.large_surface_mesh_1.get_points().items()}

    def profile_small_surface_mesh_current_attachment(self):
        self.multi_object_attachment._current_distance(
            self.small_surface_mesh_1_points, self.small_surface_mesh_1, self.small_surface_mesh_2, self.kernel)

    def profile_large_surface_mesh_current_attachment(self):
        self.multi_object_attachment._current_distance(
            self.large_surface_mesh_1_points, self.large_surface_mesh_1, self.large_surface_mesh_2, self.kernel)

    def profile_small_surface_mesh_varifold_attachment(self):
        self.multi_object_attachment._varifold_distance(
            self.small_surface_mesh_1_points, self.small_surface_mesh_1, self.small_surface_mesh_2, self.kernel)

    def profile_large_surface_mesh_varifold_attachment(self):
        self.multi_object_attachment._varifold_distance(
            self.large_surface_mesh_1_points, self.large_surface_mesh_1, self.large_surface_mesh_2, self.kernel)

    def hello(self):
        pass


class BenchRunner:
    def __init__(self, kernel, kernel_width, tensor_scalar_type):

        self.obj = ProfileAttachments(kernel, kernel_width, tensor_scalar_type)

        # run once for warm-up: cuda pre-compile with keops
        self.obj.profile_small_surface_mesh_current_attachment()
        # print('BenchRunner::__init()__ done')

    """ The method that is to be benched must reside within the run() method """
    def run(self):
        # TODO: use current_distance(...)
        self.obj.profile_small_surface_mesh_current_attachment()

        print('.', end='', flush=True)    # uncomment to show progression

    def __exit__(self):
        print('BenchRunner::__exit()__')


def build_setup():
    kernels = ['torch', 'keops']
    tensor_scalar_type = ['torch.FloatTensor']
    types = ['TODO']
    setups = []

    for k, t in [(k, t) for k in kernels for t in tensor_scalar_type]:
        bench_setup = '''
from __main__ import BenchRunner
import torch
bench = BenchRunner('{kernel}', 1.0, {tensor_scalar_type})
'''.format(kernel=k, tensor_scalar_type=t)

        setups.append({'kernel': k, 'tensor_scalar_type': t, 'bench_setup': bench_setup})
    return setups, kernels, tensor_scalar_type, len(tensor_scalar_type)


if __name__ == "__main__":
    import timeit

    results = []

    build_setup, kernels, tensor_scalar_type, tensor_size_len = build_setup()

    # prepare and run bench
    for setup in build_setup:
        print('running setup ' + str(setup))

        res = {}
        res['setup'] = setup
        res['data'] = timeit.repeat("bench.run()", number=1, repeat=3, setup=setup['bench_setup'])
        res['min'] = min(res['data'])
        res['max'] = max(res['data'])

        print('')
        print(res)
        results.append(res)

    # cpu_res = [r['max'] for r in results if r['setup']['device'] == 'cpu']
    # cuda_res = [r['max'] for r in results if r['setup']['device'] == 'cuda:0']
    # assert(len(cpu_res) == len(cuda_res))

    fig, ax = plt.subplots()
    # plt.ylim(ymin=0)
    # ax.set_yscale('log')

    index = np.arange(tensor_size_len)
    bar_width = 0.2
    opacity = 0.4

    # extract data from raw data and add to plot
    i = 0
    for t, k in [(t, k) for t in tensor_scalar_type for k in kernels]:
        extracted_data = [r['max'] for r in results if r['setup']['tensor_scalar_type'] == t if r['setup']['kernel'] == k]
        assert(len(extracted_data) == len(index))

        ax.bar(index + bar_width * i, extracted_data, bar_width, alpha=opacity, label=t + ':' + k)
        i = i+1

    # bar1 = ax.bar(index, cpu_res, bar_width, alpha=0.4, color='b', label='cpu')
    # bar2 = ax.bar(index + bar_width, cuda_res, bar_width, alpha=0.4, color='g', label='cuda')

    ax.set_xlabel('Tensor size')
    ax.set_ylabel('Runtime (s)')
    ax.set_title('Runtime by device/size')
    ax.set_xticks(index + bar_width * ((len(kernels)*len(tensor_scalar_type))/2) - bar_width/2)
    ax.set_xticklabels([r['setup']['tensor_scalar_type'] for r in results])
    ax.legend()

    fig.tight_layout()

    plt.show()
