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
from core.models.model_functions import create_regular_grid_of_points
from core.model_tools.deformations.exponential import Exponential

path_to_small_surface_mesh_1 = 'data/landmark/surface_mesh/hippocampus_500_cells_1.vtk'
path_to_small_surface_mesh_2 = 'data/landmark/surface_mesh/hippocampus_500_cells_2.vtk'
path_to_large_surface_mesh_1 = 'data/landmark/surface_mesh/hippocampus_5000_cells_1.vtk'
path_to_large_surface_mesh_2 = 'data/landmark/surface_mesh/hippocampus_5000_cells_2.vtk'


class ProfileDeformations:
    def __init__(self, kernel_type, kernel_width, kernel_device='CPU', full_cuda=False, data_size='small'):

        np.random.seed(42)

        if full_cuda:
            Settings().tensor_scalar_type = torch.cuda.FloatTensor
        else:
            Settings().tensor_scalar_type = torch.FloatTensor

        self.exponential = Exponential()
        self.exponential.kernel = kernel_factory.factory(kernel_type, kernel_width, kernel_device)
        self.exponential.number_of_time_points = 11
        self.exponential.set_use_rk2_for_shoot(False)
        self.exponential.set_use_rk2_for_flow(False)

        reader = DeformableObjectReader()
        if data_size == 'small':
            surface_mesh = reader.create_object(path_to_small_surface_mesh_1, 'SurfaceMesh')
        elif data_size == 'large':
            surface_mesh = reader.create_object(path_to_large_surface_mesh_1, 'SurfaceMesh')

        control_points = create_regular_grid_of_points(surface_mesh.bounding_box, kernel_width)
        momenta = np.random.randn(control_points.shape)
        self.exponential.set_initial_template_points(Settings().tensor_scalar_type(surface_mesh.get_points()))
        self.exponential.set_initial_control_points(Settings().tensor_scalar_type(control_points))
        self.exponential.set_initial_momenta(Settings().tensor_scalar_type(momenta))

    def run(self):
        self.exponential.update()


class BenchRunner:
    def __init__(self, kernel, kernel_width, kernel_device):

        self.obj = ProfileAttachments(kernel, kernel_width, kernel_device)

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
    devices = ['CPU']
    types = ['TODO']
    setups = []

    for k, t in [(k, t) for k in kernels for t in devices]:
        bench_setup = '''
from __main__ import BenchRunner
import torch
bench = BenchRunner('{kernel}', 1.0, '{device}')
'''.format(kernel=k, device=t)

        setups.append({'kernel': k, 'device': t, 'bench_setup': bench_setup})
    return setups, kernels, devices, len(devices)


if __name__ == "__main__":
    import timeit

    results = []

    build_setup, kernels, devices, tensor_size_len = build_setup()

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
    for t, k in [(t, k) for t in devices for k in kernels]:
        extracted_data = [r['max'] for r in results if r['setup']['device'] == t if r['setup']['kernel'] == k]
        assert(len(extracted_data) == len(index))

        ax.bar(index + bar_width * i, extracted_data, bar_width, alpha=opacity, label=t + ':' + k)
        i = i+1

    # bar1 = ax.bar(index, cpu_res, bar_width, alpha=0.4, color='b', label='cpu')
    # bar2 = ax.bar(index + bar_width, cuda_res, bar_width, alpha=0.4, color='g', label='cuda')

    ax.set_xlabel('Tensor size')
    ax.set_ylabel('Runtime (s)')
    ax.set_title('Runtime by device/size')
    ax.set_xticks(index + bar_width * ((len(kernels)*len(devices))/2) - bar_width/2)
    ax.set_xticklabels([r['setup']['device'] for r in results])
    ax.legend()

    fig.tight_layout()

    plt.show()
