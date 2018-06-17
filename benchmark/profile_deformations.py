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
from core.models.model_functions import create_regular_grid_of_points, remove_useless_control_points
from core.model_tools.deformations.exponential import Exponential
from core.observations.deformable_objects.landmarks.surface_mesh import SurfaceMesh
from core.observations.deformable_objects.image import Image

path_to_small_surface_mesh_1 = 'data/landmark/surface_mesh/hippocampus_500_cells_1.vtk'
path_to_small_surface_mesh_2 = 'data/landmark/surface_mesh/hippocampus_500_cells_2.vtk'
path_to_large_surface_mesh_1 = 'data/landmark/surface_mesh/hippocampus_5000_cells_1.vtk'
path_to_large_surface_mesh_2 = 'data/landmark/surface_mesh/hippocampus_5000_cells_2.vtk'


class ProfileDeformations:
    def __init__(self, kernel_type, kernel_device='CPU', use_cuda=False, data_type='landmark', data_size='small'):

        np.random.seed(42)
        kernel_width = 10.

        if use_cuda:
            Settings().tensor_scalar_type = torch.cuda.FloatTensor
        else:
            Settings().tensor_scalar_type = torch.FloatTensor

        self.exponential = Exponential()
        self.exponential.kernel = kernel_factory.factory(kernel_type, kernel_width, kernel_device)
        self.exponential.number_of_time_points = 11
        self.exponential.set_use_rk2_for_shoot(False)
        self.exponential.set_use_rk2_for_flow(False)

        self.template = DeformableMultiObject()
        if data_type.lower() == 'landmark':
            reader = DeformableObjectReader()
            if data_size == 'small':
                surface_mesh = reader.create_object(path_to_small_surface_mesh_1, 'SurfaceMesh')
                control_points = create_regular_grid_of_points(surface_mesh.bounding_box, kernel_width)
            elif data_size == 'large':
                surface_mesh = reader.create_object(path_to_large_surface_mesh_1, 'SurfaceMesh')
                control_points = create_regular_grid_of_points(surface_mesh.bounding_box, kernel_width)
            else:
                surface_mesh = SurfaceMesh()
                surface_mesh.set_points(np.random.randn(int(data_size), 3))
                control_points = np.random.randn(int(data_size) // 10, 3)
            self.template.object_list.append(surface_mesh)

        elif data_type.lower() == 'image':
            image = Image()
            image.set_intensities(np.random.randn(int(data_size), int(data_size), int(data_size)))
            image.set_affine(np.eye(4))
            image.downsampling_factor = 5.
            image.update()
            control_points = create_regular_grid_of_points(image.bounding_box, kernel_width)
            control_points = remove_useless_control_points(control_points, image, kernel_width)
            self.template.object_list.append(image)

        else:
            raise RuntimeError('Unknown data_type argument. Choose between "landmark" or "image".')

        self.template.update()
        self.template_data = {key: Settings().tensor_scalar_type(value)
                                             for key, value in self.template.get_data().items()}

        self.exponential.set_initial_template_points(
            {key: Settings().tensor_scalar_type(value)
             for key, value in self.template.get_points().items()})
        self.exponential.set_initial_control_points(Settings().tensor_scalar_type(control_points))
        self.exponential.set_initial_momenta(Settings().tensor_scalar_type(np.random.randn(*control_points.shape)))

    def run(self):
        self.exponential.update()
        deformed_points = self.exponential.get_template_points()
        deformed_data = self.template.get_deformed_data(deformed_points, self.template_data)


class BenchRunner:
    def __init__(self, kernel, use_cuda, method_to_run):
        self.obj = ProfileDeformations(kernel[0], kernel[1], use_cuda, method_to_run[0], method_to_run[1])
        self.to_run = getattr(self.obj, method_to_run[2])

        # run once for warm-up: cuda pre-compile with keops
        self.run()
        # print('BenchRunner::__init()__ done')

    """ The method that is to be benched must reside within the run() method """

    def run(self):
        self.to_run()

        print('.', end='', flush=True)  # uncomment to show progression

    def __exit__(self):
        print('BenchRunner::__exit()__')


def build_setup():
    kernels = [('torch', 'CPU')]
    method_to_run = [('landmark', '50', 'run')]
    setups = []

    for k, m in [(k, m) for k in kernels for m in method_to_run]:
        bench_setup = '''
from __main__ import BenchRunner
import torch
bench = BenchRunner({kernel}, {method_to_run})
'''.format(kernel=k, method_to_run=m)

        setups.append({'kernel': k, 'method_to_run': m, 'bench_setup': bench_setup})
    return setups, kernels, method_to_run


if __name__ == "__main__":
    import timeit

    results = []

    build_setup, kernels, method_to_run = build_setup()

    # prepare and run bench
    for setup in build_setup:
        print('running setup ' + str(setup))

        res = {}
        res['setup'] = setup
        res['data'] = timeit.repeat("bench.run()", number=10, repeat=3, setup=setup['bench_setup'])
        res['min'] = min(res['data'])
        res['max'] = max(res['data'])
        res['mean'] = sum(res['data']) / float(len(res['data']))

        print('')
        print(res['data'])
        results.append(res)

    fig, ax = plt.subplots()
    # plt.ylim(ymin=0)
    # ax.set_yscale('log')

    index = np.arange(len(method_to_run))
    bar_width = 0.2
    opacity = 0.4

    # extract data from raw data and add to plot
    i = 0
    for k in [(k) for k in kernels]:
        extracted_data = [r['max'] for r in results
                          if r['setup']['kernel'] == k]

        assert (len(extracted_data) > 0)
        assert (len(extracted_data) == len(index))

        ax.bar(index + bar_width * i, extracted_data, bar_width, alpha=opacity, label=k[0] + ':' + k[1])
        i = i + 1

    # bar1 = ax.bar(index, cpu_res, bar_width, alpha=0.4, color='b', label='cpu')
    # bar2 = ax.bar(index + bar_width, cuda_res, bar_width, alpha=0.4, color='g', label='cuda')

    ax.set_xlabel('TODO')
    ax.set_ylabel('Runtime (s)')
    ax.set_title('TODO')
    ax.set_xticks(index + bar_width * ((len(kernels)) / 2) - bar_width / 2)
    ax.set_xticklabels([r['setup']['method_to_run'][1] for r in results])
    ax.legend()

    # for tick in ax.get_xticklabels():
    #     tick.set_rotation(45)

    fig.tight_layout()

    plt.show()
