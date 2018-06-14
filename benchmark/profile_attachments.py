#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import numpy as np
import torch

import support.kernels as kernel_factory
from in_out.deformable_object_reader import DeformableObjectReader
from core.model_tools.attachments.multi_object_attachment import MultiObjectAttachment
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

        self.small_surface_mesh_1 = DeformableObjectReader.read_vtk_file(path_to_small_surface_mesh_1)
        self.small_surface_mesh_1_points = {key: Settings().tensor_scalar_type(value)
                                            for key, value in self.small_surface_mesh_1.get_points()}
        self.small_surface_mesh_2 = DeformableObjectReader.read_vtk_file(path_to_small_surface_mesh_2)

        self.large_surface_mesh_1 = DeformableObjectReader.read_vtk_file(path_to_large_surface_mesh_1)
        self.large_surface_mesh_1_points = {key: Settings().tensor_scalar_type(value)
                                            for key, value in self.large_surface_mesh_1.get_points()}
        self.large_surface_mesh_2 = DeformableObjectReader.read_vtk_file(path_to_large_surface_mesh_2)

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
