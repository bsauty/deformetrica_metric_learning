import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../../../../../')

import numpy as np
import torch

from pydeformetrica.src.core.model_tools.attachments.landmarks_attachments import *

class MultiObjectAttachment:

    ####################################################################################################################
    ### Constructor:
    ####################################################################################################################

    def __init__(self):
        # List of strings, e.g. 'varifold' or 'current'.
        self.attachment_types = []

        # List of kernel objects.
        self.kernels = []

    ####################################################################################################################
    ### Public methods:
    ####################################################################################################################

    def compute_weighted_distance(self, points1, multi_obj1, multi_obj2, kernelWidths, weights, objectNorms):
        """
        Takes two multiobjects and their new point positions to compute the distances
        """
        weightedDistance = 0.
        pos = 0
        assert len(multi_obj1.ObjectList) == len(multi_obj2.ObjectList), "Cannot compute distance between multi-objects which have different number of objects"
        for i, obj1 in enumerate(multi_obj1.ObjectList):
            obj2 = multi_obj2.ObjectList[i]
            if objectNorms[i] == 'Current':
                weightedDistance += weights[i] * CurrentDistance(points1[pos:pos+obj1.GetNumberOfPoints()],
                                                    obj1, obj2, kernel_width=kernelWidths[i])
            elif objectNorms[i] == 'Varifold':
                weightedDistance += weights[i] * VarifoldDistance(points1[pos:pos + obj1.GetNumberOfPoints()],
                                                         obj1, obj2, kernel_width=kernelWidths[i])
            else:
                assert False, "Please implement the distance you are trying to use :)"
            pos += obj1.GetNumberOfPoints()
        return weightedDistance


    ####################################################################################################################
    ### Private methods:
    ####################################################################################################################
