import numpy as np
import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../../../../../')
from pydeformetrica.src.core.model_tools.attachments.landmarks_attachments import *


def ComputeMultiObjectWeightedDistance(points1, multi_obj1, multi_obj2, kernelWidths, weights, objectNorms):
    """
    Takes two multiobjects and their new point positions to compute the distances
    """
    distance = 0.
    pos = 0
    assert len(multi_obj1.ObjectList) == len(multi_obj2.ObjectList), "Cannot compute distance between multi-objects which have different number of objects"
    for i, obj1 in enumerate(multi_obj1.ObjectList):
        obj2 = multi_obj2.ObjectList[i]
        if objectNorms[i] == 'Current':
            distance += weights[i] * CurrentDistance(points1[pos:pos+obj1.GetNumberOfPoints()],
                                                obj1, obj2, kernel_width=kernelWidths[i])
        elif objectNorms[i] == 'Varifold':
            assert False, "Varifold dist to be implemented"
        else:
            assert False, "Please implement the distance you are trying to use :)"
        pos += obj1.GetNumberOfPoints()
    return distance
