import numpy as np
import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../../../../../')
from pydeformetrica.src.core.model_tools.attachments.landmarks_attachments import *


def ComputeMultiObjectDistance(points1, points2, multi_obj1, multi_obj2):
    """
    Takes two multiobjects and their new point positions to compute the distances
    This method is not fully done, TODO : use the xml values to get the right distance for each object and the right kernel width !
    """
    distance = 0.
    pos1 = 0
    pos2 = 0
    assert len(multi_obj1.ObjectList) == len(multi_obj2.ObjectList), "Cannot compute distance between multi-objects which have different number of objects"
    for i, obj1 in enumerate(multi_obj1.ObjectList):
        obj2 = multi_obj2.ObjectList[i]
        distance += OrientedSurfaceDistance(points1[pos1:pos1+obj1.GetNumberOfPoints()],
                                                obj1, obj2, kernel_width=10.)
    return distance
