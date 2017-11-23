import os.path
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../../../../')

from pydeformetrica.src.core.observations.landmark import Landmark

class SurfaceMesh(Landmark):

    """
    Triangular mesh.

    """

