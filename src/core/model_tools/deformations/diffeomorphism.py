import os.path
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../../../../')


class Diffeomorphism:

    """
    Control-point-based LDDMM diffeomophism, that transforms the template objects according to initial control points
    and momenta parameters.
    See "Morphometry of anatomical shape complexes with dense deformations and sparse parameters",
    Durrleman et al. (2013).

    """

    # Constructor.
    def __init__(self):
        self.KernelType = None
        self.KernelWidth = None
        self.NumberOfTimePoints = None

