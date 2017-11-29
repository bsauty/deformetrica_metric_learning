import os.path
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../../')

from pydeformetrica.src.support.utilities.singleton_pattern import Singleton

@Singleton
class GeneralSettings:

    """
    General settings, shared across the whole code.
    Singleton pattern.

    """

    def __init__(self):
        self.Dimension = 3