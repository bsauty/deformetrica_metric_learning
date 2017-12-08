import os.path
import sys
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../../../')
from pydeformetrica.src.support.utilities.general_settings import GeneralSettings


def saveArray(array, name):
    """
    Assuming 2-dim array here e.g. control points
    save_name = os.path.join(GeneralSettings.Instance().OutputDir, name)
    np.savetxt(save_name, array)
    """
    save_name = os.path.join(GeneralSettings.Instance().OutputDir, name)
    np.savetxt(save_name, array)

def saveMomenta(array, name):
    """
    Saving an array has dim (numsubjects, numcps, dimension), using deformetrica format
    """
    save_name = os.path.join(GeneralSettings.Instance().OutputDir, name)
    with open(save_name,"w") as f:
        f.write(str(len(array)) + " " + str(len(array[0])) + " " + str(len(array[0][0])) + "\n")
        for elt in array:
            f.write("\n")
            for elt1 in elt:
                for elt2 in elt1:
                    f.write(str(elt2)+" ")
                f.write("\n")

def loadArray(array, name):
    """
    Assuming 2-dim array here e.g. control points
    """
    return np.loadtxt(name)
