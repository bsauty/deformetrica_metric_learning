import os.path
import sys
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../../../')
from pydeformetrica.src.support.utilities.general_settings import Settings


def write_2D_array(array, name):
    """
    Assuming 2-dim array here e.g. control points
    save_name = os.path.join(Settings().output_dir, name)
    np.savetxt(save_name, array)
    """
    save_name = os.path.join(Settings().output_dir, name)
    np.savetxt(save_name, array)

def write_momenta(array, name):
    """
    Saving an array has dim (numsubjects, numcps, dimension), using deformetrica format
    """
    save_name = os.path.join(Settings().output_dir, name)
    with open(save_name,"w") as f:
        f.write(str(len(array)) + " " + str(len(array[0])) + " " + str(len(array[0,0])) + "\n")
        for elt in array:
            f.write("\n")
            for elt1 in elt:
                for elt2 in elt1:
                    f.write(str(elt2)+" ")
                f.write("\n")

def read_momenta(name):
    """
    Loads a file containing momenta, old deformetrica syntax assumed
    """
    with open(name, "r") as f:
        lines = f.readlines()
        line0 = [int(elt) for elt in lines[0].split()]
        nbSubjects, nbControlPoints, dimension = line0[0], line0[1], line0[2]
        momenta = np.zeros((nbSubjects, nbControlPoints, dimension))
        lines = lines[1:]
        for i in range(nbSubjects):
            for c in range(nbControlPoints):
                foo = lines[1 + c].split()
                assert(len(foo) == dimension)
                foo = [float(elt) for elt in foo]
                momenta[i,c,:] = foo
            lines = lines[1+nbControlPoints:]
    return momenta


def read_2D_array(array, name):
    """
    Assuming 2-dim array here e.g. control points
    """
    return np.loadtxt(name)


