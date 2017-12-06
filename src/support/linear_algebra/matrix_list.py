import os.path
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../../../')


class MatrixList:

    """
    Matrix list.
    The MatrixList is a wrapper class for list of numpy array, adding basic operations.

    """

    # Constructor.
    def __init__(self, list=[]):
        self.RawMatrixList = list

    # Push back operator
    def append(self, item):
        self.RawMatrixList.append(item)

    # Access operator.
    def __getitem__(self, item):
        return self.RawMatrixList[item]

    # Addition operator.
    def __add__(self, other):
        assert(len(self.RawMatrixList) == len(other))
        out = self.RawMatrixList
        for k in range(len(self.RawMatrixList)):
            assert(out[k].shape == other[k].shape)
            out[k] += other[k]
        return out

    def Concatenate(self):
        """
        Here we suppose that the RawMatrixList carries elements which share the same last dimension (e.g. the list of list of points of vtks)
        """
        return np.concat(self.RawMatrixList)
