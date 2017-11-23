import os.path
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../../../')

from pydeformetrica.src.core.models.deterministic_atlas import DeterministicAtlas
from pydeformetrica.src.core.estimators.gradient_ascent import GradientAscent
from pydeformetrica.src.io.xml_parameters import XmlParameters
from pydeformetrica.src.support.utilities.general_settings import GeneralSettings


"""
Read command line, read xml files, create deformable multi objects.

"""

assert len(sys.argv) >= 4, "Usage: " + sys.argv[0] + " <model.xml> <data_set.xml> <optimization_parameters.xml>"
modelXmlPath = sys.argv[1]
datasetXmlPath = sys.argv[2]
optimizationParametersXmlPath = sys.argv[3]

xmlParameters = XmlParameters()
xmlParameters.ReadAllXmls(modelXmlPath, datasetXmlPath, optimizationParametersXmlPath)



model = DeterministicAtlas()
estimator = GradientAscent()

print("Ok.")
