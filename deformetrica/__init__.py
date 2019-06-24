__version__ = '4.2.0rc'

# api
from api import Deformetrica

# core
from core import default, GpuMode

# models
import core.models as models

# model_tools
import core.model_tools.attachments as attachments
import core.model_tools.deformations as deformations
# import core.model_tools.manifolds as manifold

# io
import in_out as io

# kernels
import support.kernels as kernels

# utils
import support.utilities as utils
