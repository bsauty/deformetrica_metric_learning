import os
from .deformetrica import Deformetrica

try:
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'VERSION'), encoding='utf-8') as f:
        __version__ = f.read()
except IOError:
    __version__ = '0.0.0'
