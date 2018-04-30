Research code. Might have some instabilities !

# Installation instructions

## Requirements

- Python 3+
- VTK 7+
- Pytorch
- Cuda 8+ [optional]

## Advised procedure

- Install anaconda3: https://www.anaconda.com/download/
- Install Pytorch: http://pytorch.org/
- Test the installation of Pytorch: open a terminal, type `python`, then `import torch`. If this do not raise any error, pytorch is successfully installed.
- Clone Pydeformetrica: `git clone git@gitlab.icm-institute.org:alexandre_bone/pydeformetrica.git`
- For CUDA-compliance only [optional]:
    - Activate submodules: `git submodule init && git submodule update`
    - Checkout to the correct libkp submodule version: `cd libs/libkp && git checkout 2c40c9b4 && cd ../../`
    - Weird but necessary hack: `vim libs/libkp/__init__.py` and delete (or comment) the two lines
    - Compile the necessary files: `bash libs/libkp/python/makefile.sh && python libs/libkp/python/examples/generic_example.py`
- Try to run an example, for example `cd pydeformetrica/examples/atlas/landmark/2d/skulls && python pydeformetrica/src/launch/deformetrica.py model.xml data_set.xml optimization_parameters.xml`
