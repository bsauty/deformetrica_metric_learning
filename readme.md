Research code. Might have some instabilities !

# Installation instructions

It is recommened to use an anaconda virtual environment. See https://www.anaconda.com/download for setup instructions.


## [OPTIONAL] Anaconda environment
The provided anaconda virtual environment comes pre-installed with PyTorch WITHOUT CUDA support. 
If CUDA support is required, PyTorch must be installed from source. Instructions at: https://github.com/pytorch/pytorch#from-source

- Install anaconda3: https://www.anaconda.com/download/
- Create the conda environment: `conda env create -f environment.yml`
- Enter the virtual environment: `source activate deformetrica`

## Requirements
The following requirements are automagically installed when using the anaconda virtual environment.

- Python 3+
- Pytorch
- matplotlib
- nibabel
- Cuda 8+ [optional]

## Advised procedure

- Follow the above steps to create and enter an anaconda environment
- Clone Pydeformetrica: `git clone git@gitlab.icm-institute.org:alexandre_bone/pydeformetrica.git`
- For CUDA-compliance only [optional]:
    - Activate submodules: `git submodule init && git submodule update`
    - Checkout to the correct libkp submodule version: `cd libs/libkp && git checkout 2c40c9b4 && cd ../../`
    - Weird but necessary hack: `vim libs/libkp/__init__.py` and delete (or comment) the two lines
    - Compile the necessary files: `bash libs/libkp/python/makefile.sh && python libs/libkp/python/examples/generic_example.py`
- Try to run an example, for example `cd pydeformetrica/examples/atlas/landmark/2d/skulls && python pydeformetrica/src/launch/deformetrica.py model.xml data_set.xml optimization_parameters.xml`
