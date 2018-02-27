Research code. Might have some instabilities !

# Installation instructions

## Requirements

- Python 3+
- VTK 7+
- Pytorch
- Cuda 8+ [optional]

## Advised procedure

- Install anaconda3: https://www.anaconda.com/download/
- Download the sources for VTK: https://www.vtk.org/download/
- Build VTK, given the path PATH_TO_VTK_FOLDER: `cd $PATH_TO_VTK_FOLDER && mkdir build && cd build && cmake -DVTK_PYTHON_VERSION=3.6 -DVTK_WRAP_PYTHON=ON -DModule_vtkPythonInterpreter=ON -DPYTHON_EXECUTABLE=.../anaconda3/bin/python3.6 -DPYTHON_INCLUDE_DIR=.../anaconda3/include/python3.6m -DPYTHON_LIBRARY=.../anaconda3/lib/libpython3.6m.so -DVTK_INSTALL_PYTHON_MODULE_DIR=.../anaconda3/lib/python3.6/site-packages ../ && make -j8`
- Add to your environnement: `export PATH=/home/alexandre.bone/anaconda3/bin:$PATH` and `export PYTHONPATH="path_to_vtk_folder/build/bin:path_to_vtk_folder/build/lib:path_to_vtk_folder/build/Wrapping/Python:${PYTHONPATH}"`
- Install Pytorch: http://pytorch.org/
- Test the installation of VTK and Pytorch: open a terminal, type `python`, then `import vtk` and `import torch`. If these do not raise any error, the dependencies are successfully installed.
- Clone Pydeformetrica: `git clone git@gitlab.icm-institute.org:alexandre_bone/pydeformetrica.git`
- Activate submodules: `git submodule init && git submodule update`
- Try to run an example, for example `cd pydeformetrica/examples/atlas/landmark/2d/skulls && python pydeformetrica/src/launch/deformetrica.py model.xml data_set.xml optimization_parameters.xml`
