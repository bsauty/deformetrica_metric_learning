#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import os
from glob import glob
from os.path import splitext, basename

from setuptools import setup, find_packages

from src import __version__

try:  # for pip >= 10
    from pip._internal.req import parse_requirements
except ImportError:  # for pip <= 9.0.3
    from pip.req import parse_requirements


def str_to_bool(s):
    if s is None:
        return False

    assert isinstance(s, str), 'given argument must be a string'
    if s.lower() in ['true', '1', 'yes']:
        return True
    elif s.lower() in ['false', '0', 'no']:
        return False
    else:
        raise LookupError


# build gui by default
build_gui = str_to_bool(os.environ['BUILD_GUI']) if 'BUILD_GUI' in os.environ else True

print('Building Deformetrica version ' + __version__ + ', BUILD_GUI=' + str(build_gui))


def build_deformetrica():
    print('build_deformetrica()')
    setup(
        name='deformetrica',
        version=__version__,
        url='http://www.deformetrica.org',
        description='Software for the statistical analysis of 2D and 3D shape data.',
        long_description=open('README.md', encoding='utf-8').read(),
        author='ARAMIS Lab',
        maintainer='Deformetrica developers',
        maintainer_email='deformetrica.team@gmail.com',
        license='INRIA license',
        package_dir={'deformetrica': 'src'},
        packages=find_packages(exclude=['gui']),  # exclude gui
        py_modules=[splitext(basename(path))[0] for path in glob('*.py')],
        # data_files=[('src', ['LICENSE.txt'])],
        include_package_data=True,
        zip_safe=False,
        entry_points={
            'console_scripts': ['deformetrica=deformetrica:main'],  # CLI
        },
        classifiers=[
            'Framework :: Deformetrica',
            'Development Status :: ' + __version__,
            'Environment :: Console',
            'Operating System :: OS Independent',
            'Programming Language :: Python',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.5',
            'Programming Language :: Python :: 3.6',
            'Topic :: Scientific/Engineering',
            'Topic :: Scientific/Engineering :: Bio-Informatics',
            'Topic :: Software Development :: Libraries'
        ],
        install_requires=[
            'cmake>=3.10',
            'numpy>=1.10',
            'h5py>=2.8',    # fix: h5py conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated
            'gputil>=1.3',
            'pykeops==0.0.14'
        ]
    )


def build_deformetrica_and_gui():
    print('build_deformetrica_and_gui()')
    setup(
        name='deformetrica-gui',
        version=__version__,
        url='http://www.deformetrica.org',
        description='Software for the statistical analysis of 2D and 3D shape data.',
        long_description=open('README.md', encoding='utf-8').read(),
        author='ARAMIS Lab',
        maintainer='Deformetrica developers',
        maintainer_email='deformetrica.team@gmail.com',
        license='INRIA license',
        package_dir={'deformetrica-gui': 'src.gui'},
        packages=find_packages(),
        py_modules=[splitext(basename(path))[0] for path in glob('*.py')],
        package_data={'': ['*.json', '*.png']},
        include_package_data=True,
        # data_files=[('', ['LICENSE.txt'])],
        zip_safe=False,
        entry_points={
            'console_scripts': ['deformetrica=deformetrica:main'],  # CLI
            'gui_scripts': ['deformetrica-gui=gui.__main__:main']   # GUI
        },
        classifiers=[
            'Framework :: Deformetrica',
            'Development Status :: ' + __version__,
            'Environment :: Console',
            'Operating System :: OS Independent',
            'Programming Language :: Python',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.5',
            'Programming Language :: Python :: 3.6',
            'Topic :: Scientific/Engineering',
            'Topic :: Scientific/Engineering :: Bio-Informatics',
            'Topic :: Software Development :: Libraries'
        ],
        install_requires=[
            'cmake>=3.10',
            'numpy>=1.10',
            'h5py>=2.8',  # fix: h5py conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated
            'gputil>=1.3',
            'pykeops==0.0.14',
            'PyQt5>=5.11'
        ]
    )


if build_gui:
    build_deformetrica_and_gui()
else:
    build_deformetrica()
