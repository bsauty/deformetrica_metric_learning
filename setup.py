#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import os
from glob import glob
from os.path import splitext, basename

from setuptools import setup, find_packages

import api

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


print('Building Deformetrica version ' + api.__version__)


def build_deformetrica():
    print('build_deformetrica()')
    setup(
        name='deformetrica',
        version=api.__version__,
        url='http://www.deformetrica.org',
        description='Software for the statistical analysis of 2D and 3D shape data.',
        long_description=open('README.md', encoding='utf-8').read(),
        author='ARAMIS Lab',
        maintainer='Deformetrica developers',
        maintainer_email='deformetrica.team@gmail.com',
        license='INRIA license',
        package_dir={'': 'src'},
        packages=find_packages('src', exclude=['gui']),  # exclude gui
        py_modules=[splitext(basename(path))[0] for path in glob('src/*.py')],
        # data_files=[('src', ['LICENSE.txt'])],
        include_package_data=True,
        zip_safe=False,
        entry_points={
            'console_scripts': ['deformetrica=deformetrica:main'],  # CLI
        },
        classifiers=[
            'Framework :: Deformetrica',
            'Development Status :: ' + api.__version__,
            'Environment :: Console',
            'Operating System :: OS Independent',
            'Programming Language :: Python',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.5',
            'Programming Language :: Python :: 3.6',
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
        name='deformetrica',
        version=api.__version__,
        url='http://www.deformetrica.org',
        description='Software for the statistical analysis of 2D and 3D shape data.',
        long_description=open('README.md', encoding='utf-8').read(),
        author='ARAMIS Lab',
        maintainer='Deformetrica developers',
        maintainer_email='deformetrica.team@gmail.com',
        license='INRIA license',
        package_dir={'': 'src'},
        packages=find_packages('src'),
        py_modules=[splitext(basename(path))[0] for path in glob('src/*.py')],
        package_data={'': ['*.json', '*.png']},
        include_package_data=True,
        # data_files=[('', ['LICENSE.txt'])],
        zip_safe=False,
        entry_points={
            'console_scripts': ['deformetrica=deformetrica:main'],  # CLI
            'gui_scripts': ['deformetrica-gui=gui.__main__:main']  # GUI
        },
        classifiers=[
            'Framework :: Deformetrica',
            'Development Status :: ' + api.__version__,
            'Environment :: Console',
            'Operating System :: OS Independent',
            'Programming Language :: Python',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.5',
            'Programming Language :: Python :: 3.6',
        ],
        install_requires=[
            'cmake>=3.10',
            'numpy>=1.10',
            'h5py>=2.8',    # fix: h5py conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated
            'gputil>=1.3',
            'pykeops==0.0.14',
            'PyQt5>=5.11'
        ]
    )


# build gui by default
build_gui = str_to_bool(os.environ['BUILD_GUI']) if 'BUILD_GUI' in os.environ else True
print('BUILD_GUI=' + str(build_gui))

if build_gui:
    build_deformetrica_and_gui()
else:
    build_deformetrica()
