#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from glob import glob
from os.path import splitext, basename

from setuptools import setup, find_packages

try:  # for pip >= 10
    from pip._internal.req import parse_requirements
except ImportError: # for pip <= 9.0.3
    from pip.req import parse_requirements


setup(
    name='Deformetrica',
    version=open('VERSION').read(),
    url='http://deformetrica.org',
    description='Software for the statistical analysis of 2D and 3D shape data.',
    long_description=open('readme.md').read(),
    author='ARAMIS Lab',
    maintainer='Deformetrica developers',
    maintainer_email='',
    license='INRIA license',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    py_modules=[splitext(basename(path))[0] for path in glob('src/*.py')],
    include_package_data=True,
    zip_safe=False,
    entry_points={
        'console_scripts': ['deformetrica = deformetrica:main']
    },
    classifiers=[
        'Framework :: Deformetrica',
        'Development Status :: 4.0.0 - dev',
        'Environment :: Console',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
    ],
    install_requires=[
        'cmake>=3.10',
        'numpy>=1.10',
        'gputil>=1.3',
        'pykeops==0.0.10'
    ]
)
