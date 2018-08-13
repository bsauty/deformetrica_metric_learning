#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from setuptools import setup, find_packages

try:  # for pip >= 10
    from pip._internal.req import parse_requirements
except ImportError:     # for pip <= 9.0.3
    from pip.req import parse_requirements


setup(
    name='Deformetrica',
    version=open('VERSION', encoding='utf-8').read(),
    url='http://www.deformetrica.org',
    description='Software for the statistical analysis of 2D and 3D shape data.',
    long_description=open('README.md', encoding='utf-8').read(),
    author='ARAMIS Lab',
    maintainer='Deformetrica developers',
    maintainer_email='deformetrica.team@gmail.com',
    license='INRIA license',
    package_dir={'': 'src'},
    packages=find_packages('src'),
    package_data={'': ['*.json', '*.png']},
    include_package_data=True,
    zip_safe=False,
    entry_points={
        'console_scripts': ['deformetrica=deformetrica:main'],  # CLI
        'gui_scripts': ['deformetrica-gui=gui.__main__:main']   # GUI
    },
    classifiers=[
        'Framework :: Deformetrica',
        'Development Status :: 4.0.1',
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
