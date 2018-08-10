#!/usr/bin/env python
from __future__ import print_function
from setuptools import setup, Extension, find_packages
import numpy

CLASSIFIERS = [
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python',
    'Topic :: Scientific/Engineering',
    'Topic :: Scientific/Engineering :: Mathematics',
    'Topic :: Scientific/Engineering :: Physics',
    'Operating System :: Microsoft :: Windows',
    'Operating System :: POSIX',
    'Operating System :: Unix',
    'Operating System :: MacOS',
    'Natural Language :: English',
]

setup(
    name="simpegEMIP",
    version="0.0.13",
    packages=find_packages(),
    setup_requires=[
        'setuptools>=18.0',
        'cython',
        'numpy'
    ],
    install_requires=[
        'numpy>=1.7',
        'scipy>=0.13',
        'setuptools>=18.0',
        'cython',
        'pymatsolver>=0.1.1',
        'ipython',
        'matplotlib',
        'properties>=0.5.2',
        'vectormath>=0.2.0',
        'SimPEG',
        'discretize>=0.2.0',
        'geoana>=0.0.4',
        'simpegEM1D'
    ],
    ext_modules=[
        Extension(
            "simpegEMIP.TDEM.getJpol",
            sources=["simpegEMIP/TDEM/getJpol.pyx"],
            include_dirs=[numpy.get_include()]
        )
    ],
    author="Seogi Kang"
)
