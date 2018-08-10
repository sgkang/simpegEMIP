#!/usr/bin/env python
from __future__ import print_function
from setuptools import setup, Extension
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
    name="SimPEG-EMIP",
    version="0.0.1",
    setup_requires=[
        'numpy>=1.7',
        'scipy>=0.13',
        'setuptools>=18.0',
        'cython',
        'pymatsolver>=0.1.1',
        'ipython',
        'matplotlib',
        'properties>=0.5.2',
        'vectormath>=0.2.0',
        'discretize>=0.2.0',
        'geoana>=0.0.4'
    ],
    ext_modules=[
        Extension(
            "getJpol",
            sources=["simpegEMIP/TDEM/getJpol.pyx"],
            include_dirs=[numpy.get_include()]
        )
    ],
    author="Seogi Kang"
)
