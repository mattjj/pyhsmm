from distutils.core import setup
from Cython.Build import cythonize
import numpy

# TODO I hate distutils, write a Makefile alternative

setup(
        ext_modules=cythonize('**/*.pyx'),
        include_dirs=[numpy.get_include(),],
        )

