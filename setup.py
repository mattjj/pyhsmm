from distutils.core import setup
import numpy

from util.cython import cythonize # my version of Cython.Build.cythonize

setup(
    ext_modules=cythonize('**/*.pyx'),
    include_dirs=[numpy.get_include(),],
)

