from distutils.core import setup
import numpy as np

from cyutil import cythonize # my version of Cython.Build.cythonize

setup(
    ext_modules=cythonize('**/*.pyx'),
    include_dirs=[np.get_include(),],
)

