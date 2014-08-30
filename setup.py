from distutils.core import setup
import numpy as np
import sys

from util.cyutil import cythonize # my version of Cython.Build.cythonize

# NOTE: distutils makes no sense

extra_link_args = ['-fopenmp']
extra_compile_args = ['-DHMM_TEMPS_ON_HEAP','-fopenmp']

if '--with-old-clang' in sys.argv:
    sys.argv.remove('--with-old-clang')
    extra_compile_args.append('-stdlib=libc++')
    extra_link_args.append('-stdlib=libc++')

if '--no-openmp' in sys.argv:
    sys.argv.remove('--with-openmp')
    extra_compile_args.remove('-fopenmp')
    extra_link_args.remove('-fopenmp')

if '--with-native' in sys.argv:
    sys.argv.remove('--with-native')
    extra_compile_args.append('-march=native')

ext_modules = cythonize('**/*.pyx')
for e in ext_modules:
    e.extra_compile_args.extend(extra_compile_args)
    e.extra_link_args.extend(extra_link_args)

setup(
    ext_modules=ext_modules,
    include_dirs=[np.get_include(),],
)

