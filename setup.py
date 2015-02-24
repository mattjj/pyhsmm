from distutils.core import setup
import numpy as np
import sys, os

##
## TODO: ideally, make Cython optional. Allow compilation
## compilation from Cython-generated *.c files, which would
## allow users to install the package without having Cython.
## Technically only developers need to be able to run Cython.
## 
try:
    import Cython
    from Cython.Build import cythonize
except ImportError:
    print "Cannot import Cython! Cython is required for pyhsmm."

## Not necessary anymore
#from util import cyutil
#from cyutil import cythonize # my version of Cython.Build.cythonize

# NOTE: distutils makes no sense

extra_link_args = []
extra_compile_args = ['-DHMM_TEMPS_ON_HEAP','-DNDEBUG','-w']

if '--with-old-clang' in sys.argv:
    sys.argv.remove('--with-old-clang')
    extra_compile_args.append('-stdlib=libc++')
    extra_link_args.append('-stdlib=libc++')

if '--with-openmp' in sys.argv:
    sys.argv.remove('--with-openmp')
    extra_compile_args.append('-fopenmp')
    extra_link_args.append('-fopenmp')

if '--with-native' in sys.argv:
    sys.argv.remove('--with-native')
    extra_compile_args.append('-march=native')

if '--with-mkl' in sys.argv:
    sys.argv.remove('--with-mkl')
    extra_compile_args.extend(['-m64','-I' + os.environ['MKLROOT'] + '/include','-DEIGEN_USE_MKL_ALL'])
    extra_link_args.extend(('-Wl,--start-group %(MKLROOT)s/lib/intel64/libmkl_intel_lp64.a %(MKLROOT)s/lib/intel64/libmkl_core.a %(MKLROOT)s/lib/intel64/libmkl_sequential.a -Wl,--end-group -lm' % {'MKLROOT':os.environ['MKLROOT']}).split(' '))

if '--with-assembly' in sys.argv:
    sys.argv.remove('--with-assembly')
    extra_compile_args.extend(['--save-temps','-masm=intel','-fverbose-asm'])

ext_modules = cythonize('./pyhsmm/*/*.pyx')
for e in ext_modules:
    e.extra_compile_args.extend(extra_compile_args)
    e.extra_link_args.extend(extra_link_args)

long_description = open("./readme.md").read()

##
## Temporary hack: check for pybasicbayes presence
## and register it as a submodule of pyhsmm. This means
## the repository had to be cloned with '--recursive' flag.
## The real solution is to make pybasicbayes a real Python package
## too and register it as a dependency in 'install_requires' below.
##
if not os.path.isdir("./pyhsmm/basic/pybasicbayes"):
    print "Cannot find \'pybasicbayes\'. Did you clone the pyhsmm repository " \
          "with --clone?"
    print "Aborting installation..."
    sys.exit(1)

PYHSMM_VERSION = "0.1"

setup(name = 'pyhsmm',
      version = PYHSMM_VERSION,
      description = "Bayesian inference in HSMMs and HMMs",
      long_description = long_description,
      author = 'Matt Johnson',
      author_email = 'mattjj@csail.mit.edu',
      maintainer = 'Matt Johnson',
      maintainer_email = 'mattjj@csail.mit.edu',
      packages = ['pyhsmm',
                  'pyhsmm.basic',
                  'pyhsmm.internals',
                  'pyhsmm.plugins',
                  'pyhsmm.testing',
                  'pyhsmm.util'],
      platforms = 'ALL',
      keywords = ['bayesian', 'inference', 'mcmc', 'time-series',
                  'monte-carlo'],
      install_requires = [
          "Cython >= 0.20.1",
          "numpy",
          "scipy",
          "matplotlib"
          ],
       ext_modules=ext_modules,
       include_dirs=[np.get_include(),]
)

