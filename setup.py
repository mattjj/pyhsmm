from distutils.core import setup, Extension
import numpy as np
import sys
import os
import shutil
from glob import glob

import setuphelper

PYHSMM_VERSION = "0.1.3"

########################################################
# check if Cython is available and define pyx location #
########################################################
    
cython_avail = setuphelper.is_cython_avail()
# where Cython code lives
cython_pathspec = os.path.join('pyhsmm', '**', '*.pyx')

# where Cython-generated C/C++ files are
cython_generated_fnames = \
  setuphelper.get_cython_generated_sources(cython_pathspec)

############################################
## clean build files (for developers only) #
############################################

if len(sys.argv) >= 2 and sys.argv[1] == "clean":
    setuphelper.clean_build(fnames_to_remove=cython_generated_fnames)

###########################
#  compilation arguments  #
###########################

extra_link_args = []
extra_compile_args = []

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
    # NOTE: there's no way this will work on Windows
    extra_compile_args.extend(['-m64','-I' + os.environ['MKLROOT'] + '/include','-DEIGEN_USE_MKL_ALL'])
    extra_link_args.extend(('-Wl,--start-group %(MKLROOT)s/lib/intel64/libmkl_intel_lp64.a %(MKLROOT)s/lib/intel64/libmkl_core.a %(MKLROOT)s/lib/intel64/libmkl_sequential.a -Wl,--end-group -lm' % {'MKLROOT':os.environ['MKLROOT']}).split(' '))

if '--with-assembly' in sys.argv:
    sys.argv.remove('--with-assembly')
    extra_compile_args.extend(['--save-temps','-masm=intel','-fverbose-asm'])

if '--with-cython' in sys.argv:
    sys.argv.remove('--with-cython')
    # if cython is not available, flag error
    if not cython_avail:
        print "Asked to use Cython but it is not installed."
        sys.exit(1)
    use_cython = True
else:
    use_cython = False


#######################################
# handle source distributions (sdist) #
#######################################

# If we're using sdist, then we have to use Cython
if len(sys.argv) >= 2 and sys.argv[1] == "sdist":
    if not cython_avail:
        print "Making sdist requires Cython to be installed."
        sys.exit(1)
    use_cython = True

#######################
#  extension modules  #
#######################
ext_modules = []
cmdclass = {}
if use_cython:
    print "Using Cython.."
    from Cython.Build import cythonize
    from Cython.Distutils import build_ext
    ext_modules = cythonize(cython_pathspec)
    cmdclass.update({'build_ext': build_ext})
else:
    print "Not using Cython. Building from C/C++ source..."
    paths = [os.path.splitext(fp)[0] for fp in glob(cython_pathspec)]
    names = ['.'.join(os.path.split(p)) for p in paths]
    for name, path in zip(names, paths):
        # Note: this assumes that all Cython generated files
        # are *.cpp and will fail for *.c generated Cython files
        source_path = path + ".cpp"
        if not os.path.isfile(source_path):
            print "Warning: could not find %s" %(source_path)
            print "  - Skipping"
            continue
        # Make paths into Python module names
        name = name.replace("/", ".")
        print "Making extension %s" %(name)
        ext_module = Extension(name,
                               sources=[source_path],
                               include_dirs=[os.path.join('deps', 'Eigen3')],
                               extra_compile_args=['-O3','-std=c++11','-DNDEBUG','-w',
                                                   '-DHMM_TEMPS_ON_HEAP'])
        ext_modules.append(ext_module)

for e in ext_modules:
    e.extra_compile_args.extend(extra_compile_args)
    e.extra_link_args.extend(extra_link_args)


############
#  basics  #
############

setup(name='pyhsmm',
      version=PYHSMM_VERSION,
      description="Bayesian inference in HSMMs and HMMs",
      author='Matthew James Johnson',
      author_email='mattjj@csail.mit.edu',
      maintainer='Matthew James Johnson',
      maintainer_email='mattjj@csail.mit.edu',
      url="https://github.com/mattjj/pyhsmm",
      packages=['pyhsmm',
                'pyhsmm.basic',
                'pyhsmm.internals',
                'pyhsmm.examples',
                'pyhsmm.plugins',
                'pyhsmm.testing',
                'pyhsmm.util'],
      platforms='ALL',
      keywords=['bayesian', 'inference', 'mcmc', 'time-series',
                'monte-carlo'],
      install_requires=[
          "numpy",
          "scipy",
          "matplotlib",
          "nose",
          "pybasicbayes",
      ],
      package_data={"pyhsmm": [os.path.join("examples", "*.txt")]},
      cmdclass=cmdclass,
      ext_modules=ext_modules,
      include_dirs=[np.get_include(),],
      classifiers=[
          'Intended Audience :: Science/Research',
          'Programming Language :: Python',
          'Programming Language :: C++'
      ])

