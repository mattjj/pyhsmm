from distutils.core import setup, Extension
import numpy as np
import sys
import os
from glob import glob

PYHSMM_VERSION = "0.1.3"

# Clean files (for developers only)
if len(sys.argv) >= 2 and sys.argv[1] == "clean":
    print "Cleaning files..."
    if os.path.isdir("build"):
        os.removedirs("build/")
    fnames_to_remove = glob(os.path.join("pyhsmm", "**", "*.so"))
    fnames_to_remove.extend(glob("*.egg-info"))
    # Remove *.cpp/*.c files that are in pyhsmm/
    # note that this assumes that all *.cpp/*.c files are Cython-generated
    fnames_to_remove.extend(glob(os.path.join("pyhsmm", "**", "*.cpp")))
    fnames_to_remove.extend(glob(os.path.join("pyhsmm", "**", "*.c")))
    for fname in fnames_to_remove:
        # Remove files
        os.remove(so_fname)

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
    use_cython = True
else:
    use_cython = False


#######################
#  extension modules  #
#######################

cython_pathspec = os.path.join('pyhsmm', '**', '*.pyx')

if use_cython:
    from Cython.Build import cythonize
    ext_modules = cythonize(cython_pathspec)
else:
    paths = [os.path.splitext(fp)[0] for fp in glob(cython_pathspec)]
    names = ['.'.join(os.path.split(p)) for p in paths]
    ext_modules = []
    for name, path in zip(names, paths):
        source_path = path + ".cpp"
        if not os.path.isfile(source_path):
            print "Warning: could not find %s" %(source_path)
            print "  - Skipping"
            continue
        ext_module = Extension(name,
                               sources=[source_path],
                               include_dirs=[os.path.join('deps','Eigen3')],
                               extra_compile_args=['-O3','-std=c++11','-DNDEBUG','-w',
                                                   '-DHMM_TEMPS_ON_HEAP'])

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
      ext_modules=ext_modules,
      include_dirs=[np.get_include(),],
      classifiers=[
          'Intended Audience :: Science/Research',
          'Programming Language :: Python',
          'Programming Language :: C++',
      ])

