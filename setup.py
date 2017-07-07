from __future__ import print_function
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext as _build_ext
from setuptools.command.sdist import sdist as _sdist
from distutils.command.clean import clean as _clean
from distutils.errors import CompileError
from warnings import warn
import os
import sys
from glob import glob
import tarfile
import shutil

from future.standard_library import install_aliases
install_aliases()
from urllib.request import urlretrieve

# use cython if we can import it successfully
try:
    from Cython.Distutils import build_ext as _build_ext
except ImportError:
    use_cython = False
else:
    use_cython = True

# wrap the build_ext command to handle numpy bootstrap and compilation errors
class build_ext(_build_ext):
    # see http://stackoverflow.com/q/19919905 for explanation
    def finalize_options(self):
        _build_ext.finalize_options(self)
        __builtins__.__NUMPY_SETUP__ = False
        import numpy as np
        self.include_dirs.append(np.get_include())

    # if extension modules fail to build, keep going anyway
    def run(self):
        try:
            _build_ext.run(self)
        except CompileError:
            warn('Failed to build extension modules')
            import traceback
            print(traceback.format_exc(), file=sys.stderr)

# wrap the sdist command to try to generate cython sources
class sdist(_sdist):
    def run(self):
        try:
            from Cython.Build import cythonize
            cythonize(os.path.join('pyhsmm','**','*.pyx'))
        except:
            warn('Failed to generate extension files from Cython sources')
        finally:
            _sdist.run(self)

# wrap the clean command to remove object files
class clean(_clean):
    def run(self):
        try:
            for f in glob(os.path.join('pyhsmm','**','*.so')):  # not recursive before Python 3.5
                os.remove(f)
        except:
            warn('Failed to remove all object files')
        finally:
            _clean.run(self)

# make dependency directory
if not os.path.exists('deps'):
    os.mkdir('deps')

# download Eigen if we don't have it in deps
eigenurl = 'http://bitbucket.org/eigen/eigen/get/3.2.6.tar.gz'
eigentarpath = os.path.join('deps', 'Eigen.tar.gz')
eigenpath = os.path.join('deps', 'Eigen')
if not os.path.exists(eigenpath):
    print('Downloading Eigen...')
    urlretrieve(eigenurl, eigentarpath)
    with tarfile.open(eigentarpath, 'r') as tar:
        tar.extractall('deps')
    thedir = glob(os.path.join('deps', 'eigen-eigen-*'))[0]
    shutil.move(os.path.join(thedir, 'Eigen'), eigenpath)
    print('...done!')

# make a list of extension modules
extension_pathspec = os.path.join('pyhsmm','**','*.pyx')  # not recursive before Python 3.5
paths = [os.path.splitext(fp)[0] for fp in glob(extension_pathspec)]
names = ['.'.join(os.path.split(p)) for p in paths]
ext_modules = [
    Extension(
        name, sources=[path + '.cpp'],
        include_dirs=['deps'],
        extra_compile_args=['-O3','-std=c++11','-DNDEBUG','-w','-DHMM_TEMPS_ON_HEAP'])
    for name, path in zip(names,paths)]

# if using cython, rebuild the extension files from the .pyx sources
if use_cython:
    from Cython.Build import cythonize
    try:
        ext_modules = cythonize(extension_pathspec)
    except:
        warn('Failed to generate extension module code from Cython files')

# put it all together with a call to setup()
setup(name='pyhsmm',
      version='0.1.6',
      description="Bayesian inference in HSMMs and HMMs",
      author='Matthew James Johnson',
      author_email='mattjj@csail.mit.edu',
      url="https://github.com/mattjj/pyhsmm",
      license='MIT',
      packages=['pyhsmm', 'pyhsmm.basic', 'pyhsmm.internals', 'pyhsmm.util'],
      platforms='ALL',
      keywords=['bayesian', 'inference', 'mcmc', 'time-series', 'monte-carlo',
                'variational inference', 'mean field', 'vb'],
      install_requires=[
          "numpy", "scipy", "matplotlib", "nose", "pybasicbayes >= 0.1.3", "future", "six"],
      setup_requires=['numpy', "future", "six"],
      ext_modules=ext_modules,
      classifiers=[
          'Development Status :: 4 - Beta',
          'Intended Audience :: Science/Research',
          'Programming Language :: Python',
          'Programming Language :: C++'],
      cmdclass={'build_ext': build_ext, 'sdist': sdist, 'clean': clean})
