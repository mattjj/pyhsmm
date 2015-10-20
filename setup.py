from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext as _build_ext
from distutils.errors import CompileError
from warnings import warn
import os.path
from glob import glob

try:
    from Cython.Distutils import build_ext as _build_ext
except ImportError:
    use_cython = False
else:
    use_cython = True

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

extension_pathspec = os.path.join('pyhsmm','**','*.pyx')
if use_cython:
    from Cython.Build import cythonize
    ext_modules = cythonize(extension_pathspec)
else:
    paths = [os.path.splitext(fp)[0] for fp in glob(extension_pathspec)]
    names = ['.'.join(os.path.split(p)) for p in paths]
    ext_modules = [
        Extension(
            name, sources=[path + '.cpp'],
            include_dirs=[os.path.join('pyhsmm','deps','Eigen3')],
            extra_compile_args=['-O3','-std=c++11','-DNDEBUG','-w','-DHMM_TEMPS_ON_HEAP'])
        for name, path in zip(names,paths)]

setup(name='pyhsmm',
      version='0.1.5',
      description="Bayesian inference in HSMMs and HMMs",
      author='Matthew James Johnson',
      author_email='mattjj@csail.mit.edu',
      url="https://github.com/mattjj/pyhsmm",
      license='MIT',
      packages=['pyhsmm', 'pyhsmm.basic', 'pyhsmm.internals', 'pyhsmm.util'],
      platforms='ALL',
      keywords=['bayesian', 'inference', 'mcmc', 'time-series', 'monte-carlo'],
      install_requires=[
          "Cython >= 0.20.1", "numpy", "scipy", "matplotlib",
          "nose", "pybasicbayes >= 0.1.3", ],
      setup_requires=['numpy'],
      package_data={"pyhsmm": [os.path.join("examples", "*.txt")]},
      ext_modules=ext_modules,
      classifiers=[
          'Development Status :: 4 - Beta',
          'Intended Audience :: Science/Research',
          'Programming Language :: Python',
          'Programming Language :: C++'],
      cmdclass={'build_ext': build_ext})
