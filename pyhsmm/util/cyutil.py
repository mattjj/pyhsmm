import Cython.Build

# This module exists to make the default behavior of cythonize not generate
# modules in a 'pyhsmm' subdirectory. Ultimately this change is necessary
# because this package doesn't follow python packaging conventions, so a better
# long-term fix is to follow conventions and make a proper setup.py script.
# https://github.com/mattjj/pyhsmm/issues/28


def cythonize(*args,**kwargs):
    module_list = Cython.Build.cythonize(*args,**kwargs)
    for m in module_list:
        if m.name.startswith('pyhsmm.'):
            m.name = m.name[7:]
    return module_list

