##
## Utilities to aid setup.py
##
import os
import glob
from glob import glob

def is_cython_avail():
    """
    Check if Cython is available. Return True if so,
    False otherwise.
    """
    try:
        from Cython.Build import cythonize
        return True
    except:
        return False

def get_cython_generated_sources(pyx_glob, exts=[".c", ".cpp", ".cc"]):
    """
    Given a glob expression to *.pyx files, return
    the filenames of Cython-generated C/C++ code that
    matches the *.pyx files.
    """
    cython_sources = []
    for fname in glob(pyx_glob):
        base_fname = os.path.splitext(fname)[0]
        # Collect all the files ending in the extensions in
        # 'exts' if they match a *.pyx file name
        for fname_ext in exts:
            source_fname = base_fname + fname_ext
            if os.path.isfile(source_fname):
                cython_sources.append(source_fname)
    return cython_sources

def clean_build(fnames_to_remove=[]):
    print "Cleaning files..."
    fnames_to_remove.extend(glob("*.egg-info"))
    # Remove *.cpp/*.c files that are in pyhsmm/
    fnames_to_remove.extend(fnames_to_remove)
    for fname in fnames_to_remove:
        # Remove files if you can
        try:
            print "Removing %s" %(fname)
            if os.path.isdir(fname):
                shutil.rmtree(fname)
            else:
                os.remove(fname)
        except:
            pass
    
    
