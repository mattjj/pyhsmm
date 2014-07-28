# distutils: extra_compile_args = -O3 -w -DEIGEN_DONT_PARALLELIZE -DNDEBUG -fopenmp -std=c++11 -m64 -I/opt/intel/mkl/include/ -fpermissive -march=native -DEIGEN_USE_MKL_ALL -mavx2
# distutils: extra_link_args = -fopenmp -Wl,--no-as-needed -L/opt/intel/mkl/lib/intel64 -lmkl_intel_lp64 -lmkl_core -lmkl_gnu_thread -ldl -lm
# distutils: language = c++
# distutils: include_dirs = deps/Eigen3/ internals/ /opt/intel/mkl/include/
# distutils: library_dirs = /opt/intel/mkl/lib/intel64/
# cython: boundscheck = False

### -DEIGEN_USE_MKL_ALL -march=native
### distutils: mkl_libs = mkl_def, mkl_intel_lp64, mkl_core, mkl_gnu_thread

import numpy as np
cimport numpy as np

from libcpp.vector cimport vector
from libc.stdint cimport int32_t, int64_t
from cython cimport floating

# TODO do more type generic stuff (less double, more floating)

from cython.parallel import prange

# TODO pass in num threads

cdef extern from "temp2.h":
    cdef cppclass dummy[Type]:
        dummy()
        void gmm_likes(
            int T, int Tblock, int N, int K, int D,
            Type *data, Type *weights,
            Type *Js, Type *mus_times_Js, Type *normalizers,
            int32_t *changepoints,
            Type *aBBl)

def gmm_likes(
        double[:,::1] data not None,        # T x D
        double[:,:,::1] sigmas not None,    # N x K x D
        double[:,:,::1] mus not None,       # N x K x D
        double[:,::1] weights not None,     # N x K
        int32_t[:,::1] changepoints not None, # T x 2
        double[:,::1] aBBl not None,        # T x N
        ):
    cdef dummy[double] ref
    cdef int T = data.shape[0]
    cdef int Tblock = aBBl.shape[0]
    cdef int N = sigmas.shape[0]
    cdef int K = sigmas.shape[1]
    cdef int D = sigmas.shape[2]

    cdef double[:,:,::1] Js = -1./(2*np.asarray(sigmas))
    cdef double[:,:,::1] mus_times_Js = 2*np.asarray(mus)*np.asarray(Js)
    cdef double[:,::1] normalizers = \
            (np.asarray(mus)**2*np.asarray(Js) \
            - 1./2*np.log(2*np.pi*np.asarray(sigmas))).sum(2)

    ref.gmm_likes(T,Tblock,N,K,D,
        &data[0,0],&weights[0,0],
        &Js[0,0,0],&mus_times_Js[0,0,0],&normalizers[0,0],
        &changepoints[0,0],
        &aBBl[0,0])

