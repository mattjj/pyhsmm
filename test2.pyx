# distutils: sources = mult_fast.cpp
# distutils: language = c++
# distutils: extra_compile_args = -O3 -w -march=native
# distutils: include_dirs = deps/Eigen3/

# TODO -DEIGEN_DONT_PARALLELIZE

import numpy as np
cimport numpy as np

import cython
from libc.stdint cimport int32_t
from libcpp.vector cimport vector

cdef extern from "mult_fast.h" namespace "std":
    void c_fast_mult "fast_mult" (
        int N, int32_t *Nsubs, int32_t *rs, double *ps,
        double *super_trans, vector[double*]& sub_transs, vector[double*]& sub_inits,
        double *v, double *out)

@cython.boundscheck(False)
@cython.wraparound(False)
def fast_mult(
        np.ndarray[np.double_t,ndim=1,mode='c'] v not None,
        np.ndarray[np.double_t,ndim=2,mode='c'] super_trans not None,
        np.ndarray[np.int32_t,ndim=1,mode='c'] rs not None,
        np.ndarray[np.double_t,ndim=1,mode='c'] ps not None,
        list sub_transs,
        list sub_initstates):

    # create Nsubs array
    cdef np.ndarray[np.int32_t,ndim=1,mode='c'] Nsubs
    Nsubs = np.ascontiguousarray([s.shape[0] for s in sub_transs],dtype='int32')

    # pack sub_transs (list of numpy arrays) into a std::vector<double *>
    cdef vector[double*] sub_transs_vect
    cdef np.ndarray[np.double_t,ndim=2,mode='c'] temp
    for i in xrange(len(sub_transs)):
        temp = sub_transs[i]
        sub_transs_vect.push_back(&temp[0,0])

    # pack sub_initstates (list of numpy arrays) into a std::vector
    cdef vector[double*] sub_initstates_vect
    cdef np.ndarray[np.double_t,ndim=1,mode='c'] temp2
    for i in xrange(len(sub_initstates)):
        temp2 = sub_initstates[i]
        sub_initstates_vect.push_back(&temp2[0])

    # allocate output
    cdef np.ndarray[np.double_t,ndim=1,mode='c'] out = np.zeros(v.shape[0])

    # call the routine
    c_fast_mult(super_trans.shape[0],&Nsubs[0],&rs[0],&ps[0],&super_trans[0,0],
            sub_transs_vect,sub_initstates_vect,&v[0],&out[0])

    return out

