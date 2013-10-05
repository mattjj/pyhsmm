# distutils: name = internals.subhmm_messages_interface
# distutils: sources = internals/subhmm_messages.cpp
# distutils: language = c++
# distutils: extra_compile_args = -O3 -w -march=native
# distutils: include_dirs = deps/Eigen3/

import numpy as np
cimport numpy as np

import cython
from libc.stdint cimport int32_t
from libcpp.vector cimport vector

cdef extern from "subhmm_messages.h" namespace "std":
    float f_messages_backwards_normalized "subhmm::messages_backwards_normalized" (
        int T, int bigN, int N, int32_t *Nsubs,
        int32_t *rs, float *ps, float *super_trans,
        vector[float*]& sub_transs, vector[float*]& sub_inits, vector[float*]& aBls,
        float *betan)

@cython.boundscheck(False)
@cython.wraparound(False)
def messages_backwards_normalized(
        np.ndarray[np.float32_t,ndim=2,mode='c'] super_trans not None,
        np.ndarray[np.int32_t,ndim=1,mode='c'] rs not None,
        np.ndarray[np.float32_t,ndim=1,mode='c'] ps not None,
        list sub_transs,
        list sub_initstates,
        list aBls,
        np.ndarray[np.float32_t,ndim=2,mode='c'] betan = None):

    # create Nsubs array
    cdef np.ndarray[np.int32_t,ndim=1,mode='c'] Nsubs
    Nsubs = np.ascontiguousarray([s.shape[0] for s in sub_transs],dtype='int32')

    # pack sub_transs (list of numpy arrays) into a std::vector<float *>
    cdef vector[float*] sub_transs_vect
    cdef np.ndarray[np.float32_t,ndim=2,mode='c'] temp
    for i in xrange(len(sub_transs)):
        temp = sub_transs[i]
        sub_transs_vect.push_back(&temp[0,0])

    # pack sub_initstates (list of numpy arrays) into a std::vector<float *>
    cdef vector[float*] sub_initstates_vect
    cdef np.ndarray[np.float32_t,ndim=1,mode='c'] temp2
    for i in xrange(len(sub_initstates)):
        temp2 = sub_initstates[i]
        sub_initstates_vect.push_back(&temp2[0])

    # pack aBls (list of numpy arrays) into a std::vector<float *>
    cdef vector[float*] aBls_vect
    cdef np.ndarray[np.float32_t,ndim=2,mode='c'] temp3
    for i in xrange(len(sub_initstates)):
        temp3 = aBls[i]
        aBls_vect.push_back(&temp3[0,0])

    # allocate output
    cdef int T = aBls[0].shape[0]
    cdef int bigN = sum([r*Nsub for r,Nsub in zip(rs,Nsubs)])
    if betan is None:
        betan = np.empty((T,bigN),dtype='float32')

    # call the routine
    loglike = f_messages_backwards_normalized(
            T,bigN,super_trans.shape[0],&Nsubs[0],
            &rs[0],&ps[0],&super_trans[0,0],
            sub_transs_vect,sub_initstates_vect,aBls_vect,
            &betan[0,0])

    return betan, loglike

