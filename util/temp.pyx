# distutils: extra_compile_args = -O3 -w -DEIGEN_DONT_PARALLELIZE -DNDEBUG -fopenmp -std=c++11
# distutils: extra_link_args = -fopenmp
# distutils: language = c++
# distutils: include_dirs = deps/Eigen3/ internals/

import numpy as np
cimport numpy as np

from libcpp.vector cimport vector
from libc.stdint cimport int32_t, int64_t
from cython import floating

from cython.parallel import prange

cdef extern from "temp.h":
    cdef cppclass dummy[Type]:
        dummy()
        void getstats(
            int M, int T, int D, int32_t *stateseq, Type *data,
            Type *stats) nogil

def getstats(num_states, stateseqs, datas):
    cdef int i
    cdef dummy[double] ref

    cdef int M = num_states
    cdef int K = len(datas)
    cdef int D = datas[0].shape[1]
    cdef int32_t[::1] Ts = np.array([d.shape[0] for d in datas]).astype('int32')

    cdef vector[int32_t*] stateseqs_v
    cdef vector[double*] datas_v
    cdef double[:,:] temp
    cdef int32_t[:] temp2
    for i in range(K):
        temp = datas[i]
        datas_v.push_back(&temp[0,0])
        temp2 = stateseqs[i]
        stateseqs_v.push_back(&temp2[0])

    cdef double[:,:,::1] out = np.zeros((2*K,M,2*D+1)) # NOTE: 2*K to avoid false sharing

    with nogil:
        for i in prange(K):
            ref.getstats(M,Ts[i],D,stateseqs_v[i],datas_v[i],&out[2*i,0,0])

    ret = []
    for row in np.sum(out,axis=0):
        n = row[-1]
        xbar = row[:D] / (n if n > 0 else 1.)
        sumsq = row[D:2*D] - 2*xbar*row[:D] + n*xbar**2
        ret.append((n,xbar,sumsq))
    return ret

