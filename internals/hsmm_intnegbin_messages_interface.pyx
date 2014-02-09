# distutils: name = internals.hsmm_intnegbin_messages_interface
# distutils: language = c++
# distutils: extra_compile_args = -O3 -w -DNDEBUG -DEIGEN_DONT_PARALLELIZE -fopenmp -std=c++11
# distutils: extra_link_args = -fopenmp
# distutils: include_dirs = deps/Eigen3/
# cython: boundscheck = False

import numpy as np
cimport numpy as np

from libc.stdint cimport int32_t
from libcpp.vector cimport vector
from cython cimport floating

from cython.parallel import prange

# TODO should just use HSMM sample forwards, messages routine should save betal
# and betastarl

cdef extern from "hsmm_intnegbin_messages.h":
    cdef cppclass inbhsmmc[Type]:
        inbhsmmc()
        Type messages_forwards_normalized(
            int M, int T, Type *A, Type *pi0, Type *aBl,
            int *rs, Type *ps, Type *alphan) nogil

cdef extern from "hmm_messages.h":
    cdef cppclass hmmc[Type]:
        hmmc()
        void sample_backwards_normalized(
            int M, int T, Type *AT, Type *alphan,
            int32_t *stateseq, Type *randseq) nogil

def messages_forwards_normalized(
        floating[:,::1] A not None,
        floating [:,::1] aBl not None,
        floating[::1] pi0 not None,
        int[::1] rs not None,
        floating[::1] ps not None,
        np.ndarray[floating,ndim=2,mode="c"] alphan not None,
        ):
    cdef inbhsmmc[floating] ref
    cdef floating loglike = \
            ref.messages_forwards_normalized(A.shape[0],aBl.shape[0],&A[0,0],&pi0[0],
                    &aBl[0,0],&rs[0],&ps[0],&alphan[0,0])
    return alphan, loglike

def resample_normalized_multiple(
        floating[:,::1] bigA not None,
        floating[:,::1] hsmm_A not None,
        floating[::1] pi0 not None,
        int[::1] rs not None,
        floating[::1] ps not None,
        list aBls not None,
        list stateseqs not None,
        ):
    cdef inbhsmmc[floating] ref
    cdef hmmc[floating] hmmref
    cdef int i

    cdef int num = len(aBls)
    cdef int N = hsmm_A.shape[0]
    cdef int bigN = bigA.shape[0]
    cdef floating[:,:] bigAT = bigA.T.copy()
    cdef int[:] Ts = np.array([seq.shape[0] for seq in stateseqs]).astype(np.int32)
    cdef int[:] starts = np.concatenate(((0,),np.cumsum(Ts)[:-1])).astype(np.int32)

    alphans = [np.empty((aBl.shape[0]+1,bigA.shape[0])) for aBl in aBls]

    cdef vector[floating*] aBls_vect
    cdef vector[floating*] alphans_vect
    cdef vector[int32_t*] stateseqs_vect
    cdef floating[:,:] temp
    cdef int32_t[:] temp2
    for i in range(num):
        temp = aBls[i]
        aBls_vect.push_back(&temp[0,0])
        temp = alphans[i]
        alphans_vect.push_back(&temp[0,0])
        temp2 = stateseqs[i]
        stateseqs_vect.push_back(&temp2[0])

    cdef floating[:] loglikes
    cdef floating[:] randseq
    if floating is double:
        loglikes = np.empty(num,dtype=np.double)
        randseq = np.random.random(size=np.sum(Ts)).astype(np.double)
    else:
        loglikes = np.empty(num,dtype=np.float)
        randseq = np.random.random(size=np.sum(Ts)).astype(np.float)

    with nogil:
        for i in prange(num):
            loglikes[i] = ref.messages_forwards_normalized(N,Ts[i],&hsmm_A[0,0],
                    &pi0[0],aBls_vect[i],&rs[0],&ps[0],alphans_vect[i])
            hmmref.sample_backwards_normalized(bigN,Ts[i],&bigAT[0,0],
                    alphans_vect[i],stateseqs_vect[i],&randseq[starts[i]])

    return np.asarray(loglikes)

