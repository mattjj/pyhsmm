# distutils: name = internals.hsmm_messages_interface
# distutils: language = c++
# distutils: extra_compile_args = -O3 -w -DNDEBUG -std=c++11
# distutils: include_dirs = deps/Eigen3/
# cython: boundscheck = False

import numpy as np
cimport numpy as np

from libc.stdint cimport int32_t
from libcpp.vector cimport vector

# NOTE: using the cython.floating fused type with typed memory views generates
# all possible type combinations, is not intended here.
# https://groups.google.com/forum/#!topic/cython-users/zdlliIRF1a4
from cython cimport floating

from cython.parallel import prange

cdef extern from "hsmm_messages.h":
    cdef cppclass hsmmc[Type]:
        hsmmc()
        void messages_backwards_log(
            int M, int T, Type *A, Type *aBl, Type *aDl, Type *aDsl,
            Type *betal, Type *betastarl, int right_censoring, int trunc) nogil
        void sample_forwards_log(
            int M, int T, Type *A, Type *pi0, Type *aBl, Type *aD,
            Type *betal, Type *betastarl, int32_t *stateseq, Type *randseq) nogil

def messages_backwards_log(
        floating[:,::1] A not None,
        floating[:,::1] aBl not None,
        floating[:,::1] aDl not None,
        floating[:,::1] aDsl not None,
        np.ndarray[floating,ndim=2,mode="c"] betal not None,
        np.ndarray[floating,ndim=2,mode="c"] betastarl not None,
        int right_censoring, int trunc):
    cdef hsmmc[floating] ref

    ref.messages_backwards_log(A.shape[0],aBl.shape[0],&A[0,0],
            &aBl[0,0],&aDl[0,0],&aDsl[0,0],&betal[0,0],&betastarl[0,0],
            right_censoring,trunc)

    return betal, betastarl

def sample_forwards_log(
        floating[:,::1] A not None,
        floating[:,::1] caBl not None,
        floating[:,::1] aDl not None,
        floating[::1] pi0 not None,
        floating[:,::1] betal not None,
        floating[:,::1] betastarl not None,
        np.ndarray[np.int32_t,ndim=1,mode="c"] stateseq not None,
        ):
    cdef hsmmc[floating] ref

    # NOTE: not all of randseq will be consumed; one entry is consumed for each
    # duration and one for each transition, so they can only all be used if each
    # duration is deterministically 1
    cdef floating[:] randseq
    if floating is double:
        randseq = np.random.random(size=2*caBl.shape[0]).astype(np.double)
    else:
        randseq = np.random.random(size=2*caBl.shape[0]).astype(np.float)

    ref.sample_forwards_log(A.shape[0],caBl.shape[0],&A[0,0],&pi0[0],
            &caBl[0,0],&aDl[0,0],&betal[0,0],&betastarl[0,0],&stateseq[0],&randseq[0])

    return stateseq

def resample_log_multiple(
        floating[:,::1] A not None,
        floating[::1] pi0 not None,
        floating[:,::1] aDl not None,
        floating[:,::1] aDsl not None,
        list aBls not None,
        int[::1] right_censorings not None,
        int[::1] truncs not None,
        list stateseqs not None,
        ):
    cdef hsmmc[floating] ref
    cdef int i

    cdef int num = len(aBls)
    cdef int N = A.shape[0]
    cdef int[:] Ts = np.array([seq.shape[0] for seq in stateseqs],dtype=np.int32)
    cdef int[:] starts = np.concatenate(((0,),2*np.cumsum(Ts)[:-1])).astype(np.int32)

    # NOTE: allocate temps, pad to avoid false sharing in the cache

    betals = [np.empty((aBl.shape[0]+1,aBl.shape[1]),dtype=aBl.dtype) for aBl in aBls]
    betastarls = [np.empty_like(betal) for betal in betals]
    caBls = [np.vstack((np.zeros(N,dtype=aBl.dtype),np.cumsum(aBl[:-1],axis=0))) for aBl in aBls]

    cdef vector[floating*] aBls_vect
    cdef vector[floating*] caBls_vect
    cdef vector[floating*] betals_vect
    cdef vector[floating*] betastarls_vect
    cdef vector[int32_t*] stateseqs_vect
    cdef floating[:,:] temp
    cdef int32_t[:] temp2
    for i in range(num):
        temp = aBls[i]
        aBls_vect.push_back(&temp[0,0])
        temp = caBls[i]
        caBls_vect.push_back(&temp[0,0])
        temp = betals[i]
        betals_vect.push_back(&temp[0,0])
        temp = betastarls[i]
        betastarls_vect.push_back(&temp[0,0])
        temp2 = stateseqs[i]
        stateseqs_vect.push_back(&temp2[0])

    cdef floating[:] randseq
    cdef floating[:] loglikes
    if floating is double:
        randseq = np.random.random(size=2*np.sum(Ts)).astype(np.double)
        loglikes = np.empty(num,dtype=np.double)
    else:
        randseq = np.random.random(size=2*np.sum(Ts)).astype(np.float)
        loglikes = np.empty(num,dtype=np.float)

    with nogil:
        for i in prange(num):
            ref.messages_backwards_log(N,Ts[i],&A[0,0],
                    aBls_vect[i],&aDl[0,0],&aDsl[0,0],betals_vect[i],betastarls_vect[i],
                    right_censorings[i],truncs[i])
            ref.sample_forwards_log(N,Ts[i],&A[0,0],&pi0[0],
                    caBls_vect[i],&aDl[0,0],betals_vect[i],betastarls_vect[i],
                    stateseqs_vect[i],&randseq[starts[i]])

    for i in range(num):
        temp = <floating[:Ts[i],:N]> betastarls_vect[i]
        loglikes[i] = np.logaddexp.reduce(
                np.asarray(pi0) + np.asarray(aBls[i])[0] + np.asarray(temp)[0])

    return np.asarray(loglikes)

