# distutils: name = internals.hmm_messages_interface
# distutils: language = c++
# distutils: extra_compile_args = -O3 -w -DNDEBUG -std=c++11
# distutils: include_dirs = deps/Eigen3/
# cython: boundscheck = False

# NOTE: cython can use templated classes but not templated functions in 0.19.1,
# hence the wrapper class. no syntax for directly calling static members, though

import numpy as np
cimport numpy as np

from libc.stdint cimport int32_t
from libcpp.vector cimport vector

# NOTE: using the cython.floating fused type with typed memory views generates
# all possible type combinations, is not intended here.
# https://groups.google.com/forum/#!topic/cython-users/zdlliIRF1a4
from cython cimport floating

from cython.parallel import prange

cdef extern from "hmm_messages.h":
    cdef cppclass hmmc[Type]: # NOTE: default states type is int32_t
        hmmc()
        void messages_backwards_log(
            int M, int T, Type *A, Type *aBl, Type *betal) nogil
        void messages_forwards_log(
            int M, int T, Type *A, Type *pi0, Type *aBl, Type *alphal) nogil
        Type messages_forwards_normalized(
            int M, int T, Type *A, Type *pi0, Type *aBl, Type *alphan) nogil
        void sample_forwards_log(
            int M, int T, Type *A, Type *pi0, Type *aBl, Type *betal,
            int32_t *stateseq, Type *randseq) nogil
        void sample_backwards_normalized(
            int M, int T, Type *AT, Type *alphan,
            int32_t *stateseq, Type *randseq) nogil
        void viterbi(
            int M, int T, Type *A, Type *pi0, Type *aBl, int32_t *stateseq) nogil
        Type expected_statistics_log(
            int M, int T, Type *log_trans_potential, Type *log_likelihood_potential,
            Type *alphal, Type *betal,
            Type *expected_states, Type *expected_transcounts)

def messages_backwards_log(
        floating[:,::1] A not None,
        floating[:,::1] aBl not None,
        np.ndarray[floating,ndim=2,mode="c"] betal not None,
        ):
    cdef hmmc[floating] ref
    ref.messages_backwards_log(A.shape[0],aBl.shape[0],&A[0,0],&aBl[0,0],&betal[0,0])
    return betal

def messages_forwards_log(
        floating[:,::1] A not None,
        floating[:,::1] aBl not None,
        floating[::1] pi0 not None,
        np.ndarray[floating,ndim=2,mode="c"] alphal not None,
        ):
    cdef hmmc[floating] ref
    ref.messages_forwards_log(A.shape[0],aBl.shape[0],&A[0,0],&pi0[0],&aBl[0,0],
            &alphal[0,0])
    return alphal

def sample_forwards_log(
        floating[:,::1] A not None,
        floating[:,::1] aBl not None,
        floating[::1] pi0 not None,
        floating[:,::1] betal not None,
        np.ndarray[np.int32_t,ndim=1,mode="c"] stateseq not None,
        ):
    cdef hmmc[floating] ref

    cdef floating[:] randseq
    if floating is double:
        randseq = np.random.random(size=aBl.shape[0]).astype(np.double)
    else:
        randseq = np.random.random(size=aBl.shape[0]).astype(np.float)

    ref.sample_forwards_log(A.shape[0],aBl.shape[0],&A[0,0],&pi0[0],&aBl[0,0],
            &betal[0,0],&stateseq[0],&randseq[0])

    return stateseq

def expected_statistics_log(
        np.ndarray[floating,ndim=2,mode='c'] log_trans_potential not None,
        np.ndarray[floating,ndim=2,mode='c'] log_likelihood_potential not None,
        np.ndarray[floating,ndim=2,mode='c'] alphal not None,
        np.ndarray[floating,ndim=2,mode='c'] betal not None,
        np.ndarray[floating,ndim=2,mode='c'] expected_states not None,
        np.ndarray[floating,ndim=2,mode='c'] expected_transcounts not None,
        ):
    # cdef int t
    # cdef int T = expected_states.shape[0]
    # cdef double normalizer = np.logaddexp.reduce(alphal[-1])
    # for t in range(T-1):
    #     np.exp(alphal[t] + betal[t] - normalizer,out=expected_states[t])
    #     expected_transcounts += np.exp(alphal[t][:,None] + betal[t+1]
    #             + log_likelihood_potential[t+1] + log_trans_potential - normalizer)
    # np.exp(alphal[T-1] + betal[T-1] - normalizer,out=expected_states[T-1])

    # return expected_states, expected_transcounts, normalizer

    cdef hmmc[floating] ref
    cdef floating log_normalizer = ref.expected_statistics_log(
            log_trans_potential.shape[0],alphal.shape[0],
            &log_trans_potential[0,0],
            &log_likelihood_potential[0,0],
            &alphal[0,0],
            &betal[0,0],
            &expected_states[0,0],
            &expected_transcounts[0,0])
    return expected_states, expected_transcounts, log_normalizer

def messages_forwards_normalized(
        floating[:,::1] A not None,
        floating[:,::1] aBl not None,
        floating[::1] pi0 not None,
        np.ndarray[floating,ndim=2,mode="c"] alphan not None,
        ):
    cdef hmmc[floating] ref
    cdef floating loglike = \
            ref.messages_forwards_normalized(A.shape[0],aBl.shape[0],&A[0,0],&pi0[0],
                &aBl[0,0],&alphan[0,0])
    return alphan, loglike

def sample_backwards_normalized(
        floating[:,::1] AT not None,
        floating[:,::1] alphan not None,
        np.ndarray[np.int32_t,ndim=1,mode="c"] stateseq not None,
        ):
    cdef hmmc[floating] ref

    cdef floating[:] randseq
    if floating is double:
        randseq = np.random.random(size=alphan.shape[0]).astype(np.double)
    else:
        randseq = np.random.random(size=alphan.shape[0]).astype(np.float)

    ref.sample_backwards_normalized(AT.shape[0],alphan.shape[0],&AT[0,0],
            &alphan[0,0],&stateseq[0],&randseq[0])

    return stateseq

# NOTE: the purpose of this method is to dispatch to OpenMP, so it only makes
# sense if this file is compiled with CCFLAGS=-fopenmp LDFLAGS=-fopenmp
def resample_normalized_multiple(
        floating[:,::1] A not None,
        floating[::1] pi0 not None,
        list aBls not None,
        list stateseqs not None,
        ):
    cdef hmmc[floating] ref
    cdef int i

    # NOTE: to run without the gil, we have to unpack all the data we need from
    # the python objects

    cdef int num = len(aBls)
    cdef int N = A.shape[0]
    cdef floating[:,:] AT = A.T.copy()
    cdef int[:] Ts = np.array([seq.shape[0] for seq in stateseqs],dtype=np.int32)
    cdef int[:] starts = np.concatenate(((0,),np.cumsum(Ts)[:-1])).astype(np.int32)

    # NOTE: allocate temps, pad to avoid false sharing in the cache

    alphans = [np.empty((aBl.shape[0]+1,aBl.shape[1])) for aBl in aBls]

    # NOTE: this next bit is converting python lists to C++ vectors for nogil

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
            loglikes[i] = ref.messages_forwards_normalized(N,Ts[i],&A[0,0],
                    &pi0[0],aBls_vect[i],alphans_vect[i])
            ref.sample_backwards_normalized(N,Ts[i],&AT[0,0],
                    alphans_vect[i],stateseqs_vect[i],&randseq[starts[i]])

    return np.asarray(loglikes)

def viterbi(
        floating[:,::1] A not None,
        floating[:,::1] aBl not None,
        floating[::1] pi0 not None,
        np.ndarray[np.int32_t,ndim=1,mode="c"] stateseq not None,
        ):
    cdef hmmc[floating] ref
    ref.viterbi(A.shape[0],aBl.shape[0],&A[0,0],&pi0[0],&aBl[0,0],
                &stateseq[0])
    return stateseq

