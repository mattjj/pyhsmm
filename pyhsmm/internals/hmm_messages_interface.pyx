# distutils: language = c++
# distutils: extra_compile_args = -std=c++11 -O3 -w -DNDEBUG -DHMM_TEMPS_ON_HEAP
# distutils: include_dirs = deps/
# cython: boundscheck = False

# NOTE: cython can use templated classes but not templated functions in 0.19.1,
# hence the wrapper class. no syntax for directly calling static members, though

import numpy as np
cimport numpy as np

from libc.stdint cimport int32_t
from libcpp.vector cimport vector
from libcpp cimport bool

# NOTE: using the cython.floating fused type with typed memory views generates
# all possible type combinations, is not intended here.
# https://groups.google.com/forum/#!topic/cython-users/zdlliIRF1a4
from cython cimport floating

from cython.parallel import prange

cdef extern from "hmm_messages.h":
    cdef cppclass hmmc[Type]: # NOTE: default states type is int32_t
        hmmc()
        void messages_backwards_log(
            bool hetero, int M, int T, Type *A, Type *aBl, Type *betal) nogil
        void messages_forwards_log(
            bool hetero, int M, int T,
            Type *A, Type *pi0, Type *aBl, Type *alphal) nogil
        Type messages_forwards_normalized(
            bool hetero, int M, int T,
            Type *A, Type *pi0, Type *aBl, Type *alphan) nogil
        void sample_forwards_log(
            bool hetero, int M, int T,
            Type *A, Type *pi0, Type *aBl, Type *betal,
            int32_t *stateseq, Type *randseq) nogil
        void sample_backwards_normalized(
            bool hetero, int M, int T, Type *AT, Type *alphan,
            int32_t *stateseq, Type *randseq) nogil
        void viterbi(
            int M, int T, Type *A, Type *pi0, Type *aBl, int32_t *stateseq) nogil
        Type expected_statistics_log(
            bool hetero, int M, int T,
            Type *log_trans_potential, Type *log_likelihood_potential,
            Type *alphal, Type *betal,
            Type *expected_states, Type *expected_transcounts)

def messages_backwards_log(
        A not None,
        floating[:,::1] aBl not None,
        np.ndarray[floating,ndim=2,mode="c"] betal not None,
        ):
    cdef hmmc[floating] ref
    cdef bool hetero = A.ndim == 3
    cdef floating[:,:,::1] _A = A if hetero else np.expand_dims(A, 0)

    ref.messages_backwards_log(
        hetero, A.shape[1], aBl.shape[0], &_A[0,0,0], &aBl[0,0], &betal[0,0])
    return betal

def messages_forwards_log(
        A not None,
        floating[:,::1] aBl not None,
        floating[::1] pi0 not None,
        np.ndarray[floating,ndim=2,mode="c"] alphal not None,
        ):
    cdef hmmc[floating] ref
    cdef bool hetero = A.ndim == 3
    cdef floating[:,:,::1] _A = A if hetero else np.expand_dims(A, 0)
    ref.messages_forwards_log(
        hetero, A.shape[1], aBl.shape[0], &_A[0,0,0], &pi0[0], &aBl[0,0],
        &alphal[0,0])
    return alphal

def sample_forwards_log(
        A not None,
        floating[:,::1] aBl not None,
        floating[::1] pi0 not None,
        floating[:,::1] betal not None,
        int32_t[::1] stateseq not None,
        ):
    cdef hmmc[floating] ref
    cdef bool hetero = A.ndim == 3
    cdef floating[:,:,::1] _A = A if hetero else np.expand_dims(A, 0)

    cdef floating[:] randseq
    if floating is double:
        randseq = np.random.random(size=aBl.shape[0]).astype(np.double)
    else:
        randseq = np.random.random(size=aBl.shape[0]).astype(np.float)

    ref.sample_forwards_log(
            hetero, A.shape[1], aBl.shape[0], &_A[0,0,0], &pi0[0], &aBl[0,0],
            &betal[0,0], &stateseq[0], &randseq[0])

    return np.asarray(stateseq)

def expected_statistics_log(
        log_trans_potential not None,
        np.ndarray[floating,ndim=2,mode='c'] log_likelihood_potential not None,
        np.ndarray[floating,ndim=2,mode='c'] alphal not None,
        np.ndarray[floating,ndim=2,mode='c'] betal not None,
        np.ndarray[floating,ndim=2,mode='c'] expected_states not None,
	expected_transcounts not None,
        ):

    cdef hmmc[floating] ref
    cdef bool hetero = log_trans_potential.ndim == 3
    cdef floating[:,:,::1] _A = log_trans_potential if hetero \
        else np.expand_dims(log_trans_potential, 0)
    cdef floating[:,:,::1] _expected_transcounts = \
    	 expected_transcounts if hetero \
         else np.expand_dims(expected_transcounts, 0)


    cdef floating log_normalizer = ref.expected_statistics_log(
            hetero,
            log_trans_potential.shape[1], alphal.shape[0],
            &_A[0,0,0],
            &log_likelihood_potential[0,0],
            &alphal[0,0],
            &betal[0,0],
            &expected_states[0,0],
            &_expected_transcounts[0,0,0])
    return expected_states, expected_transcounts, log_normalizer

def messages_forwards_normalized(
        A not None,
        floating[:,::1] aBl not None,
        floating[::1] pi0 not None,
        np.ndarray[floating,ndim=2,mode="c"] alphan not None,
        ):
    cdef hmmc[floating] ref
    cdef bool hetero = A.ndim == 3
    cdef floating[:,:,::1] _A = A if hetero else np.expand_dims(A, 0)

    cdef floating loglike = \
        ref.messages_forwards_normalized(
            hetero, A.shape[1], aBl.shape[0], &_A[0,0,0], &pi0[0],
            &aBl[0,0], &alphan[0,0])
    return alphan, loglike

def sample_backwards_normalized(
        AT not None,
        floating[:,::1] alphan not None,
        int32_t[::1] stateseq not None,
        ):
    cdef hmmc[floating] ref
    cdef bool hetero = AT.ndim == 3
    cdef floating[:,:,::1] _AT = AT if hetero else np.expand_dims(AT, 0)

    cdef floating[:] randseq
    if floating is double:
        randseq = np.random.random(size=alphan.shape[0]).astype(np.double)
    else:
        randseq = np.random.random(size=alphan.shape[0]).astype(np.float)

    ref.sample_backwards_normalized(
        hetero, AT.shape[1], alphan.shape[0], &_AT[0,0,0],
        &alphan[0,0], &stateseq[0], &randseq[0])

    return np.asarray(stateseq)

def viterbi(
        floating[:,::1] A not None,
        floating[:,::1] aBl not None,
        floating[::1] pi0 not None,
        int32_t[::1] stateseq not None,
        ):
    cdef hmmc[floating] ref
    ref.viterbi(A.shape[1],aBl.shape[0],&A[0,0],&pi0[0],&aBl[0,0],
                &stateseq[0])
    return np.asarray(stateseq)

