# distutils: name = internals.hmm_messages_interface
# distutils: language = c++
# distutils: extra_compile_args = -O3 -march=native -g -w -DNDEBUG
# distutils: include_dirs = deps/Eigen3/

# TODO E step functions

# TODO default arguments of None don't work with np.ndarray typed arguments;
# file bug report with cython?

# NOTE: cython can use templated classes but not templated functions in 0.19.1,

import numpy as np
cimport numpy as np

import cython

from libc.stdint cimport int32_t
from cython cimport floating

cdef extern from "hmm_messages.h":
    cdef cppclass hmmc[Type]:
        hmmc()
        void messages_backwards_log (
            int M, int T, Type *A, Type *aBl, Type *betal)
        void messages_forwards_log(
            int M, int T, Type *A, Type *pi0, Type *aBl, Type *alphal)
        Type messages_forwards_normalized(
            int M, int T, Type *A, Type *pi0, Type *aBl, Type *alphan)
        void sample_forwards_log (
            int M, int T, Type *A, Type *pi0, Type *aBl, Type *betal, int32_t *stateseq)
        void sample_backwards_normalized(
            int M, int T, Type *AT, Type *alphan, int32_t *stateseq)
        void viterbi (
            int M, int T, Type *A, Type *pi0, Type *aBl, int32_t *stateseq)

def messages_backwards_log(
        np.ndarray[floating,ndim=2,mode="c"] A not None,
        np.ndarray[floating,ndim=2,mode="c"] aBl not None,
        np.ndarray[floating,ndim=2,mode="c"] betal not None,
        ):
    cdef hmmc[floating] ref
    ref.messages_backwards_log(A.shape[0],aBl.shape[0],&A[0,0],&aBl[0,0],&betal[0,0])
    return betal

def messages_forwards_log(
        np.ndarray[floating,ndim=2,mode="c"] A not None,
        np.ndarray[floating,ndim=2,mode="c"] aBl not None,
        np.ndarray[floating,ndim=1,mode="c"] pi0 not None,
        np.ndarray[floating,ndim=2,mode="c"] alphal not None,
        ):
    cdef hmmc[floating] ref
    ref.messages_forwards_log(A.shape[0],aBl.shape[0],&A[0,0],&pi0[0],&aBl[0,0],
            &alphal[0,0])
    return alphal

def messages_forwards_normalized(
        np.ndarray[floating,ndim=2,mode="c"] A not None,
        np.ndarray[floating,ndim=2,mode="c"] aBl not None,
        np.ndarray[floating,ndim=1,mode="c"] pi0 not None,
        np.ndarray[floating,ndim=2,mode="c"] alphan not None,
        ):
    cdef hmmc[floating] ref
    cdef floating loglike = \
            ref.messages_forwards_normalized(A.shape[0],aBl.shape[0],&A[0,0],&pi0[0],
                &aBl[0,0],&alphan[0,0])
    return alphan, loglike

def sample_forwards_log(
        np.ndarray[floating,ndim=2,mode="c"] A not None,
        np.ndarray[floating,ndim=2,mode="c"] aBl not None,
        np.ndarray[floating,ndim=1,mode="c"] pi0 not None,
        np.ndarray[floating,ndim=2,mode="c"] betal not None,
        np.ndarray[np.int32_t,ndim=1,mode="c"] stateseq not None,
        ):
    cdef hmmc[floating] ref
    ref.sample_forwards_log(A.shape[0],aBl.shape[0],&A[0,0],&pi0[0],&aBl[0,0],
            &betal[0,0],&stateseq[0])
    return stateseq

def sample_backwards_normalized(
        np.ndarray[floating,ndim=2,mode="c"] AT not None,
        np.ndarray[floating,ndim=2,mode="c"] alphan not None,
        np.ndarray[np.int32_t,ndim=1,mode="c"] stateseq not None,
        ):
    cdef hmmc[floating] ref
    ref.sample_backwards_normalized(AT.shape[0],alphan.shape[0],&AT[0,0],
            &alphan[0,0],&stateseq[0])
    return stateseq

def viterbi(
        np.ndarray[floating,ndim=2,mode="c"] A not None,
        np.ndarray[floating,ndim=2,mode="c"] aBl not None,
        np.ndarray[floating,ndim=1,mode="c"] pi0 not None,
        np.ndarray[np.int32_t,ndim=1,mode="c"] stateseq not None,
        ):
    cdef hmmc[floating] ref
    ref.viterbi(A.shape[0],aBl.shape[0],&A[0,0],&pi0[0],&aBl[0,0],
                &stateseq[0])
    return stateseq

