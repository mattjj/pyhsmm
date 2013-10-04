# distutils: sources = hmm_messages.cpp util.cpp
# distutils: language = c++
# distutils: extra_compile_args = -O3 -w -march=native
# distutils: include_dirs = ../deps/Eigen3/

# TODO E step functions
# TODO normalized messages

import numpy as np
cimport numpy as np

import cython

from libc.stdint cimport int32_t

cdef extern from "hmm_messages.h" namespace "hmm":
    void c_messages_backwards_log "hmm::messages_backwards_log" (
            int M, int T, double *A, double *aBl, double *betal)

    void c_messages_forwards_log "hmm::messages_forwards_log" (
            int M, int T, double *A, double *pi0, double *aBl, double *alphal)

    double c_messages_forwards_normalized "hmm::messages_forwards_normalized" (
            int M, int T, double *A, double *pi0, double *aBl, double *alphan)

    void c_sample_forwards_log "hmm::sample_forwards_log" (
            int M, int T,
            double *A, double *pi0, double *aBl, double *betal, int32_t *stateseq)

    void c_sample_backwards_normalized "hmm::sample_backwards_normalized" (
            int M, int T, double *A, double *alphan, int32_t *stateseq)

    void c_viterbi "hmm::viterbi" (
            int M, int T, double *A, double *pi0, double *aBl, int32_t *stateseq)

@cython.boundscheck(False)
@cython.wraparound(False)
def messages_backwards_log(
        np.ndarray[np.double_t,ndim=2,mode="c"] A not None,
        np.ndarray[np.double_t,ndim=2,mode="c"] aBl not None,
        np.ndarray[np.double_t,ndim=2,mode="c"] betal=None,
        ):
    if betal is None:
        betal = np.empty_like(aBl)
    c_messages_backwards_log(A.shape[0],aBl.shape[0],&A[0,0],&aBl[0,0],&betal[0,0])
    return betal

@cython.boundscheck(False)
@cython.wraparound(False)
def messages_forwards_log(
        np.ndarray[np.double_t,ndim=2,mode="c"] A not None,
        np.ndarray[np.double_t,ndim=2,mode="c"] aBl not None,
        np.ndarray[np.double_t,ndim=1,mode="c"] pi0 not None,
        np.ndarray[np.double_t,ndim=2,mode="c"] alphal=None,
        ):
    if alphal is None:
        alphal = np.empty_like(aBl)
    c_messages_forwards_log(A.shape[0],aBl.shape[0],&A[0,0],&pi0[0],&aBl[0,0],
            &alphal[0,0])
    return alphal

@cython.boundscheck(False)
@cython.wraparound(False)
def messages_forwards_normalized(
        np.ndarray[np.double_t,ndim=2,mode="c"] A not None,
        np.ndarray[np.double_t,ndim=2,mode="c"] aBl not None,
        np.ndarray[np.double_t,ndim=1,mode="c"] pi0 not None,
        np.ndarray[np.double_t,ndim=2,mode="c"] alphan=None,
        ):
    if alphan is None:
        alphan = np.empty_like(aBl)
    loglike = c_messages_forwards_normalized(A.shape[0],aBl.shape[0],&A[0,0],&pi0[0],
            &aBl[0,0],&alphan[0,0])
    return alphan, loglike

@cython.boundscheck(False)
@cython.wraparound(False)
def sample_forwards_log(
        np.ndarray[np.double_t,ndim=2,mode="c"] A not None,
        np.ndarray[np.double_t,ndim=2,mode="c"] aBl not None,
        np.ndarray[np.double_t,ndim=1,mode="c"] pi0 not None,
        np.ndarray[np.double_t,ndim=2,mode="c"] betal not None,
        np.ndarray[np.int32_t,ndim=1,mode="c"] stateseq=None,
        ):
    if stateseq is None:
        stateseq = np.empty(aBl.shape[0],dtype='int32')
    c_sample_forwards_log(A.shape[0],aBl.shape[0],&A[0,0],&pi0[0],&aBl[0,0],
            &betal[0,0],&stateseq[0])
    return stateseq

@cython.boundscheck(False)
@cython.wraparound(False)
def sample_backwards_normalized(
        np.ndarray[np.double_t,ndim=2,mode="c"] A not None,
        np.ndarray[np.double_t,ndim=2,mode="c"] alphan not None,
        np.ndarray[np.int32_t,ndim=1,mode="c"] stateseq=None,
        ):
    if stateseq is None:
        stateseq = np.empty(alphan.shape[0],dtype='int32')
    c_sample_backwards_normalized(A.shape[0],alphan.shape[0],&A[0,0],
            &alphan[0,0],&stateseq[0])
    return stateseq

@cython.boundscheck(False)
@cython.wraparound(False)
def viterbi(
        np.ndarray[np.double_t,ndim=2,mode="c"] A not None,
        np.ndarray[np.double_t,ndim=2,mode="c"] aBl not None,
        np.ndarray[np.double_t,ndim=1,mode="c"] pi0 not None,
        np.ndarray[np.int32_t,ndim=1,mode="c"] stateseq=None,
        ):
    if stateseq is None:
        stateseq = np.empty(aBl.shape[0],dtype='int32')
    c_viterbi(A.shape[0],aBl.shape[0],&A[0,0],&pi0[0],&aBl[0,0],
            &stateseq[0])
    return stateseq

