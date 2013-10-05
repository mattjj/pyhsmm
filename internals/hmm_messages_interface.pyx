# distutils: name = internals.hmm_messages_interface
# distutils: language = c++
# distutils: extra_compile_args = -O3 -march=native -w
# distutils: include_dirs = deps/Eigen3/

# NOTE: afaict, cython doesn't support templated methods (only templated
# classes), so the 'cdef extern from' block needs explicit instantiation.
# but the tests on the cython fused type cython.floating generate separate code
# paths at compile time! neato!

# NOTE: it seems cython.floating doesn't like default arguments (of None, at
# least) so outputs can't be allocated in this code TODO file bug report

# TODO E step functions

import numpy as np
cimport numpy as np

import cython

from libc.stdint cimport int32_t

cdef extern from "hmm_messages.h":
    # float versions

    void f_messages_backwards_log "hmm::messages_backwards_log" (
            int M, int T, float *A, float *aBl, float *betal)

    void f_messages_forwards_log "hmm::messages_forwards_log" (
            int M, int T, float *A, float *pi0, float *aBl, float *alphal)

    float f_messages_forwards_normalized "hmm::messages_forwards_normalized" (
            int M, int T, float *A, float *pi0, float *aBl, float *alphan)

    void f_sample_forwards_log "hmm::sample_forwards_log" (
            int M, int T,
            float *A, float *pi0, float *aBl, float *betal, int32_t *stateseq)

    void f_sample_backwards_normalized "hmm::sample_backwards_normalized" (
            int M, int T, float *A, float *alphan, int32_t *stateseq)

    void f_viterbi "hmm::viterbi" (
            int M, int T, float *A, float *pi0, float *aBl, int32_t *stateseq)

    # double versions

    void d_messages_backwards_log "hmm::messages_backwards_log" (
            int M, int T, double *A, double *aBl, double *betal)

    void d_messages_forwards_log "hmm::messages_forwards_log" (
            int M, int T, double *A, double *pi0, double *aBl, double *alphal)

    double d_messages_forwards_normalized "hmm::messages_forwards_normalized" (
            int M, int T, double *A, double *pi0, double *aBl, double *alphan)

    void d_sample_forwards_log "hmm::sample_forwards_log" (
            int M, int T,
            double *A, double *pi0, double *aBl, double *betal, int32_t *stateseq)

    void d_sample_backwards_normalized "hmm::sample_backwards_normalized" (
            int M, int T, double *A, double *alphan, int32_t *stateseq)

    void d_viterbi "hmm::viterbi" (
            int M, int T, double *A, double *pi0, double *aBl, int32_t *stateseq)

@cython.boundscheck(False)
@cython.wraparound(False)
def messages_backwards_log(
        np.ndarray[cython.floating,ndim=2,mode="c"] A not None,
        np.ndarray[cython.floating,ndim=2,mode="c"] aBl not None,
        np.ndarray[cython.floating,ndim=2,mode="c"] betal not None,
        ):
    if cython.floating is double:
        d_messages_backwards_log(A.shape[0],aBl.shape[0],&A[0,0],&aBl[0,0],&betal[0,0])
    else:
        f_messages_backwards_log(A.shape[0],aBl.shape[0],&A[0,0],&aBl[0,0],&betal[0,0])
    return betal

@cython.boundscheck(False)
@cython.wraparound(False)
def messages_forwards_log(
        np.ndarray[cython.floating,ndim=2,mode="c"] A not None,
        np.ndarray[cython.floating,ndim=2,mode="c"] aBl not None,
        np.ndarray[cython.floating,ndim=1,mode="c"] pi0 not None,
        np.ndarray[cython.floating,ndim=2,mode="c"] alphal not None,
        ):
    if cython.floating is double:
        d_messages_forwards_log(A.shape[0],aBl.shape[0],&A[0,0],&pi0[0],&aBl[0,0],
                &alphal[0,0])
    else:
        f_messages_forwards_log(A.shape[0],aBl.shape[0],&A[0,0],&pi0[0],&aBl[0,0],
                &alphal[0,0])
    return alphal

@cython.boundscheck(False)
@cython.wraparound(False)
def messages_forwards_normalized(
        np.ndarray[cython.floating,ndim=2,mode="c"] A not None,
        np.ndarray[cython.floating,ndim=2,mode="c"] aBl not None,
        np.ndarray[cython.floating,ndim=1,mode="c"] pi0 not None,
        np.ndarray[cython.floating,ndim=2,mode="c"] alphan not None,
        ):
    if cython.floating is double:
        loglike = d_messages_forwards_normalized(A.shape[0],aBl.shape[0],&A[0,0],&pi0[0],
                &aBl[0,0],&alphan[0,0])
    else:
        loglike = f_messages_forwards_normalized(A.shape[0],aBl.shape[0],&A[0,0],&pi0[0],
                &aBl[0,0],&alphan[0,0])
    return alphan, loglike

@cython.boundscheck(False)
@cython.wraparound(False)
def sample_forwards_log(
        np.ndarray[cython.floating,ndim=2,mode="c"] A not None,
        np.ndarray[cython.floating,ndim=2,mode="c"] aBl not None,
        np.ndarray[cython.floating,ndim=1,mode="c"] pi0 not None,
        np.ndarray[cython.floating,ndim=2,mode="c"] betal not None,
        np.ndarray[np.int32_t,ndim=1,mode="c"] stateseq not None,
        ):
    if cython.floating is double:
        d_sample_forwards_log(A.shape[0],aBl.shape[0],&A[0,0],&pi0[0],&aBl[0,0],
                &betal[0,0],&stateseq[0])
    else:
        f_sample_forwards_log(A.shape[0],aBl.shape[0],&A[0,0],&pi0[0],&aBl[0,0],
                &betal[0,0],&stateseq[0])
    return stateseq

@cython.boundscheck(False)
@cython.wraparound(False)
def sample_backwards_normalized(
        np.ndarray[cython.floating,ndim=2,mode="c"] A not None,
        np.ndarray[cython.floating,ndim=2,mode="c"] alphan not None,
        np.ndarray[np.int32_t,ndim=1,mode="c"] stateseq not None,
        ):
    if cython.floating is double:
        d_sample_backwards_normalized(A.shape[0],alphan.shape[0],&A[0,0],
                &alphan[0,0],&stateseq[0])
    else:
        f_sample_backwards_normalized(A.shape[0],alphan.shape[0],&A[0,0],
                &alphan[0,0],&stateseq[0])
    return stateseq

@cython.boundscheck(False)
@cython.wraparound(False)
def viterbi(
        np.ndarray[cython.floating,ndim=2,mode="c"] A not None,
        np.ndarray[cython.floating,ndim=2,mode="c"] aBl not None,
        np.ndarray[cython.floating,ndim=1,mode="c"] pi0 not None,
        np.ndarray[np.int32_t,ndim=1,mode="c"] stateseq not None,
        ):
    if cython.floating is double:
        d_viterbi(A.shape[0],aBl.shape[0],&A[0,0],&pi0[0],&aBl[0,0],
                &stateseq[0])
    else:
        f_viterbi(A.shape[0],aBl.shape[0],&A[0,0],&pi0[0],&aBl[0,0],
                &stateseq[0])
    return stateseq

