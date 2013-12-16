# distutils: name = internals.hsmm_messages_interface
# distutils: language = c++
# distutils: extra_compile_args = -O3 -march=native -w -DNDEBUG
# distutils: include_dirs = deps/Eigen3/

import numpy as np
cimport numpy as np

import cython

from libc.stdint cimport int32_t
from cython cimport floating

cdef extern from "hsmm_messages.h":
    cdef cppclass hsmmc[Type]:
        hsmmc()
        void messages_backwards_log(
            int M, int T, Type *A, Type *aBl, Type *aDl,
            Type *betal, Type *betastarl)
        void sample_forwards_log(
            int M, int T, Type *A, Type *pi0, Type *aBl, Type *aDl,
            Type *betal, Type *betastarl, int32_t *stateseq)

def messages_backwards_log(
        np.ndarray[floating,ndim=2,mode="c"] A not None,
        np.ndarray[floating,ndim=2,mode="c"] aBl not None,
        np.ndarray[floating,ndim=2,mode="c"] aDl not None,
        np.ndarray[floating,ndim=2,mode="c"] betal not None,
        np.ndarray[floating,ndim=2,mode="c"] betastarl not None):
    cdef hsmmc[floating] ref
    ref.messages_backwards_log(A.shape[0],aBl.shape[0],&A[0,0],
            &aBl[0,0],&aDl[0,0],&betal[0,0],&betastarl[0,0])
    return betal, betastarl

def sample_forwards_log(
        np.ndarray[floating,ndim=2,mode="c"] A not None,
        np.ndarray[floating,ndim=2,mode="c"] aBl not None,
        np.ndarray[floating,ndim=2,mode="c"] aDl not None,
        np.ndarray[floating,ndim=1,mode="c"] pi0 not None,
        np.ndarray[floating,ndim=2,mode="c"] betal not None,
        np.ndarray[floating,ndim=2,mode="c"] betastarl not None,
        np.ndarray[np.int32_t,ndim=1,mode="c"] stateseq not None,
        ):
    cdef hsmmc[floating] ref
    ref.sample_forwards_log(A.shape[0],aBl.shape[0],&A[0,0],&pi0[0],
            &aBl[0,0],&aDl[0,0],&betal[0,0],&betastarl[0,0],&stateseq[0])
    return stateseq

