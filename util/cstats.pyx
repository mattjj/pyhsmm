# distutils: extra_compile_args = -O3 -w
# cython: boundscheck = False
# cython: wraparound = False
# cython: cdivision = True

import numpy as np
cimport numpy as np

from libc.stdint cimport int32_t
from cython cimport floating, integral

from cython.parallel import prange

cdef inline int32_t csample_discrete_normalized(floating[::1] distn, floating u):
    cdef int i
    cdef int N = distn.shape[0]
    cdef floating tot = u

    for i in range(N):
        tot -= distn[i]
        if tot < 0:
            break

    return i

def sample_markov(
        int T,
        np.ndarray[floating, ndim=2, mode="c"] trans_matrix,
        np.ndarray[floating, ndim=1, mode="c"] init_state_distn
        ):
    cdef int32_t[::1] out = np.empty(T,dtype=np.int32)
    cdef floating[:,::1] A = trans_matrix / trans_matrix.sum(1)[:,None]
    cdef floating[::1] pi = init_state_distn / init_state_distn.sum()

    cdef floating[::1] randseq
    if floating is double:
        randseq = np.random.random(T).astype(np.double)
    else:
        randseq = np.random.random(T).astype(np.float)

    cdef int t
    out[0] = csample_discrete_normalized(pi,randseq[0])
    for t in range(1,T):
        out[t] = csample_discrete_normalized(A[out[t-1]],randseq[t])

    return np.asarray(out)

def sample_crp_tablecounts(
        floating concentration,
        integral[:,::1] customers,
        floating[::1] colweights,
        ):
    cdef integral[:,::1] m = np.zeros_like(customers)
    cdef int i, j, k
    cdef integral tot = np.sum(customers)

    cdef floating[::1] randseq
    if floating is double:
        randseq = np.random.random(tot).astype(np.double)
    else:
        randseq = np.random.random(tot).astype(np.float)

    tmp = np.empty_like(customers)
    tmp[0,0] = 0
    tmp.flat[1:] = np.cumsum(np.ravel(customers)[:customers.size-1],dtype=tmp.dtype)
    cdef integral[:,::1] starts = tmp

    with nogil:
        for i in prange(customers.shape[0]):
            for j in range(customers.shape[1]):
                for k in range(customers[i,j]):
                    m[i,j] += randseq[starts[i,j]+k] \
                        < (concentration * colweights[j]) / (k+concentration*colweights[j])

    return np.asarray(m)

def count_transitions(int32_t[::1] stateseq, int num_states):
    cdef int T = stateseq.shape[0]
    cdef int i
    cdef int32_t[:,::1] out = np.zeros((num_states,num_states),dtype=np.int32)
    for i in range(T-1):
        out[stateseq[i],stateseq[i+1]] += 1
    return np.asarray(out)

