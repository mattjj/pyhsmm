# distutils: language = c++
# distutils: extra_compile_args = -O3 -fopenmp -std=c++11 -DEIGEN_NO_MALLOC
# distutils: extra_link_args = -fopenmp
# cython: boundscheck = False

# TODO with new Cython, don't need all this vector packing stuff (thanks to the
# patch I submitted last year!)

import numpy as np
cimport numpy as np

from libcpp.vector cimport vector
from libcpp cimport bool
from libc.stdint cimport int32_t, int64_t
from cython cimport floating, integral
from cython.parallel import prange

cdef extern from "ghmm_resampling.h":
    cdef cppclass dummy[Type]:
        dummy()
        Type resample_ghmm(
            int M, int T, int D,
            Type *pi_0, Type *A,
            Type *natparams, Type *normalizers,
            Type *data,
            Type *stats, int32_t *counts, int32_t *transcounts, int32_t *stateseq,
            Type *randseq, Type *alphan) nogil
        void initParallel()

def resample_arhmm(
        list pi_0s,
        list As,
        floating[:,:,::1] params,
        floating[::1] normalizers,
        list datas,
        list stateseqs,
        list randseqs,
        list alphans):
    cdef int i, j
    cdef dummy[floating] ref

    cdef int M = params.shape[0]   # number of states
    cdef int K = len(datas)        # number of sequences
    cdef int D = datas[0].shape[1] # dimension of data (unstrided)
    cdef int32_t[::1] Ts = np.array([d.shape[0] for d in datas]).astype('int32')

    cdef vector[int32_t*] stateseqs_v
    cdef int32_t[:] temp
    for i in range(K):
        temp = stateseqs[i]
        stateseqs_v.push_back(&temp[0])

    cdef vector[floating*] randseqs_v
    cdef floating[:] temp2
    for i in range(K):
        temp2 = randseqs[i]
        randseqs_v.push_back(&temp2[0])

    cdef vector[floating*] datas_v
    cdef floating[:,:] temp3
    for i in range(K):
        temp3 = datas[i]
        datas_v.push_back(&temp3[0,0])

    cdef vector[floating*] alphans_v
    cdef floating[:,:] temp4
    for i in range(K):
        temp4 = alphans[i]
        alphans_v.push_back(&temp4[0,0])

    cdef vector[floating*] As_v
    for i in range(K):
        temp4 = As[i]
        As_v.push_back(&temp4[0,0])

    cdef vector[floating*] pi_0s_v
    for i in range(K):
        temp2 = pi_0s[i]
        pi_0s_v.push_back(&temp2[0])

    # NOTE: 2*K for false sharing
    cdef int32_t[:,::1] ns = np.zeros((2*K,M),dtype='int32')
    cdef int32_t[:,:,::1] transcounts = np.zeros((2*K,M,M),dtype='int32')
    cdef floating[:,:,:,::1] stats
    cdef floating[::1] likes
    if floating is double:
        stats = np.zeros((2*K,M,params.shape[1],params.shape[2]),dtype='float64')
        likes = np.zeros(K,dtype='float64')
    else:
        stats = np.zeros((2*K,M,params.shape[1],params.shape[2]),dtype='float32')
        likes = np.zeros(K,dtype='float32')

    ref.initParallel()
    with nogil:
        for j in prange(K+1):
            if j != 0:
                i = j-1
                likes[i] = ref.resample_arhmm(
                        M,Ts[i],D,
                        pi_0s_v[i],As_v[i],
                        &params[0,0,0],&normalizers[0],
                        datas_v[i],
                        &stats[2*i,0,0,0],&ns[2*i,0],&transcounts[2*i,0,0],
                        stateseqs_v[i], randseqs_v[i],alphans_v[i])

    allstats = []
    for statmat, n in zip(np.sum(stats,0),np.sum(ns,0)):
        x, xxT = statmat[0, 1:], statmat[1:,1:]

        out = np.zeros((D+2,D+2))
        out[:D,:D] = xxT
        out[-2,:D] = out[:D,-2] = x
        out[-2,-2] = out[-1,-1] = n

        allstats.append(out)

    return allstats, np.sum(transcounts,axis=0), np.asarray(likes)
