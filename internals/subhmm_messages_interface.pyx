# distutils: name = internals.subhmm_messages_interface
# distutils: sources = internals/subhmm_messages.cpp
# distutils: extra_compile_args = -O3 -w -g0 -march=native -DEIGEN_DONT_PARALLELIZE -DNDEBUG
# distutils: language = c++
# distutils: include_dirs = deps/Eigen3/
# cython: boundscheck=False
# cython: nonecheck=False
# cython: wraparound=False

import numpy as np
cimport numpy as np

import cython
from libc.stdint cimport int32_t
from libcpp.vector cimport vector

# TODO instead of passing everything, should I pass the class object directly and
# unpack it here?
# TODO can I include Eigen here and construct Eigen types to pass? then I could
# pack a struct instead of passing all these arguments...

cdef extern from "subhmm_messages.h" namespace "std":
    float f_messages_backwards_normalized "subhmm::messages_backwards_normalized" (
        int T, int bigN, int N, int32_t *Nsubs,
        int32_t *rs, float *ps, float *super_trans, float *init_state_distn,
        vector[float*]& sub_transs, vector[float*]& sub_inits, vector[float*]& aBls,
        float *betan)

    float f_messages_forwards_normalized "subhmm::messages_forwards_normalized" (
        int T, int bigN, int N, int32_t *Nsubs,
        int32_t *rs, float *ps, float *super_trans, float *init_state_distn,
        vector[float*]& sub_transs, vector[float*]& sub_inits, vector[float*]& aBls,
        float *alphan)

    float f_sample_backwards_normalized "subhmm::sample_backwards_normalized" (
        int T, int bigN,
        float *alphan, int32_t *indptr, int32_t *indices, float *bigA_data,
        int32_t *stateseq)

    void f_generate_states "subhmm::generate_states" (
        int T, int bigN, float *pi_0,
        int32_t *indptr, int32_t *indices, float *bigA_data,
        int32_t *stateseq)

    void f_steady_state "subhmm::steady_state" (
        int N, int32_t *Nsubs, int32_t *rs, float *ps,
        float *super_trans, vector[float*]& sub_transs, vector[float*]& sub_inits,
        float *v, int niter)

    float f_messages_forwards_normalized_changepoints "subhmm::messages_forwards_normalized_changepoints" (
        int T, int bigN, int N, int32_t *Nsubs,
        int32_t *rs, float *ps, float *super_trans, float *init_state_distn,
        vector[float*]& sub_transs, vector[float*]& sub_inits, vector[float*]& aBls,
        int32_t *starts, int32_t *blocklens, int Tblock,
        float *alphan)

    # testing

    void f_test_matrix_vector_mult "subhmm::test_matrix_vector_mult" (
        int N, int32_t *Nsubs, int32_t *rs, float *ps,
        float *super_trans, vector[float*]& sub_transs, vector[float*]& sub_inits,
        float *v, float *out)

    void f_test_vector_matrix_mult "subhmm::test_vector_matrix_mult" (
        int N, int32_t *Nsubs, int32_t *rs, float *ps,
        float *super_trans, vector[float*]& sub_transs, vector[float*]& sub_inits,
        float *v, float *out)

def messages_backwards_normalized(
        np.ndarray[np.float32_t,ndim=2,mode='c'] super_trans not None,
        np.ndarray[np.float32_t,ndim=1,mode='c'] init_state_distn not None,
        np.ndarray[np.int32_t,ndim=1,mode='c'] rs not None,
        np.ndarray[np.float32_t,ndim=1,mode='c'] ps not None,
        list sub_transs,
        list sub_initstates,
        list aBls,
        np.ndarray[np.float32_t,ndim=2,mode='c'] betan = None):

    # create Nsubs array
    cdef np.ndarray[np.int32_t,ndim=1,mode='c'] Nsubs
    Nsubs = np.ascontiguousarray([s.shape[0] for s in sub_transs],dtype='int32')

    # pack sub_transs (list of numpy arrays) into a std::vector<float *>
    cdef vector[float*] sub_transs_vect
    cdef np.ndarray[np.float32_t,ndim=2,mode='c'] temp
    for i in xrange(len(sub_transs)):
        temp = sub_transs[i]
        sub_transs_vect.push_back(&temp[0,0])

    # pack sub_initstates (list of numpy arrays) into a std::vector<float *>
    cdef vector[float*] sub_initstates_vect
    cdef np.ndarray[np.float32_t,ndim=1,mode='c'] temp2
    for i in xrange(len(sub_initstates)):
        temp2 = sub_initstates[i]
        sub_initstates_vect.push_back(&temp2[0])

    # pack aBls (list of numpy arrays) into a std::vector<float *>
    cdef vector[float*] aBls_vect
    cdef np.ndarray[np.float32_t,ndim=2,mode='c'] temp3
    for i in xrange(len(sub_initstates)):
        temp3 = aBls[i]
        aBls_vect.push_back(&temp3[0,0])

    # allocate output
    cdef int T = aBls[0].shape[0]
    cdef int bigN = sum([r*Nsub for r,Nsub in zip(rs,Nsubs)])
    if betan is None:
        betan = np.empty((T,bigN),dtype='float32')

    # call the routine
    loglike = f_messages_backwards_normalized(
            T,bigN,super_trans.shape[0],&Nsubs[0],
            &rs[0],&ps[0],&super_trans[0,0],&init_state_distn[0],
            sub_transs_vect,sub_initstates_vect,aBls_vect,
            &betan[0,0])

    return betan, loglike

def messages_forwards_normalized(
        np.ndarray[np.float32_t,ndim=2,mode='c'] super_trans not None,
        np.ndarray[np.float32_t,ndim=1,mode='c'] init_state_distn not None,
        np.ndarray[np.int32_t,ndim=1,mode='c'] rs not None,
        np.ndarray[np.float32_t,ndim=1,mode='c'] ps not None,
        list sub_transs,
        list sub_initstates,
        list aBls,
        np.ndarray[np.float32_t,ndim=2,mode='c'] alphan = None):

    # create Nsubs array
    cdef np.ndarray[np.int32_t,ndim=1,mode='c'] Nsubs
    Nsubs = np.ascontiguousarray([s.shape[0] for s in sub_transs],dtype='int32')

    # pack sub_transs (list of numpy arrays) into a std::vector<float *>
    cdef vector[float*] sub_transs_vect
    cdef np.ndarray[np.float32_t,ndim=2,mode='c'] temp
    for i in xrange(len(sub_transs)):
        temp = sub_transs[i]
        sub_transs_vect.push_back(&temp[0,0])

    # pack sub_initstates (list of numpy arrays) into a std::vector<float *>
    cdef vector[float*] sub_initstates_vect
    cdef np.ndarray[np.float32_t,ndim=1,mode='c'] temp2
    for i in xrange(len(sub_initstates)):
        temp2 = sub_initstates[i]
        sub_initstates_vect.push_back(&temp2[0])

    # pack aBls (list of numpy arrays) into a std::vector<float *>
    cdef vector[float*] aBls_vect
    cdef np.ndarray[np.float32_t,ndim=2,mode='c'] temp3
    for i in xrange(len(aBls)):
        temp3 = aBls[i]
        aBls_vect.push_back(&temp3[0,0])

    # allocate output
    cdef int T = aBls[0].shape[0]
    cdef int bigN = sum([r*Nsub for r,Nsub in zip(rs,Nsubs)])
    if alphan is None:
        alphan = np.empty((T,bigN),dtype='float32')

    # call the routine
    loglike = f_messages_forwards_normalized(
            T,bigN,super_trans.shape[0],&Nsubs[0],
            &rs[0],&ps[0],&super_trans[0,0],&init_state_distn[0],
            sub_transs_vect,sub_initstates_vect,aBls_vect,
            &alphan[0,0])

    return alphan, loglike

def messages_forwards_normalized_changepoints(
        np.ndarray[np.float32_t,ndim=2,mode='c'] super_trans not None,
        np.ndarray[np.float32_t,ndim=1,mode='c'] init_state_distn not None,
        np.ndarray[np.int32_t,ndim=1,mode='c'] rs not None,
        np.ndarray[np.float32_t,ndim=1,mode='c'] ps not None,
        list sub_transs,
        list sub_initstates,
        list aBls,
        np.ndarray[np.int32_t,ndim=1,mode='c'] segmentstarts not None,
        np.ndarray[np.int32_t,ndim=1,mode='c'] segmentlens not None,
        np.ndarray[np.float32_t,ndim=2,mode='c'] alphan = None):

    # create Nsubs array
    cdef np.ndarray[np.int32_t,ndim=1,mode='c'] Nsubs
    Nsubs = np.ascontiguousarray([s.shape[0] for s in sub_transs],dtype='int32')

    # pack sub_transs (list of numpy arrays) into a std::vector<float *>
    cdef vector[float*] sub_transs_vect
    cdef np.ndarray[np.float32_t,ndim=2,mode='c'] temp
    for i in xrange(len(sub_transs)):
        temp = sub_transs[i]
        sub_transs_vect.push_back(&temp[0,0])

    # pack sub_initstates (list of numpy arrays) into a std::vector<float *>
    cdef vector[float*] sub_initstates_vect
    cdef np.ndarray[np.float32_t,ndim=1,mode='c'] temp2
    for i in xrange(len(sub_initstates)):
        temp2 = sub_initstates[i]
        sub_initstates_vect.push_back(&temp2[0])

    # pack aBls (list of numpy arrays) into a std::vector<float *>
    cdef vector[float*] aBls_vect
    cdef np.ndarray[np.float32_t,ndim=2,mode='c'] temp3
    for i in xrange(len(aBls)):
        temp3 = aBls[i]
        aBls_vect.push_back(&temp3[0,0])

    # allocate output
    cdef int T = aBls[0].shape[0]
    cdef int bigN = sum([r*Nsub for r,Nsub in zip(rs,Nsubs)])
    if alphan is None:
        alphan = np.empty((T,bigN),dtype='float32')

    # call the routine
    loglike = f_messages_forwards_normalized_changepoints(
            T,bigN,super_trans.shape[0],&Nsubs[0],
            &rs[0],&ps[0],&super_trans[0,0],&init_state_distn[0],
            sub_transs_vect,sub_initstates_vect,aBls_vect,
            &segmentstarts[0], &segmentlens[0], segmentstarts.shape[0],
            &alphan[0,0])

    return alphan, loglike

def sample_backwards_normalized(
        np.ndarray[np.float32_t,ndim=2,mode='c'] alphan not None,
        np.ndarray[np.int32_t,ndim=1,mode='c'] indptr not None,
        np.ndarray[np.int32_t,ndim=1,mode='c'] indices not None,
        np.ndarray[np.float32_t,ndim=1,mode='c'] bigA_data not None,
        np.ndarray[np.int32_t,ndim=1,mode='c'] stateseq not None):
    f_sample_backwards_normalized(alphan.shape[0],alphan.shape[1],&alphan[0,0],
            &indptr[0],&indices[0],&bigA_data[0],&stateseq[0])
    return stateseq

def generate_states(
        csr_trans_matrix,
        np.ndarray[np.float32_t,ndim=1,mode='c'] pi_0 not None,
        np.ndarray[np.int32_t,ndim=1,mode='c'] big_stateseq not None):

    cdef np.ndarray[np.int32_t,ndim=1,mode='c'] indptr = csr_trans_matrix.indptr
    cdef np.ndarray[np.int32_t,ndim=1,mode='c'] indices = csr_trans_matrix.indices
    cdef np.ndarray[np.float32_t,ndim=1,mode='c'] data = csr_trans_matrix.data

    f_generate_states(big_stateseq.shape[0],
            csr_trans_matrix.shape[0],&pi_0[0],
            &indptr[0],&indices[0],&data[0],
            &big_stateseq[0])

def steady_state(
        np.ndarray[np.float32_t,ndim=1,mode='c'] v not None,
        np.ndarray[np.float32_t,ndim=2,mode='c'] super_trans not None,
        np.ndarray[np.int32_t,ndim=1,mode='c'] rs not None,
        np.ndarray[np.float32_t,ndim=1,mode='c'] ps not None,
        list sub_transs,
        list sub_initstates,
        int niter):
    # create Nsubs array
    cdef np.ndarray[np.int32_t,ndim=1,mode='c'] Nsubs
    Nsubs = np.ascontiguousarray([s.shape[0] for s in sub_transs],dtype='int32')

    # pack sub_transs (list of numpy arrays) into a std::vector<float *>
    cdef vector[float*] sub_transs_vect
    cdef np.ndarray[np.float32_t,ndim=2,mode='c'] temp
    for i in xrange(len(sub_transs)):
        temp = sub_transs[i]
        sub_transs_vect.push_back(&temp[0,0])

    # pack sub_initstates (list of numpy arrays) into a std::vector
    cdef vector[float*] sub_initstates_vect
    cdef np.ndarray[np.float32_t,ndim=1,mode='c'] temp2
    for i in xrange(len(sub_initstates)):
        temp2 = sub_initstates[i]
        sub_initstates_vect.push_back(&temp2[0])

    # call the routine
    f_steady_state(super_trans.shape[0],&Nsubs[0],&rs[0],&ps[0],&super_trans[0,0],
            sub_transs_vect,sub_initstates_vect,&v[0],niter)

# NOTE; these next ones are for testing

def test_matrix_vector_mult(
        np.ndarray[np.float32_t,ndim=1,mode='c'] v not None,
        np.ndarray[np.float32_t,ndim=2,mode='c'] super_trans not None,
        np.ndarray[np.int32_t,ndim=1,mode='c'] rs not None,
        np.ndarray[np.float32_t,ndim=1,mode='c'] ps not None,
        list sub_transs,
        list sub_initstates):

    # create Nsubs array
    cdef np.ndarray[np.int32_t,ndim=1,mode='c'] Nsubs
    Nsubs = np.ascontiguousarray([s.shape[0] for s in sub_transs],dtype='int32')

    # pack sub_transs (list of numpy arrays) into a std::vector<float *>
    cdef vector[float*] sub_transs_vect
    cdef np.ndarray[np.float32_t,ndim=2,mode='c'] temp
    for i in xrange(len(sub_transs)):
        temp = sub_transs[i]
        sub_transs_vect.push_back(&temp[0,0])

    # pack sub_initstates (list of numpy arrays) into a std::vector
    cdef vector[float*] sub_initstates_vect
    cdef np.ndarray[np.float32_t,ndim=1,mode='c'] temp2
    for i in xrange(len(sub_initstates)):
        temp2 = sub_initstates[i]
        sub_initstates_vect.push_back(&temp2[0])

    # allocate output
    cdef np.ndarray[np.float32_t,ndim=1,mode='c'] out = np.zeros(v.shape[0],dtype='float32')

    # call the routine
    f_test_matrix_vector_mult(super_trans.shape[0],&Nsubs[0],&rs[0],&ps[0],&super_trans[0,0],
            sub_transs_vect,sub_initstates_vect,&v[0],&out[0])

    return out

def test_vector_matrix_mult(
        np.ndarray[np.float32_t,ndim=1,mode='c'] v not None,
        np.ndarray[np.float32_t,ndim=2,mode='c'] super_trans not None,
        np.ndarray[np.int32_t,ndim=1,mode='c'] rs not None,
        np.ndarray[np.float32_t,ndim=1,mode='c'] ps not None,
        list sub_transs,
        list sub_initstates):

    # create Nsubs array
    cdef np.ndarray[np.int32_t,ndim=1,mode='c'] Nsubs
    Nsubs = np.ascontiguousarray([s.shape[0] for s in sub_transs],dtype='int32')

    # pack sub_transs (list of numpy arrays) into a std::vector<float *>
    cdef vector[float*] sub_transs_vect
    cdef np.ndarray[np.float32_t,ndim=2,mode='c'] temp
    for i in xrange(len(sub_transs)):
        temp = sub_transs[i]
        sub_transs_vect.push_back(&temp[0,0])

    # pack sub_initstates (list of numpy arrays) into a std::vector
    cdef vector[float*] sub_initstates_vect
    cdef np.ndarray[np.float32_t,ndim=1,mode='c'] temp2
    for i in xrange(len(sub_initstates)):
        temp2 = sub_initstates[i]
        sub_initstates_vect.push_back(&temp2[0])

    # allocate output
    cdef np.ndarray[np.float32_t,ndim=1,mode='c'] out = np.zeros(v.shape[0],dtype='float32')

    # call the routine
    f_test_vector_matrix_mult(super_trans.shape[0],&Nsubs[0],&rs[0],&ps[0],&super_trans[0,0],
            sub_transs_vect,sub_initstates_vect,&v[0],&out[0])

    return out

