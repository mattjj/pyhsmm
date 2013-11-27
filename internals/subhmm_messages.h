#ifndef SUBHMM_MESSAGES_H
#define SUBHMM_MESSAGES_H

#include <inttypes.h>
#include <vector>
#include <limits>
#include <Eigen/Core>

#if (defined _OPENMP)
#include <omp.h>
#endif

#ifndef UTIL_H
#include "util.h"
#endif

// TODO remove using namespace from header files, you noob!
using namespace Eigen;
using namespace std;

// NOTE: no doubles here! TODO template things, including the typedefs (type
// aliases? or just typdefs inside classes without C++11?)
// TODO move those nice typedefs to hmm_messages.h
// TODO make a macro for numpy alignment assumptions
// TODO make a macro for stack-alignment directive (Julia source has an example)

namespace subhmm {

    // typedef int64_t idx_t; // TODO use this type for indices, esp linear ones

    typedef Map<Matrix<float,Dynamic,Dynamic,RowMajor>,Aligned> NPMatrix;
    typedef Map<Matrix<float,Dynamic,Dynamic,RowMajor> > NPSubMatrix;
    typedef Map<Array<float,Dynamic,Dynamic,RowMajor>,Aligned> NPArray;
    typedef Map<Array<float,Dynamic,Dynamic,RowMajor> > NPSubArray;

    typedef Map<Matrix<float,Dynamic,1>,Aligned> NPVector;
    typedef Map<Matrix<float,Dynamic,1> > NPSubVector;
    typedef Map<Array<float,Dynamic,1>,Aligned> NPVectorArray;
    typedef Map<Array<float,Dynamic,1> > NPSubVectorArray;

    // fast matrix-vector products, called in the other functions

    inline
    float matrix_vector_mult(
            int N, int32_t *Nsubs, int32_t *rs, float *ps,
            NPMatrix &esuper_trans,
            vector<NPMatrix> &esub_transs, vector<NPVector> &esub_inits,
            int *blocksizes, int *blockstarts,
            float *v, float *out);

    inline
    float vector_matrix_mult(
            int N, int32_t *Nsubs, int32_t *rs, float *ps,
            NPMatrix &esuper_trans,
            vector<NPMatrix> &esub_transs, vector<NPVector> &esub_inits,
            int *blocksizes, int *blockstarts,
            float *v, float *out);

    // interface called from Python

    float messages_backwards_normalized(
            int T, int bigN, int N, int32_t *Nsubs,
            int32_t *rs, float *ps, float *super_trans, float *init_state_distn,
            vector<float*>& sub_transs, vector<float*>& sub_inits, vector<float*>& aBls,
            float *betan);

    float messages_forwards_normalized(
            int T, int bigN, int N, int32_t *Nsubs,
            int32_t *rs, float *ps, float *super_trans, float *init_state_distn,
            vector<float*>& sub_transs, vector<float*>& sub_inits, vector<float*>& aBls,
            float *alphan);

    void sample_backwards_normalized(
        int T, int bigN,
        float *alphan, int32_t *indptr, int32_t *indices, float *bigA_data,
        int32_t *stateseq);

    void generate_states(
        int T, int bigN, float *pi_0,
        int32_t *indptr, int32_t *indices, float *bigA_data,
        int32_t *stateseq);

    void steady_state(
        int N, int32_t *Nsubs, int32_t *rs, float *ps,
        float *super_trans, vector<float*>& sub_transs, vector<float*>& sub_inits,
        float *v, int niter);

    // onlysuper stuff

    float messages_forwards_normalized_onlysuper(
        int T, int onlysuperN, int bigN, int N, int32_t *Nsubs,
        int32_t *rs, float *ps, float *super_trans, float *init_state_distn,
        vector<float*>& sub_transs, vector<float*>& sub_inits, vector<float*>& aBls,
        float *alphan);

    // changepoints stuff

    // inline
    // float vector_matrix_mult_inside_segment(
    //         int N, int32_t *Nsubs, int32_t *rs, float *ps,
    //         vector<NPMatrix> &esub_transs,
    //         int *blocksizes, int *blockstarts,
    //         float *v, float *out);

    // float messages_forwards_normalized_changepoints(
    //         int T, int bigN, int N, int32_t *Nsubs,
    //         int32_t *rs, float *ps, float *super_trans, float *init_state_distn,
    //         vector<float*>& sub_transs, vector<float*>& sub_inits, vector<float*>& aBls,
    //         int32_t *starts, int32_t *blocklens, int Tblock,
    //         float *alphan);

    // these next functions are for testing from Python

    void test_matrix_vector_mult(
            int N, int32_t *Nsubs, int32_t *rs, float *ps,
            float *super_trans, vector<float*>& sub_transs, vector<float*>& sub_inits,
            float *v, float *out);

    void test_vector_matrix_mult(
            int N, int32_t *Nsubs, int32_t *rs, float *ps,
            float *super_trans, vector<float*>& sub_transs, vector<float*>& sub_inits,
            float *v, float *out);
}

#endif

