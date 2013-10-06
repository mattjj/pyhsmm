#ifndef SUBHMM_MESSAGES_H
#define SUBHMM_MESSAGES_H

#ifndef MY_CSTDINT_H
#define MY_CSTDINT_H
#include <stdint.h>
#endif

#ifndef EIGEN_CORE_H
#include <Eigen/Core>
#endif

#ifndef MY_VECTOR_H
#define MY_VECTOR_H
#include <vector>
#endif

#ifndef MY_LIMITS_H
#define MY_LIMITS_H
#include <limits>
#endif

using namespace Eigen;
using namespace std;

// NOTE: no doubles here! TODO template things

namespace subhmm {
    typedef Map<Matrix<float,Dynamic,Dynamic,RowMajor>,Aligned> NPMatrix;
    typedef Map<Matrix<float,Dynamic,Dynamic,RowMajor> > NPSubMatrix;
    typedef Map<Array<float,Dynamic,Dynamic,RowMajor>,Aligned> NPArray;
    typedef Map<Array<float,Dynamic,Dynamic,RowMajor> > NPSubArray;

    typedef Map<Matrix<float,Dynamic,1>,Aligned> NPVector;
    typedef Map<Matrix<float,Dynamic,1> > NPSubVector;
    typedef Map<Array<float,Dynamic,1>,Aligned> NPVectorArray;
    typedef Map<Array<float,Dynamic,1> > NPSubVectorArray;

    inline
    float just_fast_mult(
            int N, int32_t *Nsubs, int32_t *rs, float *ps,
            NPMatrix &esuper_trans,
            vector<NPMatrix> &esub_transs, vector<NPVector> &esub_inits,
            int *blocksizes, int *blockstarts,
            float *v, float *out);

    inline
    float just_fast_left_mult(
            int N, int32_t *Nsubs, int32_t *rs, float *ps,
            NPMatrix &esuper_trans,
            vector<NPMatrix> &esub_transs, vector<NPVector> &esub_inits,
            int *blocksizes, int *blockstarts,
            float *v, float *out);

    float messages_backwards_normalized(
            int T, int bigN, int N, int32_t *Nsubs,
            int32_t *rs, float *ps, float *super_trans,
            vector<float*>& sub_transs, vector<float*>& sub_inits,
            vector<float*>& aBls,
            float *betan);

    float messages_forwards_normalized(
            int T, int bigN, int N, int32_t *Nsubs,
            int32_t *rs, float *ps, float *super_trans, float *init_state_distn,
            std::vector<float*>& sub_transs, std::vector<float*>& sub_inits,
            std::vector<float*>& aBls,
            float *alphan);

    // these next ones are for testing

    void fast_mult(
            int N, int32_t *Nsubs, int32_t *rs, float *ps,
            float *super_trans, vector<float*>& sub_transs, vector<float*>& sub_inits,
            float *v, float *out);

    void fast_left_mult(
            int N, int32_t *Nsubs, int32_t *rs, float *ps,
            float *super_trans, vector<float*>& sub_transs, vector<float*>& sub_inits,
            float *v, float *out);
}

#endif

