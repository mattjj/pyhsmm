#ifndef HMM_MESSAGES_H
#define HMM_MESSAGES_H

#ifndef MY_CSTDINT_H
#define MY_CSTDINT_H
#include <stdint.h>
#endif

#ifndef EIGEN_CORE_H
#include <Eigen/Core>
#endif

#ifndef UTIL_H
#include "util.h"
#endif

namespace hmm {

    // Messages

    void messages_backwards_log(int M, int T, double *A, double *aBl,
            double *betal);

    void messages_forwards_log(int M, int T, double *A, double *pi0, double *aBl,
            double *alphal);

    double messages_forwards_normalized(int M, int T, double *A, double *pi0,
            double *aBl, double *alphan);

    // Sampling

    void sample_forwards_log(int M, int T, double *A, double *pi0, double *aBl,
            double *betal, int32_t *stateseq);

    void sample_backwards_normalized(int M, int T, double *A, double *alphan,
            int32_t *stateseq);

    // Viterbi

    void viterbi(int M, int T, double *A, double *aBl, double *pi0,
            int32_t *stateseq);

}

#endif
