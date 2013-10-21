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

// TODO remove stack alignment trick stuff?

using namespace Eigen;
using namespace std;

// NOTE: numpy arrays are row-major by default, while Eigen is column-major; I
// worked with each's deafult alignment, so the notion of "row" and "column"
// get transposed here compared to numpy code

// NOTE: I wrote that when I was young and naive; it'd be better just to
// use the RowMajor flag with Eigen. TODO

// NOTE: on my test machine numpy heap arrays were always aligned, but I doubt
// that's guaranteed. Still, this code assumes alignment!

// NOTE: I assume alignment of stack arrays, too

namespace hmm {

    // Messages

    template <typename Type>
    void messages_backwards_log(int M, int T, Type *A, Type *aBl, Type *betal)
    {
        // inputs
        Map<Matrix<Type,Dynamic,Dynamic>,Aligned> eAT(A,M,M);
        Map<Matrix<Type,Dynamic,Dynamic>,Aligned> eaBl(aBl,M,T);

        // outputs
        Map<Matrix<Type,Dynamic,Dynamic>,Aligned> ebetal(betal,M,T);

        // locals
        Matrix<Type,Dynamic,Dynamic> eA(M,M);
        eA = eAT.transpose();
        Matrix<Type,Dynamic,1> thesum(M);
        Type cmax;

        // computation!
        ebetal.col(T-1).setZero();
        for (int t=T-2; t>=0; t--) {
            thesum = eaBl.col(t+1) + ebetal.col(t+1);
            cmax = thesum.maxCoeff();
            ebetal.col(t) = (eA * (thesum.array() - cmax).exp().matrix()).array().log() + cmax;
        }
    }

    template <typename Type>
    void messages_forwards_log(int M, int T, Type *A, Type *pi0, Type *aBl,
            Type *alphal)
    {
        // inputs
        Map<Matrix<Type,Dynamic,Dynamic>,Aligned> eAT(A,M,M);
        Map<Array<Type,Dynamic,1>,Aligned> epi0(pi0,M);
        Map<Matrix<Type,Dynamic,Dynamic>,Aligned> eaBl(aBl,M,T);

        // outputs
        Map<Matrix<Type,Dynamic,Dynamic>,Aligned> ealphal(alphal,M,T);

        // locals
        Type cmax;

        // computation!
        ealphal.col(0) = epi0.log() + eaBl.col(0).array();
        for (int t=0; t<T-1; t++) {
            cmax = ealphal.col(t).maxCoeff();
            ealphal.col(t+1) = (eAT * (ealphal.col(t).array()
                        - cmax).array().exp().matrix()).array().log()
                + cmax + eaBl.col(t+1).array();
        }
    }

    template <typename Type>
    Type messages_forwards_normalized(int M, int T, Type *A, Type *pi0, Type *aBl,
            Type *alphan)
    {
        Map<Matrix<Type,Dynamic,Dynamic>,Aligned> eAT(A,M,M);
        Map<Matrix<Type,Dynamic,Dynamic>,Aligned> eaBl(aBl,M,T);
        Map<Matrix<Type,Dynamic,1>,Aligned> epi0(pi0,M);

        Map<Array<Type,Dynamic,Dynamic>,Aligned> ealphan(alphan,M,T);

        Type logtot = 0.;
        Type cmax, norm;

        cmax = eaBl.col(0).maxCoeff();
        ealphan.col(0) = epi0.array() * (eaBl.col(0).array() - cmax).exp();
        norm = ealphan.col(0).sum();
        ealphan.col(0) /= norm;
        logtot += log(norm) + cmax;
        for (int t=0; t<T; t++) {
            cmax = eaBl.col(t+1).maxCoeff();
            ealphan.col(t+1) = (eAT * ealphan.col(t).matrix()).array()
                * (eaBl.col(t+1).array() - cmax).exp();
            norm = ealphan.col(t+1).sum();
            ealphan.col(t+1) /= norm;
            logtot += log(norm) + cmax;
        }

        return logtot;
    }

    // Sampling

    template <typename Type>
    void sample_forwards_log(
            int M, int T, Type *A, Type *pi0, Type *aBl, Type *betal, int32_t *stateseq)
    {
        // inputs
        Map<Matrix<Type,Dynamic,Dynamic>,Aligned> eAT(A,M,M);
        Map<Matrix<Type,Dynamic,Dynamic>,Aligned> eaBl(aBl,M,T);
        Map<Matrix<Type,Dynamic,Dynamic>,Aligned> ebetal(betal,M,T);
        Map<Matrix<Type,Dynamic,1>,Aligned> epi0(pi0,M);

        // locals
        int idx;
        Matrix<Type,Dynamic,1> nextstate_unsmoothed(M);
        Matrix<Type,Dynamic,1> logdomain(M);
        Matrix<Type,Dynamic,1> nextstate_distr(M);

        // code!
        nextstate_unsmoothed = epi0;
        for (idx=0; idx < T; idx++) {
            logdomain = ebetal.col(idx) + eaBl.col(idx);
            nextstate_distr = (logdomain.array() - logdomain.maxCoeff()).exp()
                * nextstate_unsmoothed.array();
            stateseq[idx] = util::sample_discrete(M,nextstate_distr.data());
            nextstate_unsmoothed = eAT.col(stateseq[idx]);
        }
    }

    template <typename Type>
    void sample_backwards_normalized(int M, int T, Type *A, Type *alphan,
            int32_t *stateseq)
    {
        Map<Matrix<Type,Dynamic,Dynamic>,Aligned> eAT(A,M,M);
        Matrix<Type,Dynamic,Dynamic> eA(M,M);
        eA = eAT.transpose();
        Map<Matrix<Type,Dynamic,Dynamic>,Aligned> ealphan(alphan,M,T);

        Array<Type,Dynamic,1> enext_potential(M);
        enext_potential.setOnes();
        Array<Type,Dynamic,1> etemp(M);

        for (int t=T-1; t>=0; t--) {
            etemp = enext_potential * ealphan.col(t).array();
            stateseq[t] = util::sample_discrete(M,etemp.data());
            enext_potential = eA.col(stateseq[t]);
        }
    }

    // Viterbi

    template <typename Type>
    void viterbi(
            int M, int T, Type *A, Type *pi0, Type *aBl,
            int32_t *stateseq)
    {
        // inputs
        Map<Matrix<Type,Dynamic,Dynamic>,Aligned> eA(A,M,M);
        Map<Matrix<Type,Dynamic,Dynamic>,Aligned> eaBl(aBl,M,T);
        Map<Array<Type,Dynamic,1>,Aligned> epi0(pi0,M);

        // locals
        MatrixXi args(M,T);
        Matrix<Type,Dynamic,Dynamic> eAl(M,M);
        eAl = eA.array().log();
        Matrix<Type,Dynamic,1> scores(M);
        Matrix<Type,Dynamic,1> prevscores(M);
        Matrix<Type,Dynamic,1> tempvec(M);
        int maxIndex;

        // computation!
        scores.setZero();
        for (int t=T-2; t>=0; t--) {
            for (int i=0; i<M; i++) {
                tempvec = eAl.col(i) + scores + eaBl.col(t+1);
                prevscores(i) = tempvec.maxCoeff(&maxIndex);
                args(i,t+1) = maxIndex;
            }
            scores = prevscores;
        }

        (scores.array() + epi0.log() + eaBl.col(0).array()).maxCoeff(stateseq);
        for (int t=1; t<T; t++) {
            stateseq[t] = args(stateseq[t-1],t);
        }
    }
}

#endif
