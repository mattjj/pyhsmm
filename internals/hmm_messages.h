#ifndef HMM_MESSAGES_H
#define HMM_MESSAGES_H

#include <Eigen/Core>
#include <stdint.h> // int32_t
#include <limits> // infinity

#include "nptypes.h"
#include "util.h"

// TODO template on the statesequence type as well!

namespace hmm
{
    using namespace Eigen;
    using namespace std;
    using namespace nptypes;

    // Messages

    template <typename Type>
    void messages_backwards_log(int M, int T, Type *A, Type *aBl,
            Type *betal)
    {
        NPMatrix<Type> eA(A,M,M);
        NPMatrix<Type> eaBl(aBl,T,M);

        NPMatrix<Type> ebetal(betal,T,M);

#ifndef HMM_TEMPS_ON_STACK
        Matrix<Type,Dynamic,1> thesum(M);
#else
        Type thesum_buf[M];
        NPVector<Type> thesum(thesum_buf,M);
#endif
        Type cmax;

        ebetal.row(T-1).setZero();
        for (int t=T-2; t>=0; t--) {
            thesum = (eaBl.row(t+1) + ebetal.row(t+1)).transpose();
            cmax = thesum.maxCoeff();
            ebetal.row(t) = (eA * (thesum.array() - cmax).exp().matrix()).array().log() + cmax;
        }
    }

    template <typename Type>
    void messages_forwards_log(int M, int T, Type *A, Type *pi0, Type *aBl,
            Type *alphal)
    {
        NPMatrix<Type> eA(A,M,M);
        NPArray<Type> epi0(pi0,1,M);
        NPArray<Type> eaBl(aBl,T,M);

        NPArray<Type> ealphal(alphal,T,M);

        Type cmax;

        ealphal.row(0) = epi0.log() + eaBl.row(0);
        for (int t=0; t<T-1; t++) {
            cmax = ealphal.row(t).maxCoeff();
            ealphal.row(t+1) = ((ealphal.row(t) - cmax).exp().matrix() * eA).array().log()
                + cmax + eaBl.row(t+1);
        }
    }

    template <typename Type>
    Type messages_forwards_normalized(int M, int T, Type *A, Type *pi0, Type *aBl,
            Type *alphan)
    {
        NPMatrix<Type> eA(A,M,M);
        NPArray<Type> eaBl(aBl,T,M);

        NPMatrix<Type> ealphan(alphan,T,M);

        Type logtot = 0.;
        Type cmax, norm;

#ifndef HMM_TEMPS_ON_STACK
        Matrix<Type,1,Dynamic> ein_potential(1,M);
#else
        Type in_potential_buf[M];
        NPMatrix<Type> ein_potential(in_potential,1,M);
#endif

        ein_potential = NPMatrix<Type>(pi0,1,M);
        for (int t=0; t<T; t++) {
            cmax = eaBl.row(t).maxCoeff();
            ealphan.row(t) = ein_potential.array() * (eaBl.row(t) - cmax).exp();
            norm = ealphan.row(t).sum();
            if (norm != 0) {
                ealphan.row(t) /= norm;
                logtot += log(norm) + cmax;
            } else {
                // NOTE: messages are invalid here
                return -numeric_limits<Type>::infinity();
            }
            ein_potential = ealphan.row(t) * eA;
        }
        return logtot;
    }

    // Sampling

    // TODO use nptypes
    template <typename Type>
    void sample_forwards_log( int M, int T, Type *A, Type *pi0, Type *aBl, Type *betal,
            int32_t *stateseq)
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

    // TODO use nptypes
    template <typename Type>
    void sample_backwards_normalized(int M, int T, Type *AT, Type *alphan,
            int32_t *stateseq)
    {
        Map<Matrix<Type,Dynamic,Dynamic>,Aligned> eA(AT,M,M);
        Map<Matrix<Type,Dynamic,Dynamic>,Aligned> ealphan(alphan,M,T);

        Array<Type,Dynamic,1> etemp(M);

        stateseq[T-1] = util::sample_discrete(M,ealphan.col(T-1).data());
        for (int t=T-2; t>=0; t--) {
            etemp = eA.col(stateseq[t+1]).array() * ealphan.col(t).array();
            stateseq[t] = util::sample_discrete(M,etemp.data());
        }
    }

    // Viterbi

    // TODO use nptypes
    template <typename Type>
    void viterbi(int M, int T, Type *A, Type *pi0, Type *aBl,
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

// NOTE: this class exists for cython binding convenience

template <typename Type>
class hmmc
{
    public:

    static void messages_backwards_log(int M, int T, Type *A, Type *aBl,
            Type *betal)
    { hmm::messages_backwards_log(M,T,A,aBl,betal); }

    static void messages_forwards_log(int M, int T, Type *A, Type *pi0, Type *aBl,
            Type *alphal)
    { hmm::messages_forwards_log(M,T,A,pi0,aBl,alphal); }

    static void sample_forwards_log(int M, int T, Type *A, Type *pi0, Type *aBl, Type *betal,
            int32_t *stateseq)
    { hmm::sample_forwards_log(M,T,A,pi0,aBl,betal,stateseq); }

    static Type messages_forwards_normalized(int M, int T, Type *A, Type *pi0, Type *aBl,
            Type *alphan)
    { return hmm::messages_forwards_normalized(M,T,A,pi0,aBl,alphan); }

    static void sample_backwards_normalized(int M, int T, Type *AT, Type *alphan,
            int32_t *stateseq)
    { hmm::sample_backwards_normalized(M,T,AT,alphan,stateseq); }

    static void viterbi(int M, int T, Type *A, Type *pi0, Type *aBl,
            int32_t *stateseq)
    { hmm::viterbi(M,T,A,pi0,aBl,stateseq); }
};

#endif
