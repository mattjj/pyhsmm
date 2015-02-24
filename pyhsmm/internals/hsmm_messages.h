#ifndef HSMM_H
#define HSMM_H

#include <Eigen/Core>
#include <iostream> // cout, endl
#include <algorithm> // min

#include "util.h"
#include "nptypes.h"

namespace hsmm
{
    using namespace std;
    using namespace Eigen;
    using namespace nptypes;

    template <typename Type>
    void messages_backwards_log(
        int M, int T, Type *A, Type *aBl, Type *aDl, Type *aDsl,
        Type *betal, Type *betastarl, int right_censoring, int trunc)
    {
        NPMatrix<Type> eA(A,M,M);
        NPArray<Type> eaBl(aBl,T,M);
        NPArray<Type> eaDl(aDl,T,M);
        NPArray<Type> eaDsl(aDsl,T,M);

        NPArray<Type> ebetal(betal,T,M);
        NPArray<Type> ebetastarl(betastarl,T,M);

#ifdef HMM_TEMPS_ON_HEAP
        Array<Type,1,Dynamic> sumsofar(M);
        Array<Type,1,Dynamic> result(M);
        Array<Type,1,Dynamic> maxes(M);
#else
        Type sumsofar_buf[M] __attribute__((aligned(16)));
        NPRowVectorArray<Type> sumsofar(sumsofar_buf,M);
        Type result_buf[M] __attribute__((aligned(16)));
        NPRowVectorArray<Type> result(result_buf,M);
        Type maxes_buf[M] __attribute__((aligned(16)));
        NPRowVectorArray<Type> maxes(maxes_buf,M);
#endif

        Type cmax;
        ebetal.row(T-1).setZero();
        for(int t=T-1; t>=0; t--) {
            sumsofar.setZero();
            ebetastarl.row(t).setConstant(-1.0*numeric_limits<Type>::infinity());
            for(int tau=0; tau < min(trunc,T-t); tau++) {
                sumsofar += eaBl.row(t+tau);
                result = ebetal.row(t+tau) + sumsofar + eaDl.row(tau);
                maxes = ebetastarl.row(t).cwiseMax(result);
                ebetastarl.row(t) =
                    ((ebetastarl.row(t) - maxes).exp() + (result - maxes).exp()).log() + maxes;
            }
            if (right_censoring && T-t-1 < trunc) {
                result = eaBl.block(t,0,T-t,M).colwise().sum() + eaDsl.row(T-1-t);
                maxes = ebetastarl.row(t).cwiseMax(result);
                ebetastarl.row(t) =
                    ((ebetastarl.row(t) - maxes).exp() + (result - maxes).exp()).log() + maxes;
            }
            for(int i=0; i<M; i++) {
                if (ebetastarl(t,i) != ebetastarl(t,i)) {
                    ebetastarl(i,t) = -1.0*numeric_limits<Type>::infinity();
                }
            }
            if (likely(t > 0)) {
                cmax = ebetastarl.row(t).maxCoeff();
                ebetal.row(t-1) = (eA * (ebetastarl.row(t) - cmax).exp().matrix().transpose()
                        ).array().log() + cmax;
                for(int i=0; i<M; i++) {
                    if (ebetastarl(t,i) != ebetastarl(t,i)) {
                        ebetastarl(i,t) = -1.0*numeric_limits<Type>::infinity();
                    }
                }
            }
        }
    }

    template <typename Type>
    void messages_forwards_log(
        int M, int T, Type *A, Type *aBl, Type *aDl, Type *aDsl, Type *pil,
        Type *alphal, Type *alphastarl)
    {
        // TODO trunc, censoring

        NPMatrix<Type> eA(A,M,M);
        NPArray<Type> eaBl(aBl,T,M);
        NPArray<Type> eaDl(aDl,T,M);
        NPArray<Type> eaDsl(aDsl,T,M);
        NPRowVectorArray<Type> epil(pil,M);

        NPArray<Type> ealphal(alphal,T,M);
        NPArray<Type> ealphastarl(alphastarl,T,M);

#ifdef HMM_TEMPS_ON_HEAP
        Array<Type,1,Dynamic> sumsofar(M);
        Array<Type,1,Dynamic> result(M);
        Array<Type,1,Dynamic> maxes(M);
#else
        Type sumsofar_buf[M] __attribute__((aligned(16)));
        NPRowVectorArray<Type> sumsofar(sumsofar_buf,M);
        Type result_buf[M] __attribute__((aligned(16)));
        NPRowVectorArray<Type> result(result_buf,M);
        Type maxes_buf[M] __attribute__((aligned(16)));
        NPRowVectorArray<Type> maxes(maxes_buf,M);
#endif

        // TODO

    }

    template <typename FloatType, typename IntType>
    void sample_forwards_log(
        int M, int T, FloatType *A, FloatType *pi0, FloatType *caBl, FloatType *aDl,
        FloatType *betal, FloatType *betastarl, IntType *stateseq, FloatType *randseq)
    {
        NPArray<FloatType> eA(A,M,M);
        NPArray<FloatType> ecaBl(caBl,T,M);
        NPArray<FloatType> eaDl(aDl,T,M);
        NPArray<FloatType> ebetal(betal,T,M);
        NPArray<FloatType> ebetastarl(betastarl,T,M);
        NPVectorArray<IntType> estateseq(stateseq,T);

#ifdef HMM_TEMPS_ON_HEAP
        Array<FloatType,1,Dynamic> logdomain(M);
        Array<FloatType,1,Dynamic> nextstate_distr(M);
#else
        FloatType logdomain_buf[M] __attribute__((aligned(16)));
        NPRowVectorArray<FloatType> logdomain(logdomain_buf,M);
        FloatType nextstate_distr_buf[M] __attribute__((aligned(16)));
        NPRowVectorArray<FloatType> nextstate_distr(nextstate_distr_buf,M);
#endif

        int t = 0, dur, randseq_idx=0;
        IntType state;
        FloatType durprob, p_d_prior, p_d;

        nextstate_distr = NPRowVectorArray<FloatType>(pi0,M);
        while (t < T) {
            // use the messages to form the posterior over states
            logdomain = ebetastarl.row(t) - ebetastarl.row(t).maxCoeff();
            nextstate_distr *= logdomain.exp();
            if ((nextstate_distr == 0.).all()) {
                cout << "Warning: all-zero posterior state belief, following likelihood"
                    << endl;
                nextstate_distr = logdomain.exp();
            }

            // sample from nextstate_distr
            state = util::sample_discrete(M,nextstate_distr.data(),randseq[randseq_idx++]);

            // sample from duration pmf
            durprob = randseq[randseq_idx++];
            for(dur=0; durprob > 0. && t+dur < T; dur++) {
                p_d_prior = exp(eaDl(dur,state));
                if (0.0 == p_d_prior) {
                    continue;
                }
                p_d = p_d_prior * exp(ecaBl(t+dur+1,state) - ecaBl(t,state)
                            + ebetal(t+dur,state) - ebetastarl(t,state));
                durprob -= p_d;
            }
            // NOTE: if t+dur == T, the duration is censored; it will be fixed up in Python

            // set the output
            estateseq.segment(t,dur).setConstant(state);
            t += dur;
            nextstate_distr = eA.row(state);
        }
    }
}

// NOTE: this class exists for cyhton binding convenience

template <typename FloatType, typename IntType = int32_t>
class hsmmc
{
    public:

    static void messages_backwards_log(
        int M, int T, FloatType *A, FloatType *aBl, FloatType *aDl, FloatType *aDsl,
        FloatType *betal, FloatType *betastarl, bool right_censoring, int trunc)
    { hsmm::messages_backwards_log(M,T,A,aBl,aDl,aDsl,betal,betastarl,
            right_censoring,trunc); }

    static void sample_forwards_log(
        int M, int T, FloatType *A, FloatType *pi0, FloatType *aBl, FloatType *aDl,
        FloatType *betal, FloatType *betastarl,
        IntType *stateseq, FloatType *randseq)
    { hsmm::sample_forwards_log(M,T,A,pi0,aBl,aDl,betal,betastarl,stateseq,randseq); }
};

#endif
