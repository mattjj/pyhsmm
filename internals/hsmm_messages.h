#ifndef HSMM_H
#define HSMM_H

#include <Eigen/Core>
#include <iostream>
#include <stdlib.h>

#include "util.h"
#include "nptypes.h"

namespace hsmm
{
    using namespace Eigen;
    using namespace nptypes;

    using namespace std;

    template <typename Type>
    void messages_backwards_log(
        int M, int T, Type *A, Type *aBl, Type *aDl, Type *betal, Type *betastarl)
    {
        // NOTE: I haven't pulled the code here yet because there wasnt' a
        // speed advantage over python last I checked, but having a C++
        // implementation would make things easier to parallelize with OpenMP
        // (since a cython implementation needs numpy and hence the GIL, unless
        // one assumes a cblas is present, and then Eigen is still nicer)
        abort(); // TODO

        /*
        // inputs
        int etrunc = mytrunc;
        Map<MatrixXd> eaBl(aBl,%(M)d,%(T)d);
        Map<MatrixXd> eA(A,%(M)d,%(M)d);
        Map<MatrixXd> eaDl(aDl,%(M)d,%(T)d);
        Map<MatrixXd> eaDsl(aDsl,%(M)d,%(T)d);

        // outputs
        Map<MatrixXd> ebetal(betal,%(M)d,%(T)d);
        Map<MatrixXd> ebetastarl(betastarl,%(M)d,%(T)d);

        // locals
        VectorXd maxes(%(M)d), result(%(M)d), sumsofar(%(M)d);
        double cmax;

        // computation!
        for (int t = %(T)d-1; t >= 0; t--) {
            sumsofar.setZero();
            ebetastarl.col(t).setConstant(-1.0*numeric_limits<double>::infinity());
            for (int tau = 0; tau < min(etrunc,%(T)d-t); tau++) {
                sumsofar += eaBl.col(t+tau);
                result = ebetal.col(t+tau) + sumsofar + eaDl.col(tau);
                maxes = ebetastarl.col(t).cwiseMax(result);
                ebetastarl.col(t) = ((ebetastarl.col(t) - maxes).array().exp() + (result - maxes).array().exp()).log() + maxes.array();
            }
            // censoring calc
            if (%(T)d - t < etrunc) {
                result = eaBl.block(0,t,%(M)d,%(T)d-t).rowwise().sum() + eaDsl.col(%(T)d-1-t);
                maxes = ebetastarl.col(t).cwiseMax(result);
                ebetastarl.col(t) = ((ebetastarl.col(t) - maxes).array().exp() + (result - maxes).array().exp()).log() + maxes.array();
            }
            // nan issue
            for (int i = 0; i < %(M)d; i++) {
                if (ebetastarl(i,t) != ebetastarl(i,t)) {
                    ebetastarl(i,t) = -1.0*numeric_limits<double>::infinity();
                }
            }
            // betal calc
            if (t > 0) {
                cmax = ebetastarl.col(t).maxCoeff();
                ebetal.col(t-1) = (eA * (ebetastarl.col(t).array() - cmax).array().exp().matrix()).array().log() + cmax;
                for (int i = 0; i < %(M)d; i++) {
                    if (ebetal(i,t-1) != ebetal(i,t-1)) {
                        ebetal(i,t-1) = -1.0*numeric_limits<double>::infinity();
                    }
                }
            }
        }
        */
    }

    template <typename FloatType, typename IntType>
    void sample_forwards_log(
        int M, int T, FloatType *A, FloatType *pi0, FloatType *aBl, FloatType *aDl,
        FloatType *betal, FloatType *betastarl, IntType *stateseq)
    {
        abort(); // TODO

//        // inputs
//
//        Map<MatrixXd> ebetal(betal,M,T);
//        Map<MatrixXd> ebetastarl(betastarl,M,T);
//        Map<MatrixXd> eaBl(aBl,M,T);
//        Map<MatrixXd> eA(A,M,M);
//        Map<VectorXd> epi0(pi0,M);
//        Map<MatrixXd> eapmf(apmf,M,T);
//
//        // outputs
//
//        Map<VectorXi> estateseq(stateseq,T);
//        //VectorXi estateseq(T);
//        estateseq.setZero();
//
//        // locals
//        int idx, state, dur;
//        double durprob, p_d_marg, p_d, total;
//        VectorXd nextstate_unsmoothed(M);
//        VectorXd logdomain(M);
//        VectorXd nextstate_distr(M);
//
//        // code!
//        // don't think i need to seed... should include sys/time.h for this
//        // struct timeval time;
//        // gettimeofday(&time,NULL);
//        // srandom((time.tv_sec * 1000) + (time.tv_usec / 1000));
//
//        idx = 0;
//        nextstate_unsmoothed = epi0;
//
//        while (idx < T) {
//            logdomain = ebetastarl.col(idx).array() - ebetastarl.col(idx).maxCoeff();
//            nextstate_distr = logdomain.array().exp() * nextstate_unsmoothed.array();
//            if ((nextstate_distr.array() == 0.0).all()) {
//                std::cout << "Warning: this is a cryptic error message" << std::endl;
//                nextstate_distr = logdomain.array().exp();
//            }
//            // sample from nextstate_distr
//            {
//                total = nextstate_distr.sum() * (((double)random())/((double)RAND_MAX));
//                for (state = 0; (total -= nextstate_distr(state)) > 0; state++) ;
//            }
//
//            durprob = ((double)random())/((double)RAND_MAX);
//            dur = 0;
//            while (durprob > 0.0) {
//                if (dur > 2*T) {
//                    std::cout << "FAIL" << std::endl;
//                }
//
//                p_d_marg = (dur < T) ? eapmf(state,dur) : 1.0;
//                if (0.0 == p_d_marg) {
//                        dur += 1;
//                        continue;
//                }
//                if (idx+dur < T) {
//                        p_d = p_d_marg * (exp(eaBl.row(state).segment(idx,dur+1).sum()
//                                    + ebetal(state,idx+dur) - ebetastarl(state,idx)));
//                } else {
//                    break; // will be fixed in python
//                }
//                durprob -= p_d;
//                dur += 1;
//            }
//
//            estateseq.segment(idx,dur).setConstant(state);
//
//            nextstate_unsmoothed = eA.col(state);
//
//            idx += dur;
    }

    template <typename Type>
    void inbd_messages_backwards_log(
            int M, int T, Type *A, Type *aBl,
            int *start_indices, int *end_indices, Type *ps,
            Type *betal)
    {
        // TODO in python, don't transpose!
        // TODO in Python, don't scale the columns of AT (rows of A)

        NPMatrix<Type> eA(A,M,M);
        NPArray<Type> eaBl(aBl,T,M);
        NPArray<Type> ebetal(betal,T,M);

        Array<Type,Dynamic,1> incoming(M);
        Array<Type,Dynamic,1> thesum(M);
        Array<Type,Dynamic,1> esuperbetal(M);
        Type cmax, temp;

        esuperbetal.setZero();
        eA.array().colwise() *= NPVectorArray<Type>(ps,M); // NOTE: rescaled A rows
        for (int t=T-2; t>=0; t--) {
            // across-state transition part (sparse part)
            thesum = esuperbetal + eaBl.row(t+1);
            cmax = thesum.maxCoeff();
            incoming = (eA * (thesum - cmax).exp().matrix()).array().log() + cmax;

            // within-state transition part (bidiagonal block-diagonal part)
            for (int idx=0; idx<M; idx++) {
                int start = start_indices[idx];
                int end = end_indices[idx];
                Type pi = ps[idx];

                for (int i=start; i<end; i++) {
                    cmax = max(ebetal(i,t+1),ebetal(i+1,t+1));
                    if (isinf(cmax)) {
                        ebetal(i,t) = -INFINITY;
                    } else {
                        ebetal(i,t) =
                            log(pi*exp(ebetal(i,t+1)-cmax)+(1.0-pi)*exp(ebetal(i+1,t+1)-cmax))
                            + cmax + eaBl(idx,t+1);
                    }
                }
                temp = ebetal(end,t+1) + eaBl(idx,t+1);
                cmax = max(temp,incoming(idx));
                if (isinf(cmax)) {
                    ebetal(end,t) = -INFINITY;
                } else {
                    ebetal(end,t) =
                        log(pi*exp(temp-cmax)+exp(incoming(idx)-cmax)) + cmax;
                }

                esuperbetal(idx) = ebetal(start,t);
            }
        }
    }
}

// NOTE: this class exists for cyhton binding convenience

template <typename FloatType, typename IntType = int32_t>
class hsmmc
{
    public:

    static void messages_backwards_log(
        int M, int T, FloatType *A, FloatType *aBl, FloatType *aDl,
        FloatType *betal, FloatType *betastarl)
    { hsmm::messages_backwards_log(M,T,A,aBl,aDl,betal,betastarl); }

    static void sample_forwards_log(
        int M, int T, FloatType *A, FloatType *pi0, FloatType *aBl, FloatType *aDl,
        FloatType *betal, FloatType *betastarl,
        IntType *stateseq)
    { hsmm::sample_forwards_log(M,T,A,pi0,aBl,aDl,betal,betastarl,stateseq); }
};

#endif
