#ifndef HSMM_INTNEGBIN_H
#define HSMM_INTNEGBIN_H

#include <Eigen/Core>
#include <algorithm> // min
#include <limits> // infinity

#include "util.h"
#include "nptypes.h"

// TODO all HMM embedding / HSMM FIR+IIR stuff should be coded ONCE with a
// propagation callback that does the matrix multiplication and/or summing
// against other lags. what about the transformation to HSMM messages?
// with that abstractions, only one message-passing routine is needed. maybe
// something similar can be done with forward sampling.
//
// for now, just write what I've already done...


// forward sampling even with vanilla messages is O(T) !!
// EM can be O(T^2) though, unless we are smurt

// variational algorithm to construct low-order approx (instead of balanced
// truncation? or whatever nonnegative version?)

namespace hsmm
{
    using namespace std;
    using namespace Eigen;
    using namespace nptypes;

    // NOTE: this is just HMM message passing with a structured embedding; see
    // note at top of file for why this method shouldn't exist
    template <typename Type>
    Type messages_forwards_normalized(int M, int T, Type *A, Type *pi0, Type *aBl,
            int *rs, Type *ps, Type *alphan)
    {
        NPMatrix<Type> eA(A,M,M);
        NPArray<Type> eaBl(aBl,T,M);

        // NOTE: rtot = sum(rs); rstarts = concatenate(((0,),cumsum(rs[:-1])))
        int rtot = NPVector<int>(rs,M).sum();
        int rstarts[rtot];
        rstarts[0] = 0;
        for (int i=1; i<M; i++) {
            rstarts[i] = rstarts[i-1] + rs[i-1];
        }

        NPArray<Type> ealphan(alphan,T,rtot);


#ifdef HMM_TEMPS_ON_HEAP
        Array<Type,1,Dynamic> ein_potential(1,rtot);
        Array<Type,1,Dynamic> etrans_part(1,M);
#else
        Type in_potential_buf[rtot] __attribute__((aligned(16)));
        NPRowVectorArray<Type> ein_potential(in_potential_buf,rtot);
        Type trans_part_buf[M] __attribute__((aligned(16)));
        NPRowVector<Type> etrans_part(trans_part_buf,M);
#endif

        Type logtot = 0.;
        ein_potential = NPRowVectorArray<Type>(pi0,rtot);
        for (int t=0; t<T; t++) {
            // NOTE: alphan[t] = in_potential * (aBl[t] - cmax).exp()
            Type norm = 0.;
            Type cmax = eaBl.row(t).maxCoeff();
            for (int i=0; i<M; i++) {
                ealphan.row(t).segment(rstarts[i],rs[i]) =
                    ein_potential.segment(rstarts[i],rs[i]) * exp(eaBl(t,i)-cmax);
                norm += ealphan.row(t).segment(rstarts[i],rs[i]).sum();
            }

            // NOTE: logtot += log(alphan[t].sum()) + cmax; alphan[t] /= alphan[t].sum()
            if (likely(norm != 0)) {
                ealphan.row(t) /= norm;
                logtot += log(norm) + cmax;
            } else {
                ealphan.block(t,0,T-t,rtot).setZero();
                return -numeric_limits<Type>::infinity();
            }

            // NOTE: in_potential = alphan[t] * A, broken into bdiag and obdiag parts
            if (likely(t < T-1)) {
                for (int i=0; i<M; i++) {
                    ein_potential.segment(rstarts[i],rs[i]) =
                        ealphan.row(t).segment(rstarts[i],rs[i]) * ps[i];
                    ein_potential.segment(rstarts[i]+1,rs[i]-1) +=
                        ealphan.row(t).segment(rstarts[i],rs[i]-1) * (1-ps[i]);
                    etrans_part(i) = ealphan(t,rstarts[i] + rs[i] - 1) * (1-ps[i]);
                }
                etrans_part = etrans_part * eA;
                for (int i=0; i<M; i++) {
                    ein_potential(rstarts[i]) += etrans_part(i);
                }
            }
        }

        return logtot;
    }
}

// NOTE: this class exists for cython binding convenience

template <typename FloatType>
class inbhsmmc
{
    public:

    static FloatType messages_forwards_normalized(
            int M, int T, FloatType *A, FloatType *pi0, FloatType *aBl,
            int *rs, FloatType *ps, FloatType *alphan)
    { return hsmm::messages_forwards_normalized(M,T,A,pi0,aBl,rs,ps,alphan); }

};

#endif
