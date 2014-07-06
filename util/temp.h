#include <Eigen/Core>
#include <stdint.h> // int32_t

#include "nptypes.h"
#include "util.h"

using namespace Eigen;
using namespace nptypes;

template <typename Type>
class dummy
{
    public:

    static void getstats(
            int M, int T, int D, int32_t *stateseq, Type *data,
            Type *stats)
    {
        NPArray<Type> edata(data,T,D);
        NPArray<Type> estats(stats,M,2*D+1);

        for (int t=0; t < T; t++) {
            if (likely(edata(t,0) == edata(t,0))) {
                estats.block(stateseq[t],0,1,D) += edata.row(t);
                estats.block(stateseq[t],D,1,D) += edata.row(t).square();
                estats(stateseq[t],2*D) += 1;
            }
        }
    }

    static void gmm_likes(
            int T, int N, int K, int D,
            Type *data, Type *sigmas, Type *mus, Type *weights,
            int32_t *changepoints,
            Type *aBBl)
    {
        NPArray<Type> edata(data,T,D);
        NPArray<Type> esigmas(sigmas,N,K*D);
        NPArray<Type> emus(mus,N,K*D);
        NPArray<Type> eweights(weights,N,K);
        NPArray<Type> eaBBl(aBBl,T,N);
    }
};

