#include <Eigen/Core>
#include <stdint.h> // int32_t
#include <iostream> // cout, endl

#include "nptypes.h"
#include "util.h"

using namespace Eigen;
using namespace nptypes;
using namespace std;

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
            int T, int Tblock, int N, int K, int D,
            Type *data, Type *weights,
            Type *Js, Type *mus_times_Js, Type *normalizers,
            int32_t *changepoints,
            Type *aBBl)
    {
        NPArray<Type> edata(data,T,D);
        NPMatrix<Type> eweights(weights,N,K);
        NPArray<Type> eaBBl(aBBl,T,N);

        NPMatrix<Type> eJs(Js,N*K,D);
        NPMatrix<Type> emus_times_Js(mus_times_Js,N*K,D);
        NPVectorArray<Type> enormalizers(normalizers,N*K);

        Array<Type,Dynamic,1> temp(N*K,1);
        Type themax;

        for (int tbl=0; tbl<Tblock; tbl++) {
            int start = changepoints[2*tbl];
            int end = changepoints[2*tbl+1];
            eaBBl.row(tbl).setZero();

            for (int t=start; t<end; t++) {
                if (likely((edata.row(t) == edata.row(t)).all())) {
                    temp = (eJs * edata.row(t).square().matrix().transpose()).array();
                    temp -= (emus_times_Js * edata.row(t).matrix().transpose()).array();
                    temp += enormalizers;

                    for (int n=0; n<N; n++) {
                        themax = temp.segment(n*K,K).maxCoeff();
                        eaBBl(tbl,n) +=
                            log(eweights.row(n) * (temp.segment(n*K,K) - themax).exp().matrix())
                            + themax;
                    }
                }
            }
        }
    }
};

