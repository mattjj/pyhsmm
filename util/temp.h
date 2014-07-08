#include <Eigen/Core>
#include <stdint.h> // int32_t
#include <omp.h> // omp_get_num_threads, omp_get_thread_num
#include <limits> // infinity

#include "nptypes.h"
#include "util.h"

using namespace Eigen;
using namespace nptypes;
using namespace std;

template <typename Type>
class dummy
{
    public:

    static void initParallel() {
        Eigen::initParallel();
    }

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

        Eigen::initParallel();

#pragma omp parallel
        {
            Type temp_buf[N*K] __attribute__((aligned(16)));
            NPVectorArray<Type> temp(temp_buf,N*K);
            Type themax;

#pragma omp for
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
    }

    static void hsmm_messages_reduction_verticalpartition(
            int T, int N, Type *betal, Type *cB, Type *dur_potentials, Type *out)
    {
        NPSubArray<Type> ebetal(betal,T,N);
        NPSubArray<Type> ecB(cB,T,N);
        NPSubArray<Type> edp(dur_potentials,T,N);

        Map<Array<Type,1,Dynamic> > eout(out,1,N);

        Eigen::initParallel();

#pragma omp parallel
        {
            if (omp_get_thread_num() < N) {
                int num_threads = omp_get_num_threads();
                int blocklen = 1 + ((N - 1) / min(N,num_threads));
                int start = blocklen * omp_get_thread_num();
                blocklen = min(blocklen, N-start);

                Type maxes_buf[blocklen] __attribute__((aligned(16)));
                Map<Array<Type,1,Dynamic>,Aligned> maxes(maxes_buf,1,blocklen);

#ifdef TEMPS_ON_STACK
                Type thesum_buf[T*blocklen] __attribute__((aligned(16)));
                NPArray<Type> thesum(thesum_buf,T,blocklen);
#else
                Array<Type,Dynamic,Dynamic> thesum(T,blocklen);
#endif

                thesum = ebetal.block(0,start,T,blocklen) + ecB.block(0,start,T,blocklen)
                    + edp.block(0,start,T,blocklen);
                maxes = thesum.colwise().maxCoeff();

                eout.segment(start,blocklen)
                    = (thesum.rowwise() - maxes).exp().colwise().sum().log() + maxes;
            }
        }
    }

    static void hsmm_messages_reduction_horizontalpartition(
            int T, int N, Type *betal, Type *cB, Type *dur_potentials, Type *out)
    {
        NPArray<Type> ebetal(betal,T,N);
        NPArray<Type> ecB(cB,T,N);
        NPArray<Type> edp(dur_potentials,T,N);

        Eigen::initParallel();

        int max_num_threads = omp_get_max_threads();
        Type out_buf[max_num_threads*N] __attribute__((aligned(16)));

        NPArray<Type> eout(out_buf,max_num_threads,N);
        eout.setConstant(-numeric_limits<Type>::infinity());

#pragma omp parallel
        {
            int thread_num = omp_get_thread_num();
            int num_threads = omp_get_num_threads();
            int blocklen = 1 + ((T - 1) / num_threads);
            int start = blocklen * thread_num;
            blocklen = min(blocklen, T-start);

            Type maxes_buf[N] __attribute__((aligned(16)));
            Map<Array<Type,1,Dynamic>,Aligned> maxes(maxes_buf,1,N);

#ifdef TEMPS_ON_STACK
            Type thesum_buf[blocklen*N] __attribute__((aligned(16)));
            NPArray<Type> thesum(thesum_buf,blocklen,N);
#else
            Array<Type,Dynamic,Dynamic> thesum(blocklen,N);
#endif

            thesum = ebetal.block(start,0,blocklen,N) + ecB.block(start,0,blocklen,N)
                + edp.block(start,0,blocklen,N);
            maxes = thesum.colwise().maxCoeff();

            eout.block(thread_num,0,1,N) =
                (thesum.rowwise() - maxes).exp().colwise().sum().log() + maxes;
        }

        Type maxes_buf[N] __attribute__((aligned(16)));
        Map<Array<Type,1,Dynamic>,Aligned> maxes(maxes_buf,1,N);

        maxes = eout.colwise().maxCoeff();
        eout.rowwise() -= maxes;
        Map<Array<Type,1,Dynamic>,Aligned>(out,1,N) =
            eout.exp().colwise().sum().log() + maxes;
    }
};

