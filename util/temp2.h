#include <Eigen/Core>
#include <stdint.h> // int32_t
#include <omp.h> // omp_get_num_threads, omp_get_thread_num
#include <limits> // infinity
#include <iostream>

#include <mkl.h>

#include "nptypes.h"
#include "util.h"

using namespace Eigen;
using namespace nptypes;
using namespace std;

template <typename Type>
class dummy
{
    public:

    static void gmm_likes(
            int T, int Tblock, int N, int K, int D,
            Type * __restrict data, Type *__restrict weights,
            Type * __restrict Js, Type * __restrict mus_times_Js, Type * __restrict normalizers,
            int32_t * __restrict changepoints,
            Type * __restrict aBBl)
    {
        NPArray<Type> edata(data,T,D);
        NPMatrix<Type> eweights(weights,N,K);
        NPArray<Type> eaBBl(aBBl,Tblock,N);

        NPMatrix<Type> eJs(Js,N*K,D);
        NPMatrix<Type> emus_times_Js(mus_times_Js,N*K,D);
        NPVector<Type> enormalizers(normalizers,N*K);

        Eigen::initParallel();

#pragma omp parallel
        {
            Type temp_buf[N*K] __attribute__((aligned(16)));
            NPVector<Type> temp(temp_buf,N*K);
            Type temp2_buf[D] __attribute__((aligned(16)));
            NPVector<Type> temp2(temp2_buf,D);
            Type themax;

#pragma omp for
            for (int tbl=0; tbl<Tblock; tbl++) {
                int start = changepoints[2*tbl];
                int end = changepoints[2*tbl+1];
                memset(aBBl+N*tbl,0,N);

                for (int t=start; t<end; t++) {
                    if (likely((edata.row(t) == edata.row(t)).all())) { // TODO speed this up

                        __builtin_assume_aligned(normalizers,16);
                        double *foo = __builtin_assume_aligned(data + D*t,16);
                        size_t i;

                        asm("# NOTE1");
                        // temp2 = edata.row(t).square().matrix().transpose();
                        for (i=0; i < D - (D % 4); i+=4) {
                            temp2_buf[i] = foo[i] * foo[i];
                            temp2_buf[i+1] = foo[i+1] * foo[i+1];
                            temp2_buf[i+2] = foo[i+2] * foo[i+2];
                            temp2_buf[i+3] = foo[i+3] * foo[i+3];
                        }
                        for (; i < D; i+=2) {
                            temp2_buf[i] = foo[i] * foo[i];
                            temp2_buf[i+1] = foo[i+1] * foo[i+1];
                        }
                        asm("# NOTE1 END");

                        // temp = eJs * temp2;
                        asm("# NOTE2");
                        cblas_dgemv(CblasRowMajor,CblasNoTrans,
                                N*K,D,1.,Js,D,
                                temp2_buf,1,
                                0.,temp_buf,1);
                        asm("# NOTE2 END");

                        // temp -= (emus_times_Js * edata.row(t).matrix().transpose());
                        asm("# NOTE3");
                        cblas_dgemv(CblasRowMajor,CblasNoTrans,
                                N*K,D,-1.,mus_times_Js,D,
                                foo,1,
                                1.,temp_buf,1);
                        asm("# NOTE3 END");

                        // temp.noalias() += enormalizers;
                        asm("# NOTE4");
                        for (i=0; i < N*K - (N*K % 4); i+=4) {
                            temp_buf[i] += normalizers[i];
                            temp_buf[i+1] += normalizers[i+1];
                            temp_buf[i+2] += normalizers[i+2];
                            temp_buf[i+3] += normalizers[i+3];
                        }
                        for (; i < N*K; i+=2) {
                            temp_buf[i] += normalizers[i];
                            temp_buf[i+1] += normalizers[i+1];
                        }
                        asm("# NOTE4");

                        asm("# NOTE5");
                        for (int n=0; n<N; n++) {
                            themax = temp.segment(n*K,K).maxCoeff();
                            eaBBl(tbl,n) +=
                                log(eweights.row(n) * (temp.array().segment(n*K,K) - themax).exp().matrix())
                                + themax;
                        }
                        asm("# NOTE5 END");
                    }
                }
            }
        }
    }
};
