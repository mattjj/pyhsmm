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

    static Type resample_ghmm(
            int M, int T, int D,
            Type *pi_0, Type *A,
            Type *natparams, Type *normalizers,
            Type *data,
            Type *stats, int32_t *counts, int32_t *transcounts, int32_t *stateseq,
            Type *randseq,
            Type *alphan)
    {
        NPMatrix<Type> edata(data, T, D);
        NPMatrix<Type> eA(A,M,M);

        int sz = D + 1;
        NPMatrix<Type> enatparams(natparams,M*sz,sz);
        NPMatrix<Type> estats(stats,M*sz,sz);
        NPMatrix<int32_t> etranscounts(transcounts,M,M);

        NPMatrix<Type> ealphan(alphan,T,M);

        Type temp_buf[sz] __attribute__((aligned(16)));
        NPVector<Type> etemp(temp_buf,sz);
        Type data_buff[sz] __attribute__((aligned(16)));
        NPRowVector<Type> data_buf(data_buff,sz);
        Type in_potential_buf[M] __attribute__((aligned(16)));
        NPRowVector<Type> ein_potential(in_potential_buf,M);
        Type likes_buf[M] __attribute__((aligned(16)));
        NPRowVector<Type> elikes(likes_buf,M);
        bool good_data[T];

        Type norm, cmax, logtot = 0.;
        data_buff[0] = 1;  // TODO having this 1 at the start may hurt packet ops

        // likelihoods and forward messages
        ein_potential = NPMatrix<Type>(pi_0,1,M);
        for (int t=0; t < T; t++) {
            if (good_data[t] = likely((edata.row(t).array() == edata.row(t).array()).all())) {
                for (int m=0; m < M; m++) {
                    data_buf.segment(1, D) = edata.row(t);
                    etemp.noalias() =
                        enatparams.block(m*sz,0,sz,sz) * data_buf.transpose();
                    elikes(m) = etemp.dot(data_buf.transpose()) - normalizers[m];
                }

                cmax = elikes.maxCoeff();
                ealphan.row(t) = ein_potential.array() * (elikes.array() - cmax).exp();
                norm = ealphan.row(t).sum();
                ealphan.row(t) /= norm;
                logtot += log(norm) + cmax;
            } else {
                ealphan.row(t) = ein_potential;
            }

            ein_potential = ealphan.row(t) * eA;
        }

        // backward sampling and stats gathering
        ein_potential.setOnes();
        int next_state = 0;
        for (int t=T-1; t >= 0; t--) {
            elikes = ein_potential.array() * ealphan.row(t).array();
            stateseq[t] = util::sample_discrete<Type>(M,elikes.data(),randseq[t]);
            etranscounts(stateseq[t],next_state) += 1;

            ein_potential = eA.col(stateseq[t]).transpose();
            next_state = stateseq[t];

            if (good_data[t]) {
                counts[stateseq[t]] += 1;

                data_buf.segment(1, D) = edata.row(t);
                estats.block(stateseq[t]*sz,0,sz,sz)
                    .template selfadjointView<Lower>().rankUpdate(data_buf.transpose(),1.);
            }
        }

        // undo the extra trans count
        etranscounts(stateseq[T-1],0) -= 1;

        // symmetrize statistics (asymmetric storage due to rankUpdate)
        for (int m=0; m < M; m++) {
            estats.block(m*sz,0,sz,sz).template triangularView<Upper>()
                = estats.block(m*sz,0,sz,sz).transpose();
        }

        return logtot;
    }
};
