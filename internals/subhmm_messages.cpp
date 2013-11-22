#include "subhmm_messages.h"

#include <iostream>
#include <algorithm>

using namespace Eigen;
using namespace std;

using namespace subhmm;

// TODO many functions (after the first two) repeat the same "setup" code to
// wrap things in Eigen types; that should be factored out somehow, I guess by
// unpacking into a struct. better yet, if i can include Eigen in the pyx file,
// i could pack the struct in there

inline
float subhmm::matrix_vector_mult(
        int N, int32_t *Nsubs, int32_t *rs, float *ps,
        NPMatrix &esuper_trans,
        vector<NPMatrix> &esub_transs, vector<NPVector> &esub_inits,
        int *blocksizes, int *blockstarts,
        float *v, float *out)
{
    // these two lines just force eincomings to be on the stack
    float incomings[N]; // NOTE: assuming stack pointer alignment for this guy too
    NPVector eincomings(incomings,N);

    float sum_of_result = 0.;

    // NOTE: the next two loops are each parallelizeable over N

    for (int i=0; i<N; i++) {
        eincomings(i) = esub_inits[i].dot(NPSubVector(v + blockstarts[i],Nsubs[i]));
    }

    for (int i=0; i < N; i++) {
        int blockstart = blockstarts[i];
        int blocksize = blocksizes[i];
        int32_t Nsub = Nsubs[i];
        int32_t r = rs[i];
        float p = ps[i];
        float temparr[r*Nsub] __attribute__((aligned(16)));
        NPMatrix &subtrans = esub_transs[i];

        NPSubMatrix rv(v + blockstart,r,Nsub);
        NPSubMatrix rout(out + blockstart,r,Nsub);

        // within-block
        NPMatrix temp(temparr,r,Nsub);
        temp.noalias() = rv * subtrans.transpose();
        rout = (temp.array() * (1-p)).matrix();
        rout.block(0,0,r-1,Nsub) += p * temp.block(1,0,r-1,Nsub);

        // across-block
        NPSubVectorArray(out+blockstart+blocksize-Nsub,Nsub)
            += p * esuper_trans.row(i).dot(eincomings);

        // track sum
        sum_of_result += rout.sum();
    }

    return sum_of_result;
}

inline
float subhmm::vector_matrix_mult(
        int N, int32_t *Nsubs, int32_t *rs, float *ps,
        NPMatrix &esuper_trans_T,
        vector<NPMatrix> &esub_transs, vector<NPVector> &esub_inits,
        int *blocksizes, int *blockstarts,
        float *v, float *out)
{
    // these two lines just force eincomings to be on the stack
    float incomings[N] __attribute__ ((aligned(16)));
    NPVector eincomings(incomings,N);

    float sum_of_result = 0.;

    // NOTE: the next two loops are each parallelizeable over N

    for (int i=0; i<N; i++) {
        incomings[i] = ps[i] * NPSubVector(
                v + blockstarts[i] + blocksizes[i] - Nsubs[i],Nsubs[i]).sum();
    }

    for (int i=0; i < N; i++) {
        int blockstart = blockstarts[i];
        int32_t Nsub = Nsubs[i];
        int32_t r = rs[i];
        float p = ps[i];
        float temparr[r*Nsub] __attribute__((aligned(16)));
        NPMatrix &subtrans = esub_transs[i];
        NPVector &pi = esub_inits[i];

        NPSubMatrix rv(v + blockstart,r,Nsub);
        NPSubMatrix rout(out + blockstart,r,Nsub);

        // within-block
        NPMatrix temp(temparr,r,Nsub);
        temp.noalias() = rv * subtrans;
        rout = (temp.array() * (1-p)).matrix();
        rout.block(1,0,r-1,Nsub) += p * temp.block(0,0,r-1,Nsub);

        // across-block
        NPSubVector(out+blockstart,Nsub)
            += (esuper_trans_T.row(i).dot(eincomings)) * pi;

        // track sum
        sum_of_result += rout.sum();
    }

    return sum_of_result;
}

float subhmm::messages_backwards_normalized(
        int T, int bigN, int N, int32_t *Nsubs,
        int32_t *rs, float *ps, float *super_trans, float *init_state_distn,
        vector<float*>& sub_transs, vector<float*>& sub_inits, vector<float*>& aBls,
        float *betan)
{
    float temp[bigN];
    NPVectorArray etemp(temp,bigN);

    NPMatrix esuper_trans(super_trans,N,N);

    int blocksizes[N];
    int blockstarts[N];

    vector<NPMatrix> esub_transs;
    vector<NPVector> esub_inits;
    vector<NPArray> eaBls;
    int totsize = 0;
    for (int i=0; i<N; i++) {
        blockstarts[i] = totsize;
        blocksizes[i] = Nsubs[i]*rs[i];
        totsize += blocksizes[i];

        esub_transs.push_back(NPMatrix(sub_transs[i],Nsubs[i],Nsubs[i]));
        esub_inits.push_back(NPVector(sub_inits[i],Nsubs[i]));
        eaBls.push_back(NPArray(aBls[i],T,Nsubs[i]));
    }
    NPArray ebetan(betan,T,bigN);

    ebetan.row(T-1).setOnes();
    float logtot = 0., cmax;
    for (int t=T-2; t>=0; t--) {
        cmax = -1.*numeric_limits<float>::infinity();
        for (int i=0; i<N; i++) {
            cmax = max(cmax, eaBls[i].row(t+1).maxCoeff());
        }

        for (int i=0; i<N; i++) {
            for (int k=0; k<rs[i]; k++) {
                NPSubArray(temp+blockstarts[i] + k*Nsubs[i],1,Nsubs[i])
                    *= (eaBls[i].row(t) - cmax).exp();
            }
        }
        etemp *= ebetan.row(t+1);

        float tot = matrix_vector_mult(N,Nsubs,rs,ps,esuper_trans,
                esub_transs,esub_inits,blocksizes,blockstarts,
                temp,betan + bigN*t);
        ebetan.row(t) /= tot;

        logtot += cmax + log(tot);
    }

    // NOTE: all this stuff is necessary to compute the log likelihood including
    // the initial state distribution

    float in_potential[bigN];
    memset(in_potential,0,sizeof(in_potential));
    cmax = -1.*numeric_limits<float>::infinity();
    for (int i=0; i<N; i++) {
        cmax = max(cmax, eaBls[i].row(0).maxCoeff());
    }
    // TODO TODO replace this with init state distn being passed in
    for (int i=0; i<N; i++) {
        NPSubVectorArray(in_potential + blockstarts[i],Nsubs[i]) =
            init_state_distn[i] * esub_inits[i].array() * (eaBls[i].row(0) - cmax).exp().transpose();
    }
    logtot += cmax + log((NPVectorArray(in_potential,bigN) * ebetan.row(0).transpose()).sum());

    return logtot;
}

float subhmm::messages_forwards_normalized(
        int T, int bigN, int N, int32_t *Nsubs,
        int32_t *rs, float *ps, float *super_trans, float *init_state_distn,
        vector<float*>& sub_transs, vector<float*>& sub_inits, vector<float*>& aBls,
        float *alphan)
{
    int blocksizes[N];
    int blockstarts[N];

    float super_trans_T[N*N] __attribute__ ((aligned(16)));
    NPMatrix esuper_trans_T(super_trans_T,N,N);
    esuper_trans_T = NPMatrix(super_trans,N,N);
    esuper_trans_T.transposeInPlace();

    vector<NPMatrix> esub_transs;
    vector<NPVector> esub_inits;
    vector<NPArray> eaBls;
    int totsize = 0;
    for (int i=0; i<N; i++) {
        blockstarts[i] = totsize;
        blocksizes[i] = Nsubs[i]*rs[i];
        totsize += blocksizes[i];

        esub_transs.push_back(NPMatrix(sub_transs[i],Nsubs[i],Nsubs[i]));
        esub_inits.push_back(NPVector(sub_inits[i],Nsubs[i]));
        eaBls.push_back(NPArray(aBls[i],T,Nsubs[i]));
    }
    NPArray ealphan(alphan,T,bigN);

    float in_potential[bigN] __attribute__ ((aligned(16)));
    NPVectorArray ein_potential(in_potential,bigN);
    ein_potential = NPVectorArray(init_state_distn,bigN);
    float logtot = 0., cmax;
    for (int t=0; t<T; t++) {
        cmax = -1.*numeric_limits<float>::infinity();
        for (int i=0; i<N; i++) {
            cmax = max(cmax, eaBls[i].row(t).maxCoeff());
        }

        for (int i=0; i<N; i++) {
            for (int k=0; k<rs[i]; k++) {
                NPSubArray(in_potential+blockstarts[i] + k*Nsubs[i],1,Nsubs[i])
                    *= (eaBls[i].row(t) - cmax).exp();
            }
        }

        float tot = ein_potential.sum();
        ealphan.row(t) = ein_potential / tot;
        logtot += log(tot) + cmax;

        vector_matrix_mult(N,Nsubs,rs,ps,esuper_trans_T,
                esub_transs,esub_inits,blocksizes,blockstarts,
                alphan + bigN*t,in_potential);
    }
    return logtot;
}

void subhmm::sample_backwards_normalized(
        int T, int bigN,
        float *alphan, int32_t *indptr, int32_t *indices, float *bigA_data,
        int32_t *stateseq)
{
    int sz = 0;
    for (int j=0; j<bigN; j++) {
        sz = max(sz,indptr[j+1] - indptr[j]);
    }
    float temp[sz];

    stateseq[T-1] = util::sample_discrete(bigN,alphan+bigN*(T-1));
    for (int t=T-2; t>=0; t--) {
        int start = indptr[stateseq[t+1]];
        int end = indptr[stateseq[t+1]+1];
        for (int i=start; i<end; i++) {
            temp[i-start] = bigA_data[i] * alphan[t*bigN + indices[i]];
        }
        stateseq[t] = indices[start+util::sample_discrete(end-start,temp)];
    }
}

void subhmm::generate_states(
        int T, int bigN, float *pi_0,
        int32_t *indptr, int32_t *indices, float *bigA_data,
        int32_t *stateseq)
{
    stateseq[0] = util::sample_discrete(bigN,pi_0);
    for (int t=1; t<T; t++) {
        int start = indptr[stateseq[t-1]];
        int end = indptr[stateseq[t-1]+1];
        stateseq[t] = indices[start+util::sample_discrete(end-start,bigA_data + start)];
    }
}

void subhmm::steady_state(
        int N, int32_t *Nsubs, int32_t *rs, float *ps,
        float *super_trans, vector<float*>& sub_transs, vector<float*>& sub_inits,
        float *v, int niter)
{
    int blocksizes[N];
    int blockstarts[N];

    float super_trans_T[N*N] __attribute__ ((aligned(16)));
    NPMatrix esuper_trans_T(super_trans_T,N,N);
    esuper_trans_T = NPMatrix(super_trans,N,N);
    esuper_trans_T.transposeInPlace();

    vector<NPMatrix> esub_transs;
    vector<NPVector> esub_inits;
    int totsize = 0;
    for (int i=0; i<N; i++) {
        blockstarts[i] = totsize;
        blocksizes[i] = Nsubs[i]*rs[i];
        totsize += blocksizes[i];

        esub_transs.push_back(NPMatrix(sub_transs[i],Nsubs[i],Nsubs[i]));
        esub_inits.push_back(NPVector(sub_inits[i],Nsubs[i]));
    }

    NPVector ev(v,totsize);
    float temp[totsize] __attribute__ ((aligned(16)));

    for (int i=0; i<niter/2; i++) {
        vector_matrix_mult(N,Nsubs,rs,ps,esuper_trans_T,esub_transs,esub_inits,
                blocksizes,blockstarts,
                v,temp);
        vector_matrix_mult(N,Nsubs,rs,ps,esuper_trans_T,esub_transs,esub_inits,
                blocksizes,blockstarts,
                temp,v);
        ev /= ev.sum();
    }
}

// changepoint stuff

inline
float subhmm::vector_matrix_mult_inside_segment(
        int N, int32_t *Nsubs, int32_t *rs, float *ps,
        vector<NPMatrix> &esub_transs,
        int *blocksizes, int *blockstarts,
        float *v, float *out)
{
    // these two lines just force eincomings to be on the stack
    float incomings[N] __attribute__ ((aligned(16)));
    NPVector eincomings(incomings,N);

    float sum_of_result = 0.;

    // NOTE: the next two loops are each parallelizeable over N

    for (int i=0; i<N; i++) {
        incomings[i] = ps[i] * NPSubVector(
                v + blockstarts[i] + blocksizes[i] - Nsubs[i],Nsubs[i]).sum();
    }

    for (int i=0; i < N; i++) {
        int blockstart = blockstarts[i];
        int32_t Nsub = Nsubs[i];
        int32_t r = rs[i];
        float p = ps[i];
        float temparr[r*Nsub] __attribute__((aligned(16)));
        NPMatrix &subtrans = esub_transs[i];

        NPSubMatrix rv(v + blockstart,r,Nsub);
        NPSubMatrix rout(out + blockstart,r,Nsub);

        // within-block
        NPMatrix temp(temparr,r,Nsub);
        temp.noalias() = rv * subtrans;
        rout.block(0,0,r-1,Nsub) = (temp.block(0,0,r-1,Nsub).array() * (1-p)).matrix();
        rout.block(r-1,0,1,Nsub) = temp.block(r-1,0,1,Nsub).matrix();
        rout.block(1,0,r-1,Nsub) += p * temp.block(0,0,r-1,Nsub);

        // track sum
        sum_of_result += rout.sum();
    }

    return sum_of_result;
}

float subhmm::messages_forwards_normalized_changepoints(
        int T, int bigN, int N, int32_t *Nsubs,
        int32_t *rs, float *ps, float *super_trans, float *init_state_distn,
        vector<float*>& sub_transs, vector<float*>& sub_inits, vector<float*>& aBls,
        int32_t *segmentstarts, int32_t *segmentlens, int Tblock,
        float *alphan)
{
    // standard setup
    int blocksizes[N];
    int blockstarts[N];

    float super_trans_T[N*N] __attribute__ ((aligned(16)));
    NPMatrix esuper_trans_T(super_trans_T,N,N);
    esuper_trans_T = NPMatrix(super_trans,N,N);
    esuper_trans_T.transposeInPlace();

    vector<NPMatrix> esub_transs;
    vector<NPVector> esub_inits;
    vector<NPArray> eaBls;
    int totsize = 0;
    for (int i=0; i<N; i++) {
        blockstarts[i] = totsize;
        blocksizes[i] = Nsubs[i]*rs[i];
        totsize += blocksizes[i];

        esub_transs.push_back(NPMatrix(sub_transs[i],Nsubs[i],Nsubs[i]));
        esub_inits.push_back(NPVector(sub_inits[i],Nsubs[i]));
        eaBls.push_back(NPArray(aBls[i],T,Nsubs[i]));
    }
    NPArray ealphan(alphan,Tblock,bigN);

    float temp1[bigN] __attribute__ ((aligned(16)));
    float temp2[bigN] __attribute__ ((aligned(16)));
    NPVectorArray(temp1,bigN) = NPVectorArray(init_state_distn,bigN);
    float logtot = 0., cmax, *in = temp1, *out = temp2;
    for (int tblock=0; tblock<Tblock; tblock++) {
        int t = segmentstarts[tblock];

        {
            cmax = -1.*numeric_limits<float>::infinity();
            for (int i=0; i<N; i++) {
                cmax = max(cmax, eaBls[i].row(t).maxCoeff());
            }

            for (int i=0; i<N; i++) {
                for (int k=0; k<rs[i]; k++) {
                    NPSubArray(in+blockstarts[i] + k*Nsubs[i],1,Nsubs[i])
                        *= (eaBls[i].row(t) - cmax).exp();
                }
            }
            NPVectorArray ein(in,bigN);
            float tot = ein.sum();
            ein /= tot;
            logtot += log(tot) + cmax;

            vector_matrix_mult(N,Nsubs,rs,ps,esuper_trans_T,
                    esub_transs,esub_inits,blocksizes,blockstarts,
                    in,out);

            swap(in,out);
        }

        for (t += 1; t<segmentstarts[tblock]+segmentlens[tblock]; t++) {
            cmax = -1.*numeric_limits<float>::infinity();
            for (int i=0; i<N; i++) {
                cmax = max(cmax, eaBls[i].row(t).maxCoeff());
            }

            for (int i=0; i<N; i++) {
                for (int k=0; k<rs[i]; k++) {
                    NPSubArray(in+blockstarts[i] + k*Nsubs[i],1,Nsubs[i])
                        *= (eaBls[i].row(t) - cmax).exp();
                }
            }
            NPVectorArray ein(in,bigN);
            float tot = ein.sum();
            ein /= tot;
            logtot += log(tot) + cmax;

            vector_matrix_mult_inside_segment(N,Nsubs,rs,ps,esub_transs,
                    blocksizes,blockstarts,
                    in,out);

            swap(in,out);
        }

        ealphan.row(tblock) = NPVectorArray(out,bigN);
    }
    return logtot;
}

// TESTING CODE

void subhmm::test_matrix_vector_mult(
        int N, int32_t *Nsubs, int32_t *rs, float *ps,
        float *super_trans, vector<float*>& sub_transs, vector<float*>& sub_inits,
        float *v, float *out)
{
    // prep work!
    int blocksizes[N];
    int blockstarts[N];
    NPMatrix esuper_trans(super_trans,N,N);

    vector<NPMatrix> esub_transs;
    vector<NPVector> esub_inits;
    int totsize = 0;
    for (int i=0; i<N; i++) {
        blockstarts[i] = totsize;
        blocksizes[i] = Nsubs[i]*rs[i];
        totsize += blocksizes[i];

        esub_transs.push_back(NPMatrix(sub_transs[i],Nsubs[i],Nsubs[i]));
        esub_inits.push_back(NPVector(sub_inits[i],Nsubs[i]));
    }

    matrix_vector_mult(N, Nsubs, rs, ps, esuper_trans, esub_transs, esub_inits,
            blocksizes, blockstarts,
            v, out);
}

void subhmm::test_vector_matrix_mult(
        int N, int32_t *Nsubs, int32_t *rs, float *ps,
        float *super_trans, vector<float*>& sub_transs, vector<float*>& sub_inits,
        float *v, float *out)
{
    // prep work!
    int blocksizes[N];
    int blockstarts[N];

    float super_trans_T[N*N] __attribute__ ((aligned(16)));
    NPMatrix esuper_trans_T(super_trans_T,N,N);
    esuper_trans_T = NPMatrix(super_trans,N,N);
    esuper_trans_T.transposeInPlace();

    vector<NPMatrix> esub_transs;
    vector<NPVector> esub_inits;
    int totsize = 0;
    for (int i=0; i<N; i++) {
        blockstarts[i] = totsize;
        blocksizes[i] = Nsubs[i]*rs[i];
        totsize += blocksizes[i];

        esub_transs.push_back(NPMatrix(sub_transs[i],Nsubs[i],Nsubs[i]));
        esub_inits.push_back(NPVector(sub_inits[i],Nsubs[i]));
    }

    vector_matrix_mult(N, Nsubs, rs, ps, esuper_trans_T, esub_transs, esub_inits,
            blocksizes, blockstarts,
            v, out);
}

