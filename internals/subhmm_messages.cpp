#include "subhmm_messages.h"

using namespace Eigen;
using namespace std;
using namespace subhmm;

inline
float subhmm::just_fast_mult(
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
        NPMatrix &subtrans = esub_transs[i];

        NPSubMatrix rv(v + blockstart,r,Nsub);
        NPSubMatrix rout(out + blockstart,r,Nsub);

        // within-block
        MatrixXf temp(r,Nsub); // NOTE: checked asm, no malloc here with g++ -O3, note float
        temp = rv * subtrans.transpose();
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
float subhmm::just_fast_left_mult(
        int N, int32_t *Nsubs, int32_t *rs, float *ps,
        NPMatrix &esuper_trans_T,
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
        incomings[i] = ps[i] * NPSubVector(
                v + blockstarts[i] + blocksizes[i] - Nsubs[i],Nsubs[i]).sum();
    }

    for (int i=0; i < N; i++) {
        int blockstart = blockstarts[i];
        int blocksize = blocksizes[i];
        int32_t Nsub = Nsubs[i];
        int32_t r = rs[i];
        float p = ps[i];
        NPMatrix &subtrans = esub_transs[i];
        NPVector &pi = esub_inits[i];

        NPSubMatrix rv(v + blockstart,r,Nsub);
        NPSubMatrix rout(out + blockstart,r,Nsub);

        // within-block
        MatrixXf temp(r,Nsub); // NOTE: checked asm, no malloc here with g++ -O3, note float
        temp = rv * subtrans;
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
        int32_t *rs, float *ps, float *super_trans,
        std::vector<float*>& sub_transs, std::vector<float*>& sub_inits,
        std::vector<float*>& aBls,
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
    float logtot = 0.;
    for (int t=T-2; t>=0; t--) {
        float cmax = -1.*numeric_limits<float>::infinity();
        for (int i=0; i<N; i++) {
            cmax = max(cmax, eaBls[i].row(t+1).maxCoeff());
        }

        for (int i=0; i<N; i++) {
            NPSubArray(temp + blockstarts[i],rs[i],Nsubs[i]).rowwise()
                = (eaBls[i].row(t+1) - cmax).exp();
        }
        etemp *= ebetan.row(t+1);

        float tot = just_fast_mult(N,Nsubs,rs,ps,esuper_trans,
                esub_transs,esub_inits,blocksizes,blockstarts,
                temp,betan + bigN*t);
        ebetan.row(t) /= tot;

        logtot += cmax + log(tot);
    }
    return logtot;
}

float subhmm::messages_forwards_normalized(
        int T, int bigN, int N, int32_t *Nsubs,
        int32_t *rs, float *ps, float *super_trans, float *init_state_distn,
        std::vector<float*>& sub_transs, std::vector<float*>& sub_inits,
        std::vector<float*>& aBls,
        float *alphan)
{
    int blocksizes[N];
    int blockstarts[N];
    NPMatrix esuper_trans_T(super_trans,N,N);
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

    float in_potential[bigN];
    NPVectorArray ein_potential(in_potential,bigN);
    ein_potential.setZero();
    for (int i=0; i<N; i++) {
        ein_potential.segment(blockstarts[i],Nsubs[i]) =
            init_state_distn[i] * esub_inits[i].array();
    }
    float logtot = 0., cmax;
    for (int t=0; t<T; t++) {
        cmax = -1.*numeric_limits<float>::infinity();
        for (int i=0; i<N; i++) {
            cmax = max(cmax, eaBls[i].row(t).maxCoeff());
        }

        for (int i=0; i<N; i++) {
            NPSubArray(in_potential + blockstarts[i],rs[i],Nsubs[i]).rowwise()
                *= (eaBls[i].row(t) - cmax).exp();
        }
        float tot = ein_potential.sum();
        ealphan.row(t) = ein_potential / tot;
        logtot += log(tot) + cmax;

        just_fast_left_mult(N,Nsubs,rs,ps,esuper_trans_T,
                esub_transs,esub_inits,blocksizes,blockstarts,
                alphan + bigN*t,in_potential);
    }
    return logtot;
}

// these next ones are for testing

void subhmm::fast_mult(
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

    just_fast_mult(N, Nsubs, rs, ps, esuper_trans, esub_transs, esub_inits,
            blocksizes, blockstarts,
            v, out);
}

void subhmm::fast_left_mult(
        int N, int32_t *Nsubs, int32_t *rs, float *ps,
        float *super_trans, vector<float*>& sub_transs, vector<float*>& sub_inits,
        float *v, float *out)
{
    // prep work!
    int blocksizes[N];
    int blockstarts[N];

    float super_trans_T[N*N];
    NPMatrix esuper_trans_T(super_trans_T,N,N);
    esuper_trans_T = NPMatrix(super_trans,N,N).transpose();

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

    just_fast_left_mult(N, Nsubs, rs, ps, esuper_trans_T, esub_transs, esub_inits,
            blocksizes, blockstarts,
            v, out);
}


