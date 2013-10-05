#include "subhmm_messages.h"

using namespace Eigen;
using namespace std;
using namespace subhmm;

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
        eincomings(i) = esub_inits[i].transpose() * NPSubVector(v + blockstarts[i],Nsubs[i]);
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
        temp = rv * subtrans.transpose();
        rout = (temp.array() * (1-p)).matrix();
        rout.block(0,0,r-1,Nsub) += (p * temp.block(1,0,r-1,Nsub).array()).matrix();

        // across-block
        NPSubVectorArray(out+blockstart+blocksize-Nsub,Nsub)
            += p * esuper_trans.row(i).dot(eincomings);

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

    int blocksizes[N];
    int blockstarts[N];
    NPMatrix esuper_trans(super_trans,N,N);

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

