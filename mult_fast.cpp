#include "mult_fast.h"
using namespace std;
using namespace Eigen;

// NOTE: dynamic stack arrays aren't in C++ but clang and g++ support them
// NOTE: on my test machine numpy heap arrays were always aligned, but I doubt
// that's guaranteed. Still, this code assumes alignment!

// TODO instead of taking a ton of arguments, there should be a struct that the
// cython code forms on the stack and then passes to these funcs (which take it
// as a reference)
// TODO provide sane max sizes so compiler knows things

void fast_mult(
        int N, int32_t *Nsubs, int32_t *rs, double *ps,
        double *super_trans, vector<double*>& sub_transs, vector<double*>& sub_inits,
        double *v, double *out)
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
void just_fast_mult(
        int N, int32_t *Nsubs, int32_t *rs, double *ps,
        NPMatrix &esuper_trans,
        vector<NPMatrix> &esub_transs, vector<NPVector> &esub_inits,
        int *blocksizes, int *blockstarts,
        double *v, double *out)
{
    // these two lines just force eincomings to be on the stack
    double incomings[N]; // NOTE: assuming stack pointer alignment for this guy too
    Map<VectorXd,Aligned> eincomings(incomings,N);

    // NOTE: the next two loops are each parallelizeable over N

    for (int i=0; i<N; i++) {
        eincomings(i) = esub_inits[i].transpose() * NPVector(v + blockstarts[i],Nsubs[i]);
    }

    for (int i=0; i < N; i++) {
        int blockstart = blockstarts[i];
        int blocksize = blocksizes[i];
        int32_t Nsub = Nsubs[i];
        int32_t r = rs[i];
        double p = ps[i];
        NPMatrix &subtrans = esub_transs[i];
        NPVector &pi = esub_inits[i];

        NPMatrix rv(v + blockstart,r,Nsub);
        NPMatrix rout(out + blockstart,r,Nsub);

        // within-block
        MatrixXd temp(r,Nsub); // NOTE: checked asm, no malloc here with g++ -O3
        temp = rv * subtrans.transpose();
        rout = (temp.array() * (1-p)).matrix();
        rout.block(0,0,r-1,Nsub) += (p * temp.block(1,0,r-1,Nsub).array()).matrix();

        // across-block
        NPVectorArray(out+blockstart+blocksize-Nsub,Nsub)
            += p * esuper_trans.row(i).dot(eincomings);
    }
}

// TODO just_fast_mult should max and exp as it goes...

double messages_backwards_normalized(
        int T, int bigN, int N, int32_t *Nsubs,
        int32_t *rs, double *ps, double *super_trans,
        std::vector<double*>& sub_transs, std::vector<double*>& sub_inits,
        std::vector<double*>& aBls,
        double *betan)
{
    // prep work! identical to above
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

    Map<VectorXd,Aligned> eincomings(incomings,N); // TODO double-check no malloc, o/w outside

    for (int t=T-2; t>=0; t--) {
        // TODO
        // within each block, aBl is essentially repmatted. so i can reshape and
        // then dot mult. i probably want a temp array of size bigN.
        // then loop over i=1 to N reshaping chunks and .* in reshaped chunks of
        // beta. then 
    }
}
