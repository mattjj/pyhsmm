#include <Eigen/Core>
#include <vector>
#include <stdint.h>
#include <limits>

using namespace Eigen;
using namespace std;

typedef Map<Matrix<float,Dynamic,Dynamic,RowMajor>,Aligned> NPMatrix;
typedef Map<Matrix<float,Dynamic,Dynamic,RowMajor> > NPSubMatrix;
typedef Map<Array<float,Dynamic,Dynamic,RowMajor>,Aligned> NPArray;
typedef Map<Array<float,Dynamic,Dynamic,RowMajor> > NPSubArray;

typedef Map<Matrix<float,Dynamic,1>,Aligned> NPVector;
typedef Map<Matrix<float,Dynamic,1> > NPSubVector;
typedef Map<Array<float,Dynamic,1>,Aligned> NPVectorArray;
typedef Map<Array<float,Dynamic,1> > NPSubVectorArray;

void fast_mult(
        int N, int32_t *Nsubs, int32_t *rs, float *ps,
        float *super_trans, vector<float*>& sub_transs, vector<float*>& sub_inits,
        float *v, float *out);

inline
float just_fast_mult(
        int N, int32_t *Nsubs, int32_t *rs, float *ps,
        NPMatrix &esuper_trans,
        vector<NPMatrix> &esub_transs, vector<NPVector> &esub_inits,
        int *blocksizes, int *blockstarts,
        float *v, float *out);

float messages_backwards_normalized(
        int T, int bigN, int N, int32_t *Nsubs,
        int32_t *rs, float *ps, float *super_trans,
        vector<float*>& sub_transs, vector<float*>& sub_inits,
        vector<float*>& aBls,
        float *betan);

// TODO instead of taking a ton of arguments, there should be a struct that the
// cython code forms on the stack and then passes to these funcs (which take it
// as a reference)

