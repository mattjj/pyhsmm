#include <Eigen/Core>
#include <vector>
#include <stdint.h>
#include <limits>

using namespace Eigen;
using namespace std;

typedef Map<Matrix<double,Dynamic,Dynamic,RowMajor>,Aligned> NPMatrix;
typedef Map<Matrix<double,Dynamic,Dynamic,RowMajor> > NPSubMatrix;
typedef Map<Array<double,Dynamic,Dynamic,RowMajor>,Aligned> NPArray;
typedef Map<Array<double,Dynamic,Dynamic,RowMajor> > NPSubArray;

typedef Map<Matrix<double,Dynamic,1>,Aligned> NPVector;
typedef Map<Matrix<double,Dynamic,1> > NPSubVector;
typedef Map<Array<double,Dynamic,1>,Aligned> NPVectorArray;
typedef Map<Array<double,Dynamic,1> > NPSubVectorArray;

void fast_mult(
        int N, int32_t *Nsubs, int32_t *rs, double *ps,
        double *super_trans, vector<double*>& sub_transs, vector<double*>& sub_inits,
        double *v, double *out);

inline
double just_fast_mult(
        int N, int32_t *Nsubs, int32_t *rs, double *ps,
        NPMatrix &esuper_trans,
        vector<NPMatrix> &esub_transs, vector<NPVector> &esub_inits,
        int *blocksizes, int *blockstarts,
        double *v, double *out);

double messages_backwards_normalized(
        int T, int bigN, int N, int32_t *Nsubs,
        int32_t *rs, double *ps, double *super_trans,
        vector<double*>& sub_transs, vector<double*>& sub_inits,
        vector<double*>& aBls,
        double *betan);

// TODO instead of taking a ton of arguments, there should be a struct that the
// cython code forms on the stack and then passes to these funcs (which take it
// as a reference)

