#include "util.h"
using namespace std;

int util::sample_discrete(int N, double *distn)
{
    double tot = 0;
    for (int i=0; i<N; i++) { tot += distn[i]; }
    tot *= ((double) rand()) / ((double) (RAND_MAX));

    int sample_idx;
    for (sample_idx=0; sample_idx < N && (tot -= distn[sample_idx]) > 0; sample_idx++) ;

    return sample_idx;
}
