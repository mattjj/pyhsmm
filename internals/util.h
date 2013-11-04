#ifndef UTIL_H
#define UTIL_H

#ifndef MY_CSTDLIB_H
#define MY_CSTDLIB_H
#include <cstdlib>
#endif

using namespace std;

namespace util {
    template <typename T>
    int sample_discrete(int N, T *distn)
    {
        T tot = 0;
        for (int i=0; i<N; i++) { tot += distn[i]; }
        tot *= ((T) rand()) / ((T) (RAND_MAX));

        int sample_idx;
        for (sample_idx=0; sample_idx < N-1 && (tot -= distn[sample_idx]) > 0; sample_idx++) ;

        return sample_idx;
    }
}

#endif
