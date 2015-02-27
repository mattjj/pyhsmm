#ifndef UTIL_H
#define UTIL_H

#define likely(x) __builtin_expect(!!(x),true)
#define unlikely(x) __builtin_expect(!!(x),false)

namespace util {
    using namespace std;

    template <typename T>
    int sample_discrete(int N, T *distn, T rand_uniform)
    {
        T tot = 0;
        for (int i=0; i<N; i++) { tot += distn[i]; }
        tot *= rand_uniform;

        int sample_idx;
        for (sample_idx=0; sample_idx < N-1 && (tot -= distn[sample_idx]) > 0; sample_idx++) ;

        return sample_idx;
    }

    // got these next two from Eigen
    template<typename T> bool is_not_nan(const T& x) { return x==x; }
    template<typename T> bool is_finite(const T& x) { return is_not_nan(x - x); }
}

#endif
