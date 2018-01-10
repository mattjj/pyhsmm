#ifndef UTIL_H
#define UTIL_H

#ifdef _WIN32 || _WIN64
#define NO_BUILTIN_EXPECT
#endif

#ifdef NO_BUILTIN_EXPECT
#if !defined(LIKELY)
#define likely(x) (!!(x),true)
#endif
#if !defined(UNLIKELY)
#define unlikely(x) (!!(x),false)
#endif
#else
#if !defined(LIKELY)
#define likely(x) __builtin_expect(!!(x),true)
#endif
#if !defined(UNLIKELY)
#define unlikely(x) __builtin_expect(!!(x),false)
#endif
#endif

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
