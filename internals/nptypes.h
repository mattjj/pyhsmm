#ifndef TYPES_H
#define TYPES_H

#ifndef EIGEN_H
#include <Eigen/Core>
#endif

// NOTE: alias declarations require C++11
// a C++03 alternative might look like this:
// template <typename T>
// struct NPMatrix {
//     typedef Map<Matrix<T,Dynamic,Dynamic,RowMajor>,Aligned> type;
// }
// NPMatrix<float>::type foo;

// NOTE: could add concepts (template constraints) via Stroustrup's mechanism
// or Boost's BCCL. I don't have those, but these are for floating types!


namespace nptypes {
    using namespace Eigen;

    template <typename T>
    using NPSubMatrix = Map<Matrix<T,Dynamic,Dynamic,RowMajor> >;

    template <typename T>
    using NPSubArray = Map<Array<T,Dynamic,Dynamic,RowMajor> >;

    template <typename T>
    using NPSubVector = Map<Matrix<T,Dynamic,1> >;

    template <typename T>
    using NPSubRowVector = Map<Matrix<T,1,Dynamic> >;

    template <typename T>
    using NPSubVectorArray = Map<Array<T,Dynamic,1> >;

    template <typename T>
    using NPSubRowVectorArray = Map<Array<T,1,Dynamic> >;

#ifdef NPTYPES_NOT_ALIGNED
    template <typename T>
    using NPMatrix = NPSubMatrix<T>;

    template <typename T>
    using NPArray = NPSubArray<T>;

    template <typename T>
    using NPVector = NPSubVector<T>;

    template <typename T>
    using NPRowVector = NPSubRowVector<T>;

    template <typename T>
    using NPVectorArray = NPSubVectorArray<T>;

    template <typename T>
    using NPRowVectorArray = NPSubRowVectorArray<T>;
#else
    template <typename T>
    using NPMatrix = Map<Matrix<T,Dynamic,Dynamic,RowMajor>,Aligned>;

    template <typename T>
    using NPArray = Map<Array<T,Dynamic,Dynamic,RowMajor>,Aligned>;

    template <typename T>
    using NPVector = Map<Matrix<T,Dynamic,1>,Aligned>;

    template <typename T>
    using NPRowVector = Map<Matrix<T,1,Dynamic>,Aligned>;

    template <typename T>
    using NPVectorArray = Map<Array<T,Dynamic,1>,Aligned>;

    template <typename T>
    using NPRowVectorArray = Map<Array<T,1,Dynamic>,Aligned>;
#endif
}

#endif
