#pragma once
#include <algorithm>
#include <numeric>
#include "megdnn/basic_types.h"
#include "megdnn/dtype.h"
#include "src/common/utils.h"

using namespace megdnn;

/* anonymous namespace */
namespace {
using Mode = Reduce::Mode;

/* Reduce Trait */
template <Mode mode, typename ctype>
struct Trait;

template <typename ctype>
struct Trait<Mode::SUM, ctype> {
    static const ctype INIT;

    static ctype apply(ctype x, ctype y) { return x + y; }
    static ctype visit(ctype x) { return x; }
    static ctype write(ctype x, size_t) { return x; }
};
template <typename ctype>
const ctype Trait<Mode::SUM, ctype>::INIT = ctype(0);

template <typename ctype>
struct Trait<Mode::MEAN, ctype> {
    static const ctype INIT;

    static ctype apply(ctype x, ctype y) { return x + y; }
    static ctype visit(ctype x) { return x; }
    static ctype write(ctype x, size_t B) { return x / (ctype)B; }
};
template <typename ctype>
const ctype Trait<Mode::MEAN, ctype>::INIT = ctype(0);

template <typename ctype>
struct Trait<Mode::SUM_SQR, ctype> {
    static const ctype INIT;

    static ctype apply(ctype x, ctype y) { return x + y; }
    static ctype visit(ctype x) { return x * x; }
    static ctype write(ctype x, size_t) { return x; }
};
template <typename ctype>
const ctype Trait<Mode::SUM_SQR, ctype>::INIT = ctype(0);

template <typename ctype>
struct Trait<Mode::PRODUCT, ctype> {
    static const ctype INIT;

    static ctype apply(ctype x, ctype y) { return x * y; }
    static ctype visit(ctype x) { return x; }
    static ctype write(ctype x, size_t) { return x; }
};
template <typename ctype>
const ctype Trait<Mode::PRODUCT, ctype>::INIT = ctype(1);

template <typename ctype>
struct Trait<Mode::MIN, ctype> {
    static ctype apply(ctype x, ctype y) { return x < y ? x : y; }
    static ctype visit(ctype x) { return x; }
    static ctype write(ctype x, size_t) { return x; }
};

template <>
struct Trait<Mode::MIN, dt_float32> {
    using ctype = dt_float32;

    static ctype apply(ctype x, ctype y) { return (std::isnan(x) || x < y) ? x : y; }
    static ctype visit(ctype x) { return x; }
    static ctype write(ctype x, size_t) { return x; }
};

template <typename ctype>
struct Trait<Mode::MAX, ctype> {
    static ctype apply(ctype x, ctype y) { return x > y ? x : y; }
    static ctype visit(ctype x) { return x; }
    static ctype write(ctype x, size_t) { return x; }
};

template <>
struct Trait<Mode::MAX, dt_float32> {
    using ctype = dt_float32;

    static ctype apply(ctype x, ctype y) { return (std::isnan(x) || x > y) ? x : y; }
    static ctype visit(ctype x) { return x; }
    static ctype write(ctype x, size_t) { return x; }
};

/* NormOp */
template <typename ctype>
struct NormOp;

template <>
struct NormOp<dt_float32> {
    typedef dt_float32 ctype;
    static const ctype INIT;

    static ctype apply(ctype x, ctype y) { return x + y; }
    static ctype visit(ctype x, dt_float32 p) { return powf(fabs(x), p); }
    static ctype write(ctype x, size_t, dt_float32 p) { return powf(x, 1.f / p); }
};

#if !MEGDNN_DISABLE_FLOAT16
template <>
struct NormOp<dt_float16> {
    typedef dt_float16 ctype;
    static const ctype INIT;

    static ctype apply(ctype x, ctype y) { return x + y; }
    static ctype visit(ctype x, dt_float32 p) {
        return half_float::pow(half_float::abs(x), half_float::half(p));
    }
    static ctype write(ctype x, size_t, dt_float32 p) {
        return half_float::pow(x, half_float::half(1.f / p));
    }
};
#endif

template <typename ctype>
struct NormZeroOp;

template <>
struct NormZeroOp<dt_float32> {
    typedef dt_float32 ctype;
    static const ctype INIT;

    static ctype apply(ctype x, ctype y) { return x + y; }
    static ctype visit(ctype x) { return x - 0.f < 0.00001f ? 0.f : 1.f; }
    static ctype write(ctype x, size_t) { return x; }
};

#if !MEGDNN_DISABLE_FLOAT16
template <>
struct NormZeroOp<dt_float16> {
    typedef dt_float16 ctype;
    static const ctype INIT;

    static ctype apply(ctype x, ctype y) { return x + y; }
    static ctype visit(ctype x) {
        return x - half_float::half(0.f) < half_float::half(0.00001f)
                     ? half_float::half(0.f)
                     : half_float::half(1.f);
    }
    static ctype write(ctype x, size_t) { return x; }
};
#endif
}  // namespace
