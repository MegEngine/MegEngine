#include "./opr_impl.h"
#include "src/common/utils.h"
#include "src/naive/handle.h"

#include <cmath>

using namespace megdnn;
using namespace naive;

namespace {
template <typename ctype>
ctype uniform_int2float(uint64_t x);

template <>
dt_float32 uniform_int2float(uint64_t x) {
    union {
        uint32_t i;
        dt_float32 f;
    } u;
    u.i = (0x7F << 23) | (x >> 41);
    return 2 - u.f;
}

#if !MEGDNN_DISABLE_FLOAT16
template <>
dt_float16 uniform_int2float(uint64_t x) {
    union U {
        uint16_t i;
        dt_float16 f;
        U() : f(0) {}
    } u;
    u.i = (0xF << 10) | (x >> 54);
    return dt_float16(2.f) - u.f;
}
#endif

#if !MEGDNN_DISABLE_FLOAT16
template <>
dt_bfloat16 uniform_int2float(uint64_t x) {
    union U {
        uint16_t i;
        dt_bfloat16 f;
        U() : f(0) {}
    } u;
    u.i = (0x7F << 7) | (x >> 57);
    return dt_bfloat16(2.f) - u.f;
}
#endif

template <typename ctype>
void fill_uniform(Xoroshiro128plus* rng, ctype* dst, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        dst[i] = uniform_int2float<ctype>((*rng)());
    }
}

template <typename ctype>
void fill_gaussian(
        Xoroshiro128plus* rng, ctype* dst, size_t size, ctype mean, ctype stddev) {
    // gen gaussian by Box-Muller transform
    for (size_t i = 0; i + 2 <= size; i += 2) {
        ctype u1 = uniform_int2float<ctype>((*rng)()),
              u2 = uniform_int2float<ctype>((*rng)()),
              r = ctype(stddev * std::sqrt(-2 * std::log(u1))),
              theta = ctype(2 * M_PI * u2), z0 = ctype(r * std::cos(theta) + mean),
              z1 = ctype(r * std::sin(theta) + mean);
        dst[i] = z0;
        dst[i + 1] = z1;
    }
    if (size % 2) {
        ctype u1 = uniform_int2float<ctype>((*rng)()),
              u2 = uniform_int2float<ctype>((*rng)()),
              r = ctype(stddev * std::sqrt(-2 * std::log(u1))),
              theta = ctype(2 * M_PI * u2), z0 = ctype(r * std::cos(theta) + mean);
        dst[size - 1] = z0;
    }
}

template <typename T>
T normal_sample(Xoroshiro128plus* rng) {
    T v;
    fill_gaussian<T>(rng, &v, 1, T(0.f), T(1.f));
    return v;
}

template <typename T>
T uniform_sample(Xoroshiro128plus* rng) {
    return uniform_int2float<T>((*rng)());
}

template <typename T, typename U>
void fill_gamma(Xoroshiro128plus* rng, U* dst, size_t size, U* shape, U* scale) {
    for (size_t i = 0; i < size; ++i) {
        T a = static_cast<T>(shape[i]);
        T b = static_cast<T>(scale[i]);
        T scale = b;
        bool a_less_one = a < 1.f ? true : false;
        if (a <= 0) {
            dst[i] = U(0.0f);
            continue;
        };
        T d = a + (a_less_one ? 2.0f / 3.0f : -1.0f / 3.0f);
        T c = 1.0f / std::sqrt(9.0f * d);
        while (true) {
            T x, y;
            x = normal_sample<T>(rng);
            y = 1.0f + c * x;
            if (y <= 0)
                continue;
            T v = y * y * y;
            T u = uniform_sample<T>(rng);
            T xx = x * x;
            if ((u < 1.0f - 0.0331f * xx * xx) ||
                std::log(u) < 0.5f * xx + d * (1.0f - v + std::log(v))) {
                dst[i] = U(scale * d * v);
                if (a_less_one)
                    dst[i] *= U(std::pow(uniform_sample<T>(rng), T(1.f / a)));
                break;
            }
        }
    }
}

template <typename U>
void fill_multinomial_without_replacement(
        Xoroshiro128plus* rng, U* probs, dt_int32* dst, size_t num_groups,
        size_t num_samples, size_t len_probs) {
    std::vector<std::vector<U>> data(num_groups, std::vector<U>(len_probs, U(0)));
    for (size_t i = 0; i < num_groups; ++i) {
        for (size_t j = 0; j < len_probs; ++j) {
            data[i][j] = log(uniform_sample<U>(rng)) / probs[i * len_probs + j];
        }
    }

    std::vector<std::vector<dt_int32>> index(
            num_groups, std::vector<dt_int32>(len_probs, 0));
    for (size_t i = 0; i < num_groups; ++i) {
        for (size_t j = 0; j < len_probs; ++j) {
            index[i][j] = j;
        }
    }

    for (size_t i = 0; i < num_groups; ++i) {
        std::sort(
                index[i].begin(), index[i].end(), [i, &data](size_t idx1, size_t idx2) {
                    return data[i][idx1] > data[i][idx2];
                });
        std::copy(
                index[i].begin(), index[i].begin() + num_samples,
                dst + i * num_samples);
    }
}

template <typename U>
void fill_multinomial(
        Xoroshiro128plus* rng, U* probs, dt_int32* dst, size_t num_groups,
        size_t num_samples, size_t len_probs, bool replacement) {
    if (!replacement) {
        fill_multinomial_without_replacement(
                rng, probs, dst, num_groups, num_samples, len_probs);
        return;
    }
    for (size_t i = 0; i < num_groups; ++i) {
        for (size_t j = 0; j < num_samples; ++j) {
            U u = uniform_sample<U>(rng);
            U cumsum_res = U(0);
            for (size_t k = 0; k < len_probs; ++k) {
                cumsum_res += probs[i * len_probs + k];
                if (u <= cumsum_res) {
                    dst[i * num_samples + j] = k;
                    break;
                }
            }
        }
    }
}

template <typename T, typename U>
void fill_poisson(Xoroshiro128plus* rng, U* dst, U* lam, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        T lambda = static_cast<T>(lam[i]);
        T exp_neg_lambda = std::exp(-lambda);
        T log_lambda = std::log(lambda), sqrt_lambda = std::sqrt(lambda);
        T b = 0.931f + 2.53f * sqrt_lambda;
        T a = -0.059f + 0.02483f * b;
        T inv_alpha = 1.1239f + 1.1328f / (b - 3.4f);
        T vr = 0.9277f - 3.6224f / (b - 2.f);
        T u, v, u_shifted, k;
        if (lambda == 0) {
            dst[i] = U(0);
            continue;
        }
        if (lambda < 10) {
            T prod = 1, x = 0;
            u = 0;
            while (true) {
                u = uniform_sample<T>(rng);
                prod *= u;
                if (prod <= exp_neg_lambda) {
                    dst[i] = U(x);
                    break;
                }
                x += 1;
            }
            continue;
        }
        while (true) {
            u = uniform_sample<T>(rng) - T(0.5f);
            v = uniform_sample<T>(rng);
            u_shifted = T(0.5f) - std::abs(u);
            k = std::floor((T(2.f) * a / u_shifted + b) * u + lambda + T(0.43f));
            if (u_shifted >= 0.07 && v < vr) {
                dst[i] = U(k);
                break;
            }
            if (k < 0 || (u_shifted < T(0.013f) && v > u_shifted)) {
                continue;
            }
            if ((std::log(v) + std::log(inv_alpha) -
                 std::log(a / (u_shifted * u_shifted) + b)) <=
                (-lambda + k * log_lambda - std::lgamma(k + 1))) {
                dst[i] = U(k);
                break;
            }
        }
    }
}

template <typename T, typename U>
void fill_beta(Xoroshiro128plus* rng, U* dst, U* alpha, U* beta, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        T a = static_cast<T>(alpha[i]), b = static_cast<T>(beta[i]);
        if (a < 1.0f && b < 1.0f) {
            T u, v, x, y;
            while (true) {
                u = uniform_sample<T>(rng);
                v = uniform_sample<T>(rng);
                x = std::pow(u, 1.0f / a);
                y = std::pow(v, 1.0f / b);
                if (x + y < 1.0f) {
                    if (x + y > 0) {
                        dst[i] = static_cast<U>(x / (x + y));
                        break;
                    } else {
                        T logx = std::log(u) / a;
                        T logy = std::log(v) / b;
                        T log_max = std::max(logx, logy);
                        logx -= log_max;
                        logy -= log_max;
                        dst[i] = static_cast<U>(std::exp(
                                logx - std::log(std::exp(logx) + std::exp(logy))));
                        break;
                    }
                }
            }
        } else {
            T ga, gb, one = 1;
            fill_gamma<T, T>(rng, &ga, 1, &a, &one);
            fill_gamma<T, T>(rng, &gb, 1, &b, &one);
            dst[i] = static_cast<U>(ga / (ga + gb));
        }
    }
}

template <typename T>
void fill_permutation(Xoroshiro128plus* rng, T* dst, size_t size) {
    const int64_t mask = std::numeric_limits<int64_t>::max();
    for (size_t i = 0; i < size; ++i) {
        dst[i] = static_cast<T>(i);
    }
    for (int64_t i = size - 1; i > 0; --i) {
        int64_t r = static_cast<int64_t>((*rng)() & mask) % (i + 1);
        if (i != r) {
            T tmp = dst[i];
            dst[i] = dst[r];
            dst[r] = tmp;
        }
    }
}

template <typename T, typename U>
void fill_exponential(Xoroshiro128plus* rng, U* dst, U* rate, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        T r = static_cast<T>(rate[i]);
        T u = uniform_sample<T>(rng);
        dst[i] = static_cast<U>(-std::log(u) / r);
    }
}

template <typename T>
void shuffle_fwd(
        const T* __restrict sptr, T* __restrict dptr, const dt_int32* iptr,
        const size_t len, const size_t step) MEGDNN_NOEXCEPT {
    for (size_t i = 0; i < len; ++i) {
        for (size_t j = 0; j < step; ++j) {
            dptr[i * step + j] = sptr[iptr[i] * step + j];
        }
    }
}

template <typename T>
void shuffle_bwd(
        T* __restrict sptr, const T* __restrict dptr, const dt_int32* iptr,
        const size_t len, const size_t step) MEGDNN_NOEXCEPT {
    for (size_t i = 0; i < len; ++i) {
        for (size_t j = 0; j < step; ++j) {
            sptr[iptr[i] * step + j] = dptr[i * step + j];
        }
    }
}

}  // anonymous namespace

uint64_t Splitmix64::operator()() {
    uint64_t z = (m_s += UINT64_C(0x9E3779B97F4A7C15));
    z = (z ^ (z >> 30)) * UINT64_C(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94D049BB133111EB);
    return z ^ (z >> 31);
}

void Xoroshiro128plus::seed(uint64_t seed) {
    Splitmix64 r1{seed};
    m_s[0] = r1();
    m_s[1] = r1();
    m_init_seed = seed;
}

uint64_t Xoroshiro128plus::operator()() {
    const uint64_t s0 = m_s[0];
    uint64_t s1 = m_s[1];
    const uint64_t result = s0 + s1;

    s1 ^= s0;
    m_s[0] = rotl(s0, 55) ^ s1 ^ (s1 << 14);  // a, b
    m_s[1] = rotl(s1, 36);                    // c

    return result;
}

void UniformRNGImpl::exec(_megdnn_tensor_inout dst, _megdnn_workspace workspace) {
    check_exec(dst.layout, workspace.size);
    auto size = dst.layout.total_nr_elems();
    auto prng = &m_rng.ensure_seed(m_param.seed);
    switch (dst.layout.dtype.enumv()) {
#define cb(_dt)                                                                        \
    case DTypeTrait<_dt>::enumv: {                                                     \
        using ctype = DTypeTrait<_dt>::ctype;                                          \
        MEGDNN_DISPATCH_CPU_KERN_OPR({ fill_uniform(prng, dst.ptr<ctype>(), size); }); \
        return;                                                                        \
    }
        MEGDNN_FOREACH_COMPUTING_DTYPE_FLOAT(cb)
#undef cb
        default:
            megdnn_throw("bad dtype");
    }
}

void GaussianRNGImpl::exec(_megdnn_tensor_inout dst, _megdnn_workspace workspace) {
    check_exec(dst.layout, workspace.size);
    auto size = dst.layout.total_nr_elems();
    auto prng = &m_rng.ensure_seed(m_param.seed);
    switch (dst.layout.dtype.enumv()) {
#define cb(_dt)                                                                      \
    case DTypeTrait<_dt>::enumv: {                                                   \
        using ctype = DTypeTrait<_dt>::ctype;                                        \
        ctype mean(m_param.mean), std(m_param.std);                                  \
        MEGDNN_DISPATCH_CPU_KERN_OPR(                                                \
                { fill_gaussian<ctype>(prng, dst.ptr<ctype>(), size, mean, std); }); \
        return;                                                                      \
    }
        MEGDNN_FOREACH_COMPUTING_DTYPE_FLOAT(cb)
#undef cb
        default:
            megdnn_throw("bad dtype");
    }
}

void GammaRNGImpl::exec(
        _megdnn_tensor_in shape, _megdnn_tensor_in scale, _megdnn_tensor_out dst,
        _megdnn_workspace workspace) {
    check_exec(shape.layout, scale.layout, dst.layout, workspace.size);
    auto size = dst.layout.total_nr_elems();
    auto prng = &m_rng.ensure_seed(m_param.seed);
    switch (dst.layout.dtype.enumv()) {
#define cb(_dt)                                                       \
    case DTypeTrait<_dt>::enumv: {                                    \
        using ctype = DTypeTrait<_dt>::ctype;                         \
        MEGDNN_DISPATCH_CPU_KERN_OPR({                                \
            fill_gamma<float>(                                        \
                    prng, dst.ptr<ctype>(), size, shape.ptr<ctype>(), \
                    scale.ptr<ctype>());                              \
        };);                                                          \
        return;                                                       \
    }
        MEGDNN_FOREACH_COMPUTING_DTYPE_FLOAT(cb)
#undef cb
        default:
            megdnn_throw("bad dtype");
    }
}

void MultinomialRNGImpl::exec(
        _megdnn_tensor_in probs, _megdnn_tensor_out dst, _megdnn_workspace workspace) {
    check_exec(probs.layout, dst.layout, workspace.size);
    auto prng = &m_rng.ensure_seed(m_param.seed);
    size_t num_groups = probs.layout.shape[0];
    size_t num_samples = m_param.num_samples;
    size_t len_probs = probs.layout.shape[1];
    bool replacement = m_param.replacement;
    switch (probs.layout.dtype.enumv()) {
#define cb(_dt)                                                                \
    case DTypeTrait<_dt>::enumv: {                                             \
        using ctype = DTypeTrait<_dt>::ctype;                                  \
        MEGDNN_DISPATCH_CPU_KERN_OPR({                                         \
            fill_multinomial(                                                  \
                    prng, probs.ptr<ctype>(), dst.ptr<dt_int32>(), num_groups, \
                    num_samples, len_probs, replacement);                      \
        };);                                                                   \
        return;                                                                \
    }
        MEGDNN_FOREACH_COMPUTING_DTYPE_FLOAT(cb)
#undef cb
        default:
            megdnn_throw("bad dtype");
    }
}

size_t MultinomialRNGImpl::get_workspace_in_bytes(
        const TensorLayout&, const TensorLayout&) {
    return 0;
}

void PoissonRNGImpl::exec(
        _megdnn_tensor_in lam, _megdnn_tensor_inout dst, _megdnn_workspace workspace) {
    check_exec(lam.layout, dst.layout, workspace.size);
    auto size = dst.layout.total_nr_elems();
    auto prng = &m_rng.ensure_seed(m_param.seed);
    switch (dst.layout.dtype.enumv()) {
#define cb(_dt)                                                                  \
    case DTypeTrait<_dt>::enumv: {                                               \
        using ctype = DTypeTrait<_dt>::ctype;                                    \
        MEGDNN_DISPATCH_CPU_KERN_OPR({                                           \
            fill_poisson<float>(prng, dst.ptr<ctype>(), lam.ptr<ctype>(), size); \
        };);                                                                     \
        return;                                                                  \
    }
        MEGDNN_FOREACH_COMPUTING_DTYPE_FLOAT(cb)
#undef cb
        default:
            megdnn_throw("bad dtype");
    }
}

void BetaRNGImpl::exec(
        _megdnn_tensor_in alpha, _megdnn_tensor_in beta, _megdnn_tensor_out dst,
        _megdnn_workspace workspace) {
    check_exec(alpha.layout, beta.layout, dst.layout, workspace.size);
    auto size = dst.layout.total_nr_elems();
    auto prng = &m_rng.ensure_seed(m_param.seed);
    switch (dst.layout.dtype.enumv()) {
#define cb(_dt)                                                                    \
    case DTypeTrait<_dt>::enumv: {                                                 \
        using ctype = DTypeTrait<_dt>::ctype;                                      \
        MEGDNN_DISPATCH_CPU_KERN_OPR({                                             \
            fill_beta<float>(                                                      \
                    prng, dst.ptr<ctype>(), alpha.ptr<ctype>(), beta.ptr<ctype>(), \
                    size);                                                         \
        };);                                                                       \
        return;                                                                    \
    }
        MEGDNN_FOREACH_COMPUTING_DTYPE_FLOAT(cb)
#undef cb
        default:
            megdnn_throw("bad dtype");
    }
}

void PermutationRNGImpl::exec(_megdnn_tensor_inout dst, _megdnn_workspace workspace) {
    check_exec(dst.layout, workspace.size);
    auto size = dst.layout.total_nr_elems();
    auto prng = &m_rng.ensure_seed(m_param.seed);
    switch (dst.layout.dtype.enumv()) {
#define cb(_dt)                                                               \
    case DTypeTrait<_dt>::enumv: {                                            \
        using ctype = DTypeTrait<_dt>::ctype;                                 \
        ctype max_size = DTypeTrait<_dt>::max() - 1;                          \
        megdnn_assert((ctype(size) < max_size));                              \
        MEGDNN_DISPATCH_CPU_KERN_OPR(                                         \
                { fill_permutation<ctype>(prng, dst.ptr<ctype>(), size); };); \
        return;                                                               \
    }
        cb(::megdnn::dtype::Float32) cb(::megdnn::dtype::Int32)
                cb(::megdnn::dtype::Int16)
#undef cb
                        default : megdnn_throw("bad dtype");
    }
}

void ShuffleRNGForwardImpl::exec(
        _megdnn_tensor_in src, _megdnn_tensor_out dst, _megdnn_tensor_out indices,
        _megdnn_workspace workspace) {
    check_exec(src.layout, dst.layout, indices.layout, workspace.size);
    const auto len = indices.layout[0];
    auto prng = &m_rng.ensure_seed(m_param.seed);
    MEGDNN_DISPATCH_CPU_KERN_OPR(
            fill_permutation<dt_int32>(prng, indices.ptr<dt_int32>(), len));
    auto step = 0;
    for (size_t i = 1; i < src.layout.ndim; ++i) {
        step += src.layout[i];
    }
    if (step <= 0)
        step = 1;

#define cb(DType)                                                                 \
    if (src.layout.dtype == DType()) {                                            \
        using T = typename DTypeTrait<DType>::ctype;                              \
        MEGDNN_DISPATCH_CPU_KERN_OPR(shuffle_fwd<T>(                              \
                src.ptr<T>(), dst.ptr<T>(), indices.ptr<dt_int32>(), len, step)); \
        return;                                                                   \
    }
    MEGDNN_FOREACH_COMPUTING_DTYPE(cb)
#undef cb
}

void ShuffleRNGBackwardImpl::exec(
        _megdnn_tensor_in diff, _megdnn_tensor_in indices, _megdnn_tensor_out grad,
        _megdnn_workspace workspace) {
    check_exec(diff.layout, indices.layout, grad.layout, workspace.size);
    const auto len = indices.layout[0];
    auto step = 0;
    for (size_t i = 1; i < diff.layout.ndim; ++i) {
        step += diff.layout[i];
    }
    if (step <= 0)
        step = 1;
#define cb(DType)                                                                   \
    if (diff.layout.dtype == DType()) {                                             \
        using T = typename DTypeTrait<DType>::ctype;                                \
        MEGDNN_DISPATCH_CPU_KERN_OPR(shuffle_bwd<T>(                                \
                grad.ptr<T>(), diff.ptr<T>(), indices.ptr<dt_int32>(), len, step)); \
        return;                                                                     \
    }
    MEGDNN_FOREACH_COMPUTING_DTYPE(cb)
#undef cb
}

void ExponentialRNGImpl::exec(
        _megdnn_tensor_in rate, _megdnn_tensor_inout dst, _megdnn_workspace workspace) {
    check_exec(rate.layout, dst.layout, workspace.size);
    auto size = dst.layout.total_nr_elems();
    auto prng = &m_rng.ensure_seed(m_param.seed);
    switch (dst.layout.dtype.enumv()) {
#define cb(_dt)                                                                       \
    case DTypeTrait<_dt>::enumv: {                                                    \
        using ctype = DTypeTrait<_dt>::ctype;                                         \
        MEGDNN_DISPATCH_CPU_KERN_OPR({                                                \
            fill_exponential<float>(prng, dst.ptr<ctype>(), rate.ptr<ctype>(), size); \
        };);                                                                          \
        return;                                                                       \
    }
        MEGDNN_FOREACH_COMPUTING_DTYPE_FLOAT(cb)
#undef cb
        default:
            megdnn_throw("bad dtype");
    }
}

// vim: syntax=cpp.doxygen
