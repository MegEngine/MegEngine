#pragma once
#include <cuda_runtime_api.h>
#include <stdint.h>

#include <curand.h>
#include <curand_kernel.h>

#include "megdnn/dtype.h"
#include "src/cuda/elemwise_helper.cuh"
#include "src/cuda/utils.cuh"

#if MEGDNN_CC_HOST
#include "megdnn/oprs.h"
#endif

namespace megdnn {
namespace cuda {
namespace random {

using Philox = curandStatePhilox4_32_10_t;

QUALIFIERS float _curand_uniform(Philox* state) {
    float r = curand_uniform(state);
    if (r >= 1.0f) {
        r = 0.0f;
    }
    return r;
}

template <typename ctype, typename = void>
struct RandomKernel;

template <typename ctype>
using enable_64bit = typename std::enable_if<
        std::is_integral<ctype>::value && ((sizeof(ctype)) == 8)>::type;

template <typename ctype>
using enable_32bit = typename std::enable_if<
        std::is_integral<ctype>::value && ((sizeof(ctype)) <= 4)>::type;

template <typename ctype>
struct RandomKernel<ctype, enable_64bit<ctype>> {
    ctype* output;
    uint64_t seed, offset;
    uint64_t mask = static_cast<uint64_t>(std::numeric_limits<ctype>::max());
    __device__ void operator()(uint32_t idx) {
        Philox local_state;
        curand_init(seed, idx, offset, &local_state);
        uint4 rand = curand4(&local_state);
        uint64_t val = (static_cast<uint64_t>(rand.x) << 32) | rand.y;
        output[idx] = static_cast<ctype>(val & mask);
    }
#if MEGDNN_CC_HOST
    RandomKernel(const ctype* output, uint64_t seed, uint64_t offset)
            : output{output}, seed{seed}, offset{offset} {}
#endif
};

template <typename ctype>
struct RandomKernel<ctype, enable_32bit<ctype>> {
    ctype* output;
    uint64_t seed, offset;
    uint32_t mask = static_cast<uint32_t>(std::numeric_limits<ctype>::max());
    __device__ void operator()(uint32_t idx) {
        Philox local_state;
        curand_init(seed, idx, offset, &local_state);
        uint32_t val = curand(&local_state);
        output[idx] = static_cast<ctype>(val & mask);
    }
#if MEGDNN_CC_HOST
    RandomKernel(const ctype* output, uint64_t seed, uint64_t offset)
            : output{output}, seed{seed}, offset{offset} {}
#endif
};

template <typename ctype>
struct RangeKernel {
    ctype* output;
    __device__ void operator()(uint32_t idx) { output[idx] = static_cast<ctype>(idx); }
#if MEGDNN_CC_HOST
    RangeKernel(const ctype* output) : output{output} {}
#endif
};

template <typename ctype_src, typename ctype_dst>
struct AsTypeKernel {
    ctype_src* input;
    ctype_dst* output;
    using ctype_mask = typename std::conditional<
            std::is_integral<ctype_dst>::value, ctype_dst, ctype_src>::type;
    ctype_src mask = static_cast<ctype_src>(std::numeric_limits<ctype_mask>::max());
    __device__ void operator()(uint32_t idx) {
        output[idx] = static_cast<ctype_dst>(input[idx] & mask);
    }
#if MEGDNN_CC_HOST
    AsTypeKernel(const ctype_src* input, const ctype_dst* output)
            : input{input}, output{output} {}
#endif
};

template <typename ctype>
struct GammaKernel {
    ctype* output;
    ctype* shape;
    ctype* scale;
    uint64_t seed, offset;

    static __device__ float sample_gamma(float a, float b, Philox* state) {
        float scale = b;
        if (a <= 0)
            return 0.f;
        if (a < 1.0f) {
            scale *= powf(_curand_uniform(state), 1.0f / a);
            a += 1.0f;
        }
        float d = a - 1.0f / 3.0f;
        float c = 1.0f / sqrtf(9.0f * d);
        while (1) {
            float x, y;
            x = curand_normal(state);
            y = 1.0f + c * x;
            if (y <= 0)
                continue;

            float v = y * y * y;
            float u = _curand_uniform(state);
            float xx = x * x;

            if ((u < 1.0f - 0.0331f * xx * xx) ||
                logf(u) < 0.5f * xx + d * (1.0f - v + logf(v)))
                return scale * d * v;
        }
    }

    __device__ void operator()(uint32_t idx) {
        Philox local_state;
        curand_init(seed, idx, offset, &local_state);
        float a = static_cast<float>(shape[idx]);
        float b = static_cast<float>(scale[idx]);
        output[idx] = static_cast<ctype>(sample_gamma(a, b, &local_state));
    }

#if MEGDNN_CC_HOST
    GammaKernel(
            const TensorND& output, const TensorND& shape, const TensorND& scale,
            uint64_t seed, uint64_t offset)
            : output{output.ptr<ctype>()},
              shape{shape.ptr<ctype>()},
              scale{scale.ptr<ctype>()},
              seed{seed},
              offset{offset} {}
#endif
};

template <typename ctype>
struct PoissonKernel {
    ctype* output;
    ctype* lambda;
    uint64_t seed, offset;

    __device__ void operator()(uint32_t idx) {
        Philox local_state;
        curand_init(seed, idx, offset, &local_state);
        float lam = static_cast<float>(lambda[idx]);
        output[idx] = static_cast<ctype>(curand_poisson(&local_state, lam));
    }

#if MEGDNN_CC_HOST
    PoissonKernel(
            const TensorND& output, const TensorND& lambda, uint64_t seed,
            uint64_t offset)
            : output{output.ptr<ctype>()},
              lambda{lambda.ptr<ctype>()},
              seed{seed},
              offset{offset} {}
#endif
};

template <typename ctype>
struct BetaKernel {
    ctype* output;
    ctype* alpha;
    ctype* beta;
    uint64_t seed, offset;

    __device__ void operator()(uint32_t idx) {
        Philox local_state;
        curand_init(seed, idx, offset, &local_state);
        float a = static_cast<float>(alpha[idx]);
        float b = static_cast<float>(beta[idx]);
        if (a <= 0 || b <= 0) {
            output[idx] = 0;
            return;
        }
        if (a < 1.0f && b < 1.0f) {
            float u, v, x, y;
            while (true) {
                u = _curand_uniform(&local_state);
                v = _curand_uniform(&local_state);
                x = powf(u, 1.0f / a);
                y = powf(v, 1.0f / b);
                if (x + y < 1.0f) {
                    if (x + y > 0) {
                        output[idx] = static_cast<ctype>(x / (x + y));
                        return;
                    } else {
                        float logx = logf(u) / a;
                        float logy = logf(v) / b;
                        float log_max = logx > logy ? logx : logy;
                        logx -= log_max;
                        logy -= log_max;
                        output[idx] = static_cast<ctype>(
                                exp(logx - log(exp(logx) + exp(logy))));
                        return;
                    }
                }
            }
        } else {
            float ga = GammaKernel<float>::sample_gamma(a, 1.0f, &local_state);
            float gb = GammaKernel<float>::sample_gamma(b, 1.0f, &local_state);
            output[idx] = static_cast<ctype>(ga / (ga + gb));
            return;
        }
    }

#if MEGDNN_CC_HOST
    BetaKernel(
            const TensorND& output, const TensorND& alpha, const TensorND& beta,
            uint64_t seed, uint64_t offset)
            : output{output.ptr<ctype>()},
              alpha{alpha.ptr<ctype>()},
              beta{beta.ptr<ctype>()},
              seed{seed},
              offset{offset} {}
#endif
};

template <typename ctype>
struct ExponentialKernel {
    ctype* output;
    ctype* rate;
    uint64_t seed, offset;

    __device__ void operator()(uint32_t idx) {
        Philox local_state;
        curand_init(seed, idx, offset, &local_state);
        float rate_float = static_cast<float>(rate[idx]);
        // curand_uniform has (0,1] bounds. log(1) is 0 and exponential excludes 0.
        float uniform_rand = curand_uniform(&local_state);
        float epsilon = 1e-9f;
        float log_rand = uniform_rand >= 1 - epsilon ? -epsilon : logf(uniform_rand);
        output[idx] = static_cast<ctype>(-log_rand / rate_float);
    }

#if MEGDNN_CC_HOST
    ExponentialKernel(
            const TensorND& output, const TensorND& rate, uint64_t seed,
            uint64_t offset)
            : output{output.ptr<ctype>()},
              rate{rate.ptr<ctype>()},
              seed{seed},
              offset{offset} {}
#endif
};

template <typename ctype>
void permutation_forward(
        ctype* dst, void* workspace, size_t size, uint64_t seed, uint64_t offset,
        cudaStream_t stream);

size_t get_permutation_workspace_in_bytes(size_t N);

template <typename T>
void shuffle_forward(
        T* sptr, T* dptr, dt_int32* iptr, size_t len, size_t step, cudaStream_t stream);

template <typename T>
void shuffle_backward(
        T* dptr, dt_int32* iptr, T* sptr, size_t len, size_t step, cudaStream_t stream);

#define ARGSORT_FOREACH_CTYPE(cb) cb(float) cb(int32_t) DNN_INC_FLOAT16(cb(dt_float16))

}  // namespace random
}  // namespace cuda
}  // namespace megdnn
