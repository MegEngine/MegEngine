/**
 * \file dnn/src/naive/rng/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "src/naive/handle.h"
#include "src/common/utils.h"
#include "./opr_impl.h"

#include <cmath>

using namespace megdnn;
using namespace naive;

namespace {
    template<typename ctype>
    ctype uniform_int2float(uint64_t x);

    template<>
    dt_float32 uniform_int2float(uint64_t x) {
        union { uint32_t i; dt_float32 f; } u;
        u.i = (0x7F << 23) | (x >> 41);
        return 2 - u.f;
    }

#if !MEGDNN_DISABLE_FLOAT16
    template<>
    dt_float16 uniform_int2float(uint64_t x) {
        union U { uint16_t i; dt_float16 f; U(): f(0) {} } u;
        u.i = (0xF << 10) | (x >> 54);
        return dt_float16(2.f) - u.f;
    }
#endif

#if !MEGDNN_DISABLE_FLOAT16
    template<>
    dt_bfloat16 uniform_int2float(uint64_t x) {
        union U { uint16_t i; dt_bfloat16 f; U(): f(0) {} } u;
        u.i = (0x7F << 7) | (x >> 57);
        return dt_bfloat16(2.f) - u.f;
    }
#endif

    template<typename ctype>
    void fill_uniform(Xoroshiro128plus *rng, ctype *dst, size_t size) {
        for (size_t i = 0; i < size; ++ i) {
            dst[i] = uniform_int2float<ctype>((*rng)());
        }
    }

    template<typename ctype>
    void fill_gaussian(Xoroshiro128plus *rng, ctype *dst, size_t size,
            ctype mean, ctype stddev) {
        // gen gaussian by Box-Muller transform
        for (size_t i = 0; i + 2 <= size; i += 2) {
            ctype u1 = uniform_int2float<ctype>((*rng)()),
                  u2 = uniform_int2float<ctype>((*rng)()),
                  r = ctype(stddev * std::sqrt(-2 * std::log(u1))),
                  theta = ctype(2 * M_PI * u2),
                  z0 = ctype(r * std::cos(theta) + mean),
                  z1 = ctype(r * std::sin(theta) + mean);
            dst[i] = z0;
            dst[i + 1] = z1;
        }
        if (size % 2) {
            ctype u1 = uniform_int2float<ctype>((*rng)()),
                  u2 = uniform_int2float<ctype>((*rng)()),
                  r = ctype(stddev * std::sqrt(-2 * std::log(u1))),
                  theta = ctype(2 * M_PI * u2),
                  z0 = ctype(r * std::cos(theta) + mean);
            dst[size - 1] = z0;
        }
    }

} // anonymous namespace

uint64_t Splitmix64::operator() () {
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

uint64_t Xoroshiro128plus::operator() () {
    const uint64_t s0 = m_s[0];
    uint64_t s1 = m_s[1];
    const uint64_t result = s0 + s1;

    s1 ^= s0;
    m_s[0] = rotl(s0, 55) ^ s1 ^ (s1 << 14); // a, b
    m_s[1] = rotl(s1, 36); // c

    return result;
}


void UniformRNGImpl::exec(
        _megdnn_tensor_inout dst, _megdnn_workspace workspace) {
    check_exec(dst.layout, workspace.size);
    auto size = dst.layout.total_nr_elems();
    auto prng = &m_rng.ensure_seed(m_param.seed);
    switch (dst.layout.dtype.enumv()) {
#define cb(_dt)  \
        case DTypeTrait<_dt>::enumv: \
        { \
            auto ptr = dst.ptr<DTypeTrait<_dt>::ctype>(); \
            MEGDNN_DISPATCH_CPU_KERN_OPR({fill_uniform(prng, ptr, size); }); \
            return; \
        }
        MEGDNN_FOREACH_COMPUTING_DTYPE_FLOAT(cb)
#undef cb
        default:
            megdnn_throw("bad dtype");
    }
}

void GaussianRNGImpl::exec(
        _megdnn_tensor_inout dst, _megdnn_workspace workspace) {
    check_exec(dst.layout, workspace.size);
    auto size = dst.layout.total_nr_elems();
    auto prng = &m_rng.ensure_seed(m_param.seed);
    switch (dst.layout.dtype.enumv()) {
#define cb(_dt)  \
        case DTypeTrait<_dt>::enumv: \
        { \
            using ctype = DTypeTrait<_dt>::ctype; \
            ctype mean(m_param.mean), std(m_param.std); \
            auto ptr = dst.ptr<ctype>(); \
            MEGDNN_DISPATCH_CPU_KERN_OPR({fill_gaussian<ctype>( \
                        prng, ptr, size, mean, std); }); \
            return; \
        }
        MEGDNN_FOREACH_COMPUTING_DTYPE_FLOAT(cb)
#undef cb
        default:
            megdnn_throw("bad dtype");
    }
}

// vim: syntax=cpp.doxygen

