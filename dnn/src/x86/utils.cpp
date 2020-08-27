/**
 * \file dnn/src/x86/utils.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "src/x86/utils.h"

#include "src/common/utils.h"
#include <xmmintrin.h>

#ifdef _WIN32
// For __cpuid
#include <intrin.h>
#endif

#if MEGDNN_X86_WITH_MKL || MEGDNN_X86_WITH_OPENBLAS
#include <pmmintrin.h>
#endif

using namespace megdnn;
using namespace x86;

namespace {

struct CPUID {
    uint32_t eax, ebx, ecx, edx;
    CPUID()
    {
#if defined(_WIN32)
		int cpuInfo[4];
		__cpuid(cpuInfo, 1);
		eax = cpuInfo[0];
		ebx = cpuInfo[1];
		ecx = cpuInfo[2];
		edx = cpuInfo[3];
#else
        asm volatile(
            "cpuid\n"
            : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx)
            : "a"(1)
            : "cc");
#endif
    }
} cpuid;

bool bit(unsigned x, unsigned y)
{ return (x >> y) & 1; }

MEGDNN_ATTRIBUTE_TARGET("sse")
void transpose4x4_sse(const float *src, float *dst,
        ptrdiff_t lda, ptrdiff_t ldb) {
    __m128 row0 = _mm_loadu_ps(src + 0*lda);
    __m128 row1 = _mm_loadu_ps(src + 1*lda);
    __m128 row2 = _mm_loadu_ps(src + 2*lda);
    __m128 row3 = _mm_loadu_ps(src + 3*lda);
    _MM_TRANSPOSE4_PS(row0, row1, row2, row3);
    _mm_storeu_ps(dst + 0*ldb, row0);
    _mm_storeu_ps(dst + 1*ldb, row1);
    _mm_storeu_ps(dst + 2*ldb, row2);
    _mm_storeu_ps(dst + 3*ldb, row3);
}

void transpose_naive(const float *src, float *dst,
        ptrdiff_t lda, ptrdiff_t ldb, size_t n, size_t m) {
    rep(i, n) rep(j, m) {
        dst[i*ldb + j] = src[j*lda + i];
    }
}

bool feature_detect_avx2()
{
    uint32_t eax, ebx, ecx, edx;

    // check cpu support
#if defined(_WIN32)
    int cpuInfo[4];
    __cpuid(cpuInfo, 7);
    eax = cpuInfo[0];
    ebx = cpuInfo[1];
    ecx = cpuInfo[2];
    edx = cpuInfo[3];
#else
    asm volatile(
        "cpuid\n"
        : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx)
        : "a"(7), "c"(0)
        : "cc");
#endif

    if (!(bit(ebx, 3) && bit(ebx, 5) && bit(ebx, 8)))
        return false;

    // check os support
    asm volatile(
        "xgetbv"
        : "=a"(eax), "=d"(edx)
        : "c"(0));

    return (eax & 6) == 6;

}

bool feature_detect_vnni()
{
    uint32_t eax, ebx, ecx, edx;

    // check cpu support
#if defined(_WIN32)
    int cpuInfo[4];
    __cpuid(cpuInfo, 7);
    eax = cpuInfo[0];
    ebx = cpuInfo[1];
    ecx = cpuInfo[2];
    edx = cpuInfo[3];
#else
    asm volatile(
        "cpuid\n"
        : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx)
        : "a"(7), "c"(0)
        : "cc");
#endif
    //avx512f  ---> 16 ebx
    //avx512dq ---> 17 ebx
    //avx512bw ---> 30 ebx
    //avx512vl ---> 31 ebx
    //avx512vnni --->11 ecx
    if (!(bit(ebx, 16) && bit(ebx, 17) && bit(ebx, 30) && bit(ebx, 31) &&
          bit(ecx, 11)))
        return false;

    // check os support
    asm volatile(
        "xgetbv"
        : "=a"(eax), "=d"(edx)
        : "c"(0));

    return (eax & 6) == 6;

}

bool feature_detect_avx_fma(int ftr) {
    // see Detecting Availability and Support in
    // https://software.intel.com/en-us/articles/introduction-to-intel-advanced-vector-extensions

    // check CPU support
    if (!(bit(cpuid.ecx, 27) && bit(cpuid.ecx, ftr)))
        return false;

    // check OS support
    uint32_t edx, eax;
    asm volatile(
        "xgetbv"
        : "=a"(eax), "=d"(edx)
        : "c"(0));

    return (eax & 6) == 6;
}

bool is_avx_supported = feature_detect_avx_fma(28);
bool is_fma_supported = feature_detect_avx_fma(12);
bool is_avx2_supported = feature_detect_avx2();
bool is_vnni_supported = feature_detect_vnni();

SIMDType disabled_simd_type_thresh = SIMDType::__NR_SIMD_TYPE;

} // anonymous

namespace megdnn {

#ifndef __SSE2__
#error "megdnn at least needs sse2, please compile with -msse2 or higher"
#endif
bool x86::is_supported(SIMDType type) {
    if (type >= disabled_simd_type_thresh)
        return false;

    switch (type) {
        case SIMDType::SSE:
            return bit(cpuid.edx, 25);
        case SIMDType::SSE2:
            return bit(cpuid.edx, 26);
        case SIMDType::SSE3:
            return bit(cpuid.ecx, 0);
        case SIMDType::SSE4_1:
            return bit(cpuid.ecx, 19);
        case SIMDType::SSE4_2:
            return bit(cpuid.ecx, 20);
        case SIMDType::AVX:
            return is_avx_supported;
        case SIMDType::FMA:
            return is_fma_supported;
        case SIMDType::AVX2:
            return is_avx2_supported;
        case SIMDType::VNNI:
            return is_vnni_supported;
        default:
            break;
    }
    megdnn_throw(megdnn_mangle("unknown cpu feature"));
}

void x86::disable_simd_type(SIMDType type) {
    disabled_simd_type_thresh = type;
}

template <>
void transpose(const float *src, float *dst,
        size_t m, size_t n, ptrdiff_t lds, ptrdiff_t ldd) {
    if (lds == -1) {
        lds = n;
    }
    if (ldd == -1) {
        ldd = m;
    }

    for (size_t is = 0; is < n; is += 16) {
        for (size_t js = 0; js < m; js += 16) {
            auto ie = std::min(is+16, n),
                 je = std::min(js+16, m),
                 i = is;
            for (; i+4 <= ie; i += 4) {
                auto j = js;
                for (; j+4 <= je; j += 4) {
                    transpose4x4_sse(
                            src + j*lds + i, dst + i*ldd + j, lds, ldd);
                }
                if (j < je) {
                    transpose_naive(
                            src + j*lds + i, dst + i*ldd + j, lds, ldd,
                            4, je-j);
                }
            }
            if (i < ie) {
                transpose_naive(src + js*lds + i, dst + i*ldd + js,
                        lds, ldd, ie-i, je-js);
            }
        }
    }
}

template <>
void transpose_knc2nsck(const float *src, float *dst,
        size_t k, size_t n, size_t c, size_t n_stride) {
    if (n_stride == k * c) {
        // dst is contiguous
        transpose(src, dst, k, n * c);
    } else {
        for (size_t i = 0; i < n; ++ i) {
            transpose(src + i * c, dst + i * n_stride,
                    k, c, n * c);
        }
    }
}

MEGDNN_ATTRIBUTE_TARGET("sse")
void x86::disable_denorm() {
    _mm_setcsr(_mm_getcsr() | (_MM_FLUSH_ZERO_ON | _MM_DENORMALS_ZERO_ON));
}

} // namespace megdnn

// vim: syntax=cpp.doxygen
