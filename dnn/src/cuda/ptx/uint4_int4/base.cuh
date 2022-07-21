#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

#if ((__CUDACC_VER_MAJOR__ > 11) || \
     (__CUDACC_VER_MAJOR__ == 11 && __CUDACC_VER_MINOR__ >= 0))
#define SM80_SUPPORTED
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))
#define SM80_ENABLED
#endif
#endif

namespace convolution {
class Uint32Fastdiv {
    uint32_t m_mul, m_divisor, m_divisor_is_not_1, m_inc_dividend, m_shift;

public:
    Uint32Fastdiv();

    Uint32Fastdiv(uint32_t d) { operator=(d); }

    //! set the divisor to be d
    Uint32Fastdiv& operator=(uint32_t d);

    //! caller must ensure that dividend would not exceed this number
    static constexpr uint32_t MAX_DIVIDEND = ~0u - 1;

    __device__ __forceinline__ uint32_t divisor() const { return m_divisor; }

    __device__ __forceinline__ uint32_t divide(uint32_t dividend) const {
        uint32_t ans_for_one = dividend & ~m_divisor_is_not_1,
                 dfix = dividend + m_inc_dividend,
#if __CUDA_ARCH__
                 hi32 = __umulhi(dfix, m_mul),
#else
                 hi32 = ((uint64_t)dfix * m_mul) >> 32,
#endif
                 ans = hi32 >> m_shift;

        return (ans & m_divisor_is_not_1) | ans_for_one;
    }
};

static __forceinline__ __device__ uint32_t
operator/(uint32_t a, const Uint32Fastdiv& d) {
    return d.divide(a);
}

static __forceinline__ __device__ uint32_t
operator%(uint32_t a, const Uint32Fastdiv& d) {
    return a - d.divisor() * d.divide(a);
}

struct Conv2dInt4Param {
    uint32_t n, ic, ih, iw, fh, fw, sh, sw, ph, pw, oc, oh, ow;
    uint32_t ibs, ics, ihs;
    uint32_t obs, ocs, ohs;
    uint32_t icfhfw;
    uint32_t nhw;
    Uint32Fastdiv div_ohow;
    Uint32Fastdiv div_ow;
    Conv2dInt4Param(
            uint32_t n, uint32_t ic, uint32_t ih, uint32_t iw, uint32_t fh, uint32_t fw,
            uint32_t sh, uint32_t sw, uint32_t ph, uint32_t pw, uint32_t oc,
            uint32_t oh, uint32_t ow, uint32_t interleaved)
            : n(n),
              ic(ic),
              ih(ih),
              iw(iw),
              fh(fh),
              fw(fw),
              sh(sh),
              sw(sw),
              ph(ph),
              pw(pw),
              oc(oc),
              oh(oh),
              ow(ow) {
        constexpr uint32_t size_bits = 4;
        // all stride size in bytes
        ibs = ic * ih * iw * size_bits / 8;
        ics = ih * iw * interleaved * size_bits / 8;
        ihs = iw * interleaved * size_bits / 8;
        obs = oc * oh * ow * size_bits / 8;
        ocs = oh * ow * interleaved * size_bits / 8;
        ohs = ow * interleaved * size_bits / 8;
        icfhfw = ic * fh * fw;
        nhw = n * oh * ow;
        div_ohow = oh * ow;
        div_ow = ow;
    }
};

struct Conv2dConstantOffsetParam {
    int32_t begin;
    int32_t size;
    int32_t max;
    int32_t rewind;
};

#define CONSTANT_BUFFER_SIZE 848

struct Conv2dConstantOffset {
    Conv2dConstantOffsetParam c_offset_param;
    int c_offset[CONSTANT_BUFFER_SIZE];
};

}  // namespace convolution
