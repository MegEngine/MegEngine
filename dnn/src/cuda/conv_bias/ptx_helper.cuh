#pragma once
#include "src/cuda/int_fastdiv.cuh"
#include "src/cuda/utils.cuh"

namespace megdnn {
namespace cuda {
namespace ptx {

struct Conv2dInt8Param {
    uint32_t n, ic, ih, iw, fh, fw, sh, sw, ph, pw, oc, oh, ow;
    uint32_t ibs, ics, ihs;
    uint32_t obs, ocs, ohs;
    uint32_t icfhfw;
    uint32_t nhw;
    uint32_t oc32;
    Uint32Fastdiv div_ohow;
    Uint32Fastdiv div_ow;
    Conv2dInt8Param(
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
        ibs = ic * ih * iw;
        ics = ih * iw * interleaved;
        ihs = iw * interleaved;
        obs = oc * oh * ow;
        ocs = oh * ow * interleaved;
        ohs = ow * interleaved;
        icfhfw = ic * fh * fw;
        div_ohow = oh * ow;
        div_ow = ow;
        nhw = n * oh * ow;
        // used for dp4a kernel, reduce usage of register file
        oc32 = oc * 32;
    }
};

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
            uint32_t oh, uint32_t ow, uint32_t interleaved = 64)
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

template <uint32_t size_bits, uint32_t interleaved>
void reorder_imma_filter_bias(
        int8_t* dst_filter, float* dst_bias, const int8_t* src_filter,
        const int32_t* src_bias, float bias_scale, uint32_t OC, uint32_t IC,
        uint32_t FH, uint32_t FW, cudaStream_t stream);

template <uint32_t size_bits, uint32_t interleaved>
void reorder_imma_filter_bias_fusion_zero_point(
        int8_t* dst_filter, float* dst_bias, const int8_t* src_filter,
        const int32_t* src_bias, float bias_scale, const int32_t* reduce_filter,
        float zero_point, uint32_t OC, uint32_t IC, uint32_t FH, uint32_t FW,
        cudaStream_t stream);
}  // namespace ptx
}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cuda.doxygen
