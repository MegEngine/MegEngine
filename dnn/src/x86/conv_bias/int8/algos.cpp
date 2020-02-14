/**
 * \file dnn/src/x86/conv_bias/int8/algos.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "src/x86/conv_bias/int8/algos.h"
#include "src/common/opr_delegate.h"
#include "src/common/utils.h"
#include "src/fallback/convolution/img2col_helper.h"
#include "src/x86/conv_bias/int8/avx2_direct_conv_stride1.h"
#include "src/x86/conv_bias/int8/avx2_direct_conv_stride2.h"
#include "src/x86/conv_bias/opr_impl.h"
#include "src/x86/conv_bias/postprocess_helper.h"
#include "src/x86/handle.h"
#include "src/x86/utils.h"
#if defined(MEGDNN_X86_WITH_MKL_DNN)
#include <mkldnn.hpp>
#endif

#include <cstring>

#if defined(MEGDNN_X86_WITH_MKL_DNN)
using namespace dnnl;
#endif
using namespace megdnn;
using namespace x86;

bool ConvBiasImpl::AlgoDirectAvx2Stride1Int8::usable(
        FallbackConvBiasImpl* /*opr*/, const NCBKernSizeParam& param,
        AlgoSelectionStrategy /*algo_selection_strategy*/) const {
    auto&& fm = param.filter_meta;
    auto FH = fm.spatial[0];
    bool aviliable = ((param.src_type.enumv() == DTypeEnum::QuantizedS8 &&
                       param.filter_type.enumv() == DTypeEnum::QuantizedS8 &&
                       param.dst_type.enumv() == DTypeEnum::QuantizedS8) ||
                      (((param.src_type.enumv() == DTypeEnum::Int8 &&
                         param.filter_type.enumv() == DTypeEnum::Int8 &&
                         param.dst_type.enumv() == DTypeEnum::Int32) ||
                        (param.src_type.enumv() == DTypeEnum::QuantizedS8 &&
                         param.filter_type.enumv() == DTypeEnum::QuantizedS8 &&
                         param.dst_type.enumv() == DTypeEnum::QuantizedS32)) &&
                       param.bias_mode == BiasMode::NO_BIAS &&
                       param.nonlineMode == NonlineMode::IDENTITY)) &&
                     fm.format == Param::Format::NCHW && fm.spatial_ndim == 2 &&
                     fm.dilation[0] == 1 && fm.dilation[1] == 1 &&
                     (FH == 2 || FH == 3 || FH == 5 || FH == 7) &&
                     fm.stride[0] == 1 && fm.stride[1] == 1 &&
                     is_supported(SIMDType::AVX2);
    return aviliable;
}

WorkspaceBundle ConvBiasImpl::AlgoDirectAvx2Stride1Int8::get_bundle(
        const NCBKernSizeParam& param) {
    auto&& fm = param.filter_meta;
    size_t N = param.n;
    size_t IC = fm.icpg;
    size_t OC = fm.ocpg;
    size_t IH = param.isz[0];
    size_t IW = param.isz[1];
    size_t OH = param.osz[0];
    size_t OW = param.osz[1];
    size_t FH = fm.spatial[0];
    size_t FW = fm.spatial[1];
    size_t GROUP = fm.group;
    size_t IC_STEP = 2, OC_STEP = 4, IW_STEP = 8;

    size_t pad_h = fm.padding[0];
    size_t pad_w = fm.padding[1];
    size_t src_size = 0, filter_size = 0;

    //! pack filter, pack src
    filter_size = GROUP * round_up(OC, OC_STEP) * round_up(IC, IC_STEP) * FH *
                  FW * sizeof(int16_t);
    src_size = N * GROUP * div_ceil(IC, IC_STEP) * (IH + 2 * pad_h) *
               round_up(IW + 2 * pad_w, IW_STEP) * 2 * sizeof(int8_t);

    bool need_post_process = param.dst_type.enumv() == DTypeEnum::QuantizedS8;
    if (need_post_process) {
        size_t dst_tmp = N * GROUP * OC * OW * OH * sizeof(int32_t);
        return WorkspaceBundle(nullptr, {src_size, filter_size, dst_tmp});
    } else {
        return WorkspaceBundle(nullptr, {src_size, filter_size});
    }
}

size_t ConvBiasImpl::AlgoDirectAvx2Stride1Int8::get_workspace(
        FallbackConvBiasImpl*, const NCBKernSizeParam& param) const {
    return get_bundle(param).total_size_in_bytes();
}

SmallVector<fallback::ConvBiasImpl::NCBKern>
ConvBiasImpl::AlgoDirectAvx2Stride1Int8::get_kimpls(
        const NCBKernSizeParam& param) const {
    auto bundle = get_bundle(param);
    return direct_conv_avx2_stride1::get_kimpls(param, bundle);
}

#if defined(MEGDNN_X86_WITH_MKL_DNN)
bool ConvBiasImpl::AlgoMkldnnQint8::usable(FallbackConvBiasImpl*,
                                           const NCBKernSizeParam& param,
                                           AlgoSelectionStrategy) const {
    auto&& fm = param.filter_meta;
    return (param.src_type.enumv() == DTypeEnum::QuantizedS8 ||
            param.src_type.enumv() == DTypeEnum::Int8) &&
           (param.dst_type.enumv() == DTypeEnum::QuantizedS32 ||
            param.dst_type.enumv() == DTypeEnum::Int32) &&
           fm.format == param::ConvBias::Format::NCHW && fm.spatial_ndim == 2 &&
           fm.dilation[0] == 1 && fm.dilation[1] == 1 && !fm.should_flip &&
           param.bias_mode == BiasMode::NO_BIAS &&
           param.nonlineMode == NonlineMode::IDENTITY;
}

WorkspaceBundle ConvBiasImpl::AlgoMkldnnQint8::get_bundle(
        const NCBKernSizeParam& param) {
    if (!is_supported(SIMDType::VNNI)) {
        size_t N = param.n;
        size_t IC = param.filter_meta.icpg;
        size_t IH = param.isz[0];
        size_t IW = param.isz[1];

        size_t size = (N * IC * IH * IW) * sizeof(uint8_t);
        return WorkspaceBundle{nullptr, {size}};
    } else {
        return WorkspaceBundle{nullptr, {0}};
    }
}

#define REORDER_MEMORY(megdnn_memory, reorder_memory)                          \
    do {                                                                       \
        if (megdnn_memory.get_desc() != conv_prim_desc.src_desc()) {           \
            reorder_memory = memory(conv_prim_desc.src_desc(), eng_mkldnn);    \
            auto reorder_pd = reorder::primitive_desc(                         \
                    eng_mkldnn, megdnn_memory.get_desc(), eng_mkldnn,          \
                    reorder_memory.get_desc());                                \
            auto reorder_exe = reorder(reorder_pd);                            \
            reorder_exe.execute(stream_mkldnn, megdnn_memory, reorder_memory); \
        } else {                                                               \
            reorder_memory = megdnn_memory;                                    \
        }                                                                      \
    } while (0)

void ConvBiasImpl::AlgoMkldnnQint8::kern_mkldnn_s8x8x32(
        const NCBKernParam& param, const NCBKernIndex&) {
    UNPACK_CONV_F32_NCB_KERN_SIZES(param);
    MEGDNN_MARK_USED_VAR(N);
    auto x86_handle = static_cast<HandleImpl*>(inplace_cpu_handle().get());
    megdnn_assert(x86_handle != nullptr, "x86 handle can not be null");
    auto eng_mkldnn = x86_handle->mkldnn_engine();
    auto stream_mkldnn = x86_handle->mkldnn_stream();

    memory::dims src_shape = {1, IC, IH, IW};
    memory::dims weight_shape = {OC, IC, FH, FW};
    memory::dims dst_shape = {1, OC, OH, OW};
    memory::dims strides_shape = {SH, SW};
    memory::dims padding_shape = {PH, PW};

    auto megdnn_src_md = memory::desc({src_shape}, memory::data_type::s8,
                                      memory::format_tag::nchw);
    auto megdnn_weight_md = memory::desc({weight_shape}, memory::data_type::s8,
                                         memory::format_tag::oihw);
    auto megdnn_dst_md = memory::desc({dst_shape}, memory::data_type::s32,
                                      memory::format_tag::nchw);

    auto megdnn_weight_memory = memory(megdnn_weight_md, eng_mkldnn,
                                       const_cast<void*>(param.filter_ptr));
    int8_t* src = const_cast<int8_t*>(param.src<int8_t>());
    int32_t* dst = param.dst<int32_t>();

    auto megdnn_src_memory =
            memory(megdnn_src_md, eng_mkldnn, static_cast<void*>(src));

    auto megdnn_dst_memory =
            memory(megdnn_dst_md, eng_mkldnn, static_cast<void*>(dst));
    // Intel mkldnn compute s8*s8-->s32 convolution in none vnni machine is
    // not crect, this based https://github.com/intel/mkl-dnn/issues/375. In
    // the vnni machine s8*s8--->s32 must use reorder, can't use the megdnn
    // origin ptr, but u8*s8--->s32,mkldnn can use megdnn origin ptr
    // directly, if machine does not support vnni, there is a naive mkl-dnn
    // implement
    if (is_supported(SIMDType::VNNI)) {
        auto conv_src_md = memory::desc({src_shape}, memory::data_type::s8,
                                        memory::format_tag::any);
        auto conv_weights_md = memory::desc(
                {weight_shape}, memory::data_type::s8, memory::format_tag::any);
        auto conv_dst_md = memory::desc({dst_shape}, memory::data_type::s32,
                                        memory::format_tag::any);

        auto conv_desc = convolution_forward::desc(
                prop_kind::forward, algorithm::convolution_auto, conv_src_md,
                conv_weights_md, conv_dst_md, strides_shape, padding_shape,
                padding_shape);

        auto conv_prim_desc =
                convolution_forward::primitive_desc(conv_desc, eng_mkldnn);

        auto conv = convolution_forward(conv_prim_desc);

        memory conv_src_memory, conv_weight_memory, conv_dst_memory;

        REORDER_MEMORY(megdnn_src_memory, conv_src_memory);
        REORDER_MEMORY(megdnn_weight_memory, conv_weight_memory);

        if (megdnn_dst_memory.get_desc() != conv_prim_desc.dst_desc()) {
            conv_dst_memory = memory(conv_prim_desc.dst_desc(), eng_mkldnn);
        } else {
            conv_dst_memory = megdnn_dst_memory;
        }

        conv.execute(stream_mkldnn, {{DNNL_ARG_SRC, conv_src_memory},
                                     {DNNL_ARG_WEIGHTS, conv_weight_memory},
                                     {DNNL_ARG_DST, conv_dst_memory}});
        REORDER_MEMORY(megdnn_dst_memory, conv_dst_memory);
        stream_mkldnn.wait();
    } else {
        std::vector<primitive> net;
        std::vector<std::unordered_map<int, memory>> net_args;

        uint8_t* const_128 = static_cast<uint8_t*>(param.workspace_ptr);
        std::memset(const_128, 128u, get_bundle(param).total_size_in_bytes());

        auto megdnn_128_md = memory::desc({src_shape}, memory::data_type::u8,
                                          memory::format_tag::nchw);
        auto megdnn_128_memory = memory(megdnn_128_md, eng_mkldnn,
                                        static_cast<void*>(const_128));

        // 1.compute the conv 128 * weight(s8) -> s32
        auto conv_128_dst_memory = memory(megdnn_dst_md, eng_mkldnn);

        auto conv_desc1 = convolution_forward::desc(
                prop_kind::forward, algorithm::convolution_auto, megdnn_128_md,
                megdnn_weight_md, megdnn_dst_md, strides_shape, padding_shape,
                padding_shape);

        auto conv_prim_desc1 =
                convolution_forward::primitive_desc(conv_desc1, eng_mkldnn);

        net.push_back(convolution_forward(conv_prim_desc1));
        net_args.push_back({{DNNL_ARG_SRC, megdnn_128_memory},
                            {DNNL_ARG_WEIGHTS, megdnn_weight_memory},
                            {DNNL_ARG_DST, conv_128_dst_memory}});

        // 2.compute the conv (src+128)(u8) *weight(s8) --> s32
        //(1) src+128
        memory conv_src_add_128_memory = megdnn_128_memory;
        auto sum_128_desc = sum::primitive_desc(
                conv_src_add_128_memory.get_desc(), {1.0f, 1.0f},
                {megdnn_128_md, megdnn_src_md}, eng_mkldnn);

        net.push_back(sum(sum_128_desc));
        net_args.push_back({{DNNL_ARG_MULTIPLE_SRC, megdnn_128_memory},
                            {DNNL_ARG_MULTIPLE_SRC + 1, megdnn_src_memory},
                            {DNNL_ARG_DST, conv_src_add_128_memory}});
        //(2) conv (src+128)(u8) * weight(s8) --> s32
        auto conv_desc2 = convolution_forward::desc(
                prop_kind::forward, algorithm::convolution_auto, megdnn_128_md,
                megdnn_weight_md, megdnn_dst_md, strides_shape, padding_shape,
                padding_shape);

        auto conv_prim_desc2 =
                convolution_forward::primitive_desc(conv_desc2, eng_mkldnn);

        net.push_back(convolution_forward(conv_prim_desc2));
        net_args.push_back({{DNNL_ARG_SRC, conv_src_add_128_memory},
                            {DNNL_ARG_WEIGHTS, megdnn_weight_memory},
                            {DNNL_ARG_DST, megdnn_dst_memory}});
        // 3.sub the 128*weight
        auto sub_128_desc =
                sum::primitive_desc(megdnn_dst_md, {1.0f, -1.0f},
                                    {megdnn_dst_md, megdnn_dst_md}, eng_mkldnn);

        net.push_back(sum(sub_128_desc));
        net_args.push_back({{DNNL_ARG_MULTIPLE_SRC, megdnn_dst_memory},
                            {DNNL_ARG_MULTIPLE_SRC + 1, conv_128_dst_memory},
                            {DNNL_ARG_DST, megdnn_dst_memory}});
        // 4 excute
        for (size_t i = 0; i < net.size(); ++i) {
            net.at(i).execute(stream_mkldnn, net_args.at(i));
        }

        stream_mkldnn.wait();
    }
}

#undef REORDER_MEMORY
#endif

#if defined(MEGDNN_X86_WITH_MKL_DNN)
/* ===================== mkldnn qint8 matmul algo ===================== */
bool ConvBiasImpl::AlgoMkldnnMatmulQint8::usable(FallbackConvBiasImpl*,
                                                 const NCBKernSizeParam& param,
                                                 AlgoSelectionStrategy) const {
    auto&& fm = param.filter_meta;
    return (param.src_type.enumv() == DTypeEnum::QuantizedS8 ||
            param.src_type.enumv() == DTypeEnum::Int8) &&
           (param.dst_type.enumv() == DTypeEnum::QuantizedS32 ||
            param.dst_type.enumv() == DTypeEnum::Int32) &&
           fm.format == param::ConvBias::Format::NCHW && fm.spatial_ndim == 2 &&
           fm.group == 1 && fm.dilation[0] == 1 && fm.dilation[1] == 1 &&
           param.bias_mode == BiasMode::NO_BIAS &&
           param.nonlineMode == NonlineMode::IDENTITY &&
           //! The matmul opr is only used in single thread
           //! TODO:support the no pack matmul algo in fallback im2col + matmul
           param.nr_threads == 1_z;
}
bool ConvBiasImpl::AlgoMkldnnMatmulQint8::is_preferred(
        FallbackConvBiasImpl*, const NCBKernSizeParam& param) const {
    auto&& fm = param.filter_meta;
    megdnn_assert_internal(fm.group == 1 && fm.dilation[0] == 1 &&
                           fm.dilation[1] == 1);

    // single channel conv should never use matrix mul
    if (fm.ocpg == 1 || fm.icpg == 1)
        return false;
    return true;
}
WorkspaceBundle ConvBiasImpl::AlgoMkldnnMatmulQint8::get_bundle(
        const NCBKernSizeParam& param) {
    UNPACK_CONV_F32_NCB_KERN_SIZES(param);
    megdnn_ignore(N);
    megdnn_ignore(OC);
    auto IW2 = IH + 2 * PH;
    auto IH2 = IW + 2 * PW;
    bool can_matrix_mul_direct =
            (FH == 1 && FW == 1 && SH == 1 && SW == 1 && PH == 0 && PW == 0);
    // temp space to store padding-free src (with 4 extra floats)
    // temp space to store unrolled matrix (with 4 extra floats)
    // workspace for matrix mul opr
    size_t part0, part1, part2;
    if (can_matrix_mul_direct) {
        part0 = part1 = 0;
    } else {
        part0 = (IC * IH2 * IW2 + 4) * sizeof(int8_t);
        part1 = (IC * FH * FW * OH * OW + 4) * sizeof(int8_t);
    }
    {
        TensorLayout A_, B_, C_;
        A_ = TensorLayout({OC, IC * FH * FW}, dtype::Int8());
        B_ = TensorLayout({IC * FH * FW, OH * OW}, dtype::Int8());
        C_ = TensorLayout({OC, OH * OW}, dtype::Int32());
        part2 = get_matmul_opr()->get_workspace_in_bytes(A_, B_, C_);
    }
    return {nullptr, {part0, part1, part2}};
}
MatrixMul* ConvBiasImpl::AlgoMkldnnMatmulQint8::get_matmul_opr() {
    static CpuOprDelegationStorage<> storage;
    return storage.get<MatrixMul>();
}

void ConvBiasImpl::AlgoMkldnnMatmulQint8::kern_mkldnn_matmul_s8x8x32(
        const NCBKernParam& param, const NCBKernIndex&) {
    UNPACK_CONV_F32_NCB_KERN_SIZES(param);
    auto IH2 = IH + 2 * PH;
    auto IW2 = IW + 2 * PW;
    bool is_xcorr = !param.filter_meta.should_flip;
    auto bundle = get_bundle(param);
    bundle.set(param.workspace_ptr);

    for (size_t n = 0; n < N; ++n) {
        int8_t* src =
                const_cast<int8_t*>(param.src<int8_t>()) + n * param.inp_bs;
        int32_t* dst = param.dst<int32_t>() + n * param.out_bs;
        int8_t *B, *src2;
        if (FH == 1 && FW == 1 && SH == 1 && SW == 1 && PH == 0 && PW == 0) {
            // special case: 1x1
            B = src;
        } else {
            src2 = static_cast<int8_t*>(bundle.get(0));
            // copy src to src2;
            int8_t* src2_ptr = src2;
            const int8_t* src_ptr = src;
            rep(ic, IC) {
                if (PH != 0) {
                    std::memset(src2_ptr, 0, sizeof(int8_t) * PH * IW2);
                    src2_ptr += PH * IW2;
                }
                rep(ih, IH) {
                    if (PW != 0)
                        rep(pw, PW) * (src2_ptr++) = 0.0f;
                    std::memcpy(src2_ptr, src_ptr, sizeof(int8_t) * IW);
                    src2_ptr += IW;
                    src_ptr += IW;
                    if (PW != 0)
                        rep(pw, PW) * (src2_ptr++) = 0.0f;
                }
                if (PH != 0) {
                    std::memset(src2_ptr, 0, sizeof(int8_t) * PH * IW2);
                    src2_ptr += PH * IW2;
                }
            }

            B = static_cast<int8_t*>(bundle.get(1));
            if (SH == 1 && SW == 1) {
                if (is_xcorr) {
                    img2col<true>(src2, B, OC, OH, OW, IC, IH2, IW2, FH, FW);
                } else {
                    img2col<false>(src2, B, OC, OH, OW, IC, IH2, IW2, FH, FW);
                }
            } else {
                if (is_xcorr) {
                    img2col_stride<true>(src2, B, OC, OH, OW, IC, IH2, IW2, FH,
                                         FW, SH, SW);
                } else {
                    img2col_stride<false>(src2, B, OC, OH, OW, IC, IH2, IW2, FH,
                                          FW, SH, SW);
                }
            }
        }
        {
            TensorND A_, B_, C_;
            A_.layout = TensorLayout({OC, IC * FH * FW}, dtype::Int8());
            A_.raw_ptr = const_cast<int8_t*>(param.filter<int8_t>());
            B_.layout = TensorLayout({IC * FH * FW, OH * OW}, dtype::Int8());
            B_.raw_ptr = B;
            C_.layout = TensorLayout({OC, OH * OW}, dtype::Int32());
            C_.raw_ptr = dst;
            Workspace workspace(static_cast<dt_byte*>(bundle.get(2)),
                                bundle.get_size(2));
            get_matmul_opr()->exec(A_, B_, C_, workspace);
        }
    }
}

#endif
/* ===================== avx2 int8 stride 2 ===================== */
bool ConvBiasImpl::AlgoAVX2DirectConvStride2::usable(
        FallbackConvBiasImpl* /*opr*/, const NCBKernSizeParam& param,
        AlgoSelectionStrategy) const {
    auto&& fm = param.filter_meta;
    auto FH = fm.spatial[0];
    bool aviliable = ((param.src_type.enumv() == DTypeEnum::QuantizedS8 &&
                       param.filter_type.enumv() == DTypeEnum::QuantizedS8 &&
                       param.dst_type.enumv() == DTypeEnum::QuantizedS8) ||
                      (((param.src_type.enumv() == DTypeEnum::Int8 &&
                         param.filter_type.enumv() == DTypeEnum::Int8 &&
                         param.dst_type.enumv() == DTypeEnum::Int32) ||
                        (param.src_type.enumv() == DTypeEnum::QuantizedS8 &&
                         param.filter_type.enumv() == DTypeEnum::QuantizedS8 &&
                         param.dst_type.enumv() == DTypeEnum::QuantizedS32)) &&
                       param.bias_mode == BiasMode::NO_BIAS &&
                       param.nonlineMode == NonlineMode::IDENTITY)) &&
                     fm.format == Param::Format::NCHW && fm.spatial_ndim == 2 &&
                     fm.dilation[0] == 1 && fm.dilation[1] == 1 &&
                     (FH == 2 || FH == 3 || FH == 5 || FH == 7) &&
                     fm.stride[0] == 2 && fm.stride[1] == 2 &&
                     is_supported(SIMDType::AVX2);
    return aviliable;
}

WorkspaceBundle ConvBiasImpl::AlgoAVX2DirectConvStride2::get_bundle(
        const NCBKernSizeParam& param) {
    auto&& fm = param.filter_meta;
    size_t N = param.n;
    size_t IC = fm.icpg;
    size_t OC = fm.ocpg;
    size_t IH = param.isz[0];
    size_t IW = param.isz[1];
    size_t OH = param.osz[0];
    size_t OW = param.osz[1];
    size_t FH = fm.spatial[0];
    size_t FW = fm.spatial[1];
    size_t GROUP = fm.group;
    size_t IC_STEP = 2, OC_STEP = 4;

    size_t pad_h = fm.padding[0];
    size_t pad_w = fm.padding[1];
    size_t src_size = 0, filter_size = 0;

    //! pack filter, pack src
    filter_size = GROUP * round_up(OC, OC_STEP) * round_up(IC, IC_STEP) * FH *
                  FW * sizeof(int16_t);
    //! avx256 iw max offset 32, caused by w_remain < 16
    src_size = N * GROUP * div_ceil(IC, IC_STEP) * (IH + 2 * pad_h) *
                       (IW + 2 * pad_w) * 2 * sizeof(int8_t) +
               32;
    bool need_post_process = param.dst_type.enumv() == DTypeEnum::QuantizedS8;
    if (need_post_process) {
        size_t dst_tmp = N * GROUP * OC * OW * OH * sizeof(int32_t);
        return WorkspaceBundle(nullptr, {src_size, filter_size, dst_tmp});
    } else {
        return WorkspaceBundle(nullptr, {src_size, filter_size});
    }
}

size_t ConvBiasImpl::AlgoAVX2DirectConvStride2::get_workspace(
        FallbackConvBiasImpl*, const NCBKernSizeParam& param) const {
    return get_bundle(param).total_size_in_bytes();
}

SmallVector<fallback::ConvBiasImpl::NCBKern>
ConvBiasImpl::AlgoAVX2DirectConvStride2::get_kimpls(
        const NCBKernSizeParam& param) const {
    auto bundle = get_bundle(param);
    return direct_conv_avx2_stride2::get_kimpls(param, bundle);
}

// vim: syntax=cpp.doxygen
