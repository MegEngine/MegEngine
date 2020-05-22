/**
 * \file dnn/src/x86/pooling/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "src/x86/pooling/opr_impl.h"
#include "src/common/opr_delegate.h"
#include "src/common/utils.h"
#include "src/naive/handle.h"
#include "src/x86/handle.h"
#include "src/x86/pooling/do_max_pooling_3x3_s2x2_float_sse.h"
#include "src/x86/pooling/pooling_special_cases.h"
#include "src/x86/utils.h"

#if MEGDNN_X86_WITH_MKL_DNN
#include "mkldnn.hpp"
#endif

using namespace megdnn;
using namespace x86;

namespace {

WorkspaceBundle get_bundle(const TensorLayout& src, const TensorLayout& dst,
                           const param::Pooling& param) {
    megdnn_assert(
            is_supported(SIMDType::SSE) && src.dtype == dtype::Float32() &&
            param.format == param::Pooling::Format::NCHW &&
            param.mode == param::Pooling::Mode::MAX && param.window_h == 3 &&
            param.window_w == 3 && param.stride_h == 2 && param.stride_w == 2);
    //! max pooling 3x3 stride 2
    auto IW = src.shape[3];
    auto OW = dst.shape[3];

    WorkspaceBundle ws(nullptr,
                       {OW * src.dtype.size(), OW * src.dtype.size(),
                        OW * src.dtype.size(), (IW + 1) / 2 * src.dtype.size(),
                        (IW + 1) / 2 * src.dtype.size()},
                       16);
    return ws;
}

#if MEGDNN_X86_WITH_MKL_DNN
template <dnnl::memory::format_tag format_tag, bool use_mkl_mem>
dnnl::memory tensor_to_mkl_memory(_megdnn_tensor_in src,
                                  const dnnl::engine& mkldnn_eng,
                                  dnnl::memory::data_type mkldnn_datatype) {
    megdnn_assert(format_tag == dnnl::memory::format_tag::nChw8c ||
                          format_tag == dnnl::memory::format_tag::nchw ||
                          format_tag == dnnl::memory::format_tag::nhwc,
                  "not support format");

    dnnl::memory::dims src_shape = {
            static_cast<long>(src.layout[0]), static_cast<long>(src.layout[1]),
            static_cast<long>(src.layout[2]), static_cast<long>(src.layout[3])};
    if (format_tag == dnnl::memory::format_tag::nChw8c) {
        src_shape = {static_cast<long>(src.layout[0]),
                     static_cast<long>(src.layout[1] * 8),
                     static_cast<long>(src.layout[2]),
                     static_cast<long>(src.layout[3])};
    }
    auto megdnn_src_md =
            dnnl::memory::desc({src_shape}, mkldnn_datatype, format_tag);
    if (use_mkl_mem) {
        auto megdnn_src_memory = dnnl::memory(megdnn_src_md, mkldnn_eng);
        return megdnn_src_memory;
    } else {
        auto megdnn_src_memory = dnnl::memory(megdnn_src_md, mkldnn_eng,
                                              const_cast<void*>(src.raw_ptr));
        return megdnn_src_memory;
    }
}

#endif

}  // namespace

size_t PoolingImpl::get_workspace_in_bytes(const TensorLayout& src,
                                           const TensorLayout& dst) {
    if (is_supported(SIMDType::SSE) && src.dtype == dtype::Float32() &&
        param().mode == Mode::MAX && param().format == Param::Format::NCHW &&
        param().window_h == 3 && param().window_w == 3 &&
        param().stride_h == 2 && param().stride_w == 2) {
        WorkspaceBundle ws = get_bundle(src, dst, param());

        return ws.total_size_in_bytes();
    } else {
        return 0;
    }
}

void PoolingImpl::exec(_megdnn_tensor_in src, _megdnn_tensor_out dst,
                       _megdnn_workspace workspace) {
    check_exec(src.layout, dst.layout, workspace.size);
    size_t N = src.layout.shape[0], C = src.layout.shape[1],
           IH = src.layout.shape[2], IW = src.layout.shape[3];
    size_t OH = dst.layout.shape[2], OW = dst.layout.shape[3];

    auto mode = param().mode;
    auto FH = param().window_h, FW = param().window_w;
    auto SH = param().stride_h, SW = param().stride_w;
    auto PH = param().pad_h, PW = param().pad_w;
    bool is_average = (mode == Mode::AVERAGE);
    bool is_include = true;
    if (is_supported(SIMDType::AVX) && is_average &&
        param().format == Param::Format::NCHW &&
        src.layout.dtype == dtype::Float32() && FH == 2 && FW == 2 && SH == 2 &&
        SW == 2) {
        auto sptr = src.ptr<dt_float32>();
        auto dptr = dst.ptr<dt_float32>();
        MEGDNN_DISPATCH_CPU_KERN_OPR(rep(n, N) rep(c, C) {
            mean_pooling_w2x2_s2x2_avx(sptr + n * C * IH * IW + c * IH * IW, IH,
                                       IW, dptr + n * C * OH * OW + c * OH * OW,
                                       OH, OW, PH, PW, is_include);
        });
        return;
    }
    if (is_supported(SIMDType::SSE3) && is_average &&
        src.layout.dtype == dtype::Float32() &&
        param().format == Param::Format::NCHW && FH == 2 && FW == 2 &&
        SH == 2 && SW == 2) {
        auto sptr = src.ptr<dt_float32>();
        auto dptr = dst.ptr<dt_float32>();
        MEGDNN_DISPATCH_CPU_KERN_OPR(rep(n, N) rep(c, C) {
            mean_pooling_w2x2_s2x2_sse3(sptr + n * C * IH * IW + c * IH * IW,
                                        IH, IW,
                                        dptr + n * C * OH * OW + c * OH * OW,
                                        OH, OW, PH, PW, is_include);
        });
        return;
    }
    if (is_supported(SIMDType::SSE) && src.layout.dtype == dtype::Float32() &&
        mode == Mode::MAX && param().format == Param::Format::NCHW && FH == 2 &&
        FW == 2 && SH == 2 && SW == 2) {
        auto sptr = src.ptr<dt_float32>();
        auto dptr = dst.ptr<dt_float32>();
        MEGDNN_DISPATCH_CPU_KERN_OPR(rep(n, N) rep(c, C) {
            max_pooling_w2x2_s2x2_sse(sptr + n * C * IH * IW + c * IH * IW, IH,
                                      IW, dptr + n * C * OH * OW + c * OH * OW,
                                      OH, OW, PH, PW);
        });
        return;
    }
    if (is_supported(SIMDType::SSE) && src.layout.dtype == dtype::Float32() &&
        mode == Mode::MAX && param().format == Param::Format::NCHW && FH == 3 &&
        FW == 3 && SH == 2 && SW == 2) {
        auto sptr = src.ptr<dt_float32>();
        auto dptr = dst.ptr<dt_float32>();
        MEGDNN_DISPATCH_CPU_KERN_OPR(

                WorkspaceBundle ws =
                        get_bundle(src.layout, dst.layout, param());
                ws.set(workspace.raw_ptr); rep(n, N) rep(c, C) {
                    do_max_pooling_3x3_s2x2_float_SSE(
                            sptr + n * C * IH * IW + c * IH * IW,
                            dptr + n * C * OH * OW + c * OH * OW, IH, IW, OH,
                            OW, PH, PW, ws);
                });
        return;
    }

#if MEGDNN_X86_WITH_MKL_DNN

    // Mkldnn provide optimized code for nhwc int8 pooling now.
    // Mkldnn can not change the layout automatic.
    // Reorder nchw input to nhwc, do pooling, reorder nhwc result to nchw
    if ((src.layout.dtype.enumv() == DTypeEnum::QuantizedS8 ||
         src.layout.dtype.enumv() == DTypeEnum::Int8) &&
        mode == Mode::MAX && param().format == Param::Format::NCHW) {
        auto x86_handle = static_cast<HandleImpl*>(inplace_cpu_handle().get());

        auto mkldnn_eng = x86_handle->mkldnn_engine();
        auto mkldnn_stream = x86_handle->mkldnn_stream();
        auto mkldnn_pooling_mode = dnnl::algorithm::pooling_max;
        dnnl::memory::dims pool_strides = {SH, SW};
        dnnl::memory::dims pool_padding = {PH, PW};
        dnnl::memory::dims pool_kernel = {FH, FW};

        dnnl::memory&& megdnn_src_memory_ori =
                tensor_to_mkl_memory<dnnl::memory::format_tag::nchw, false>(
                        src, mkldnn_eng, dnnl::memory::data_type::s8);
        dnnl::memory&& megdnn_dst_memory_ori =
                tensor_to_mkl_memory<dnnl::memory::format_tag::nchw, false>(
                        dst, mkldnn_eng, dnnl::memory::data_type::s8);

        dnnl::memory&& megdnn_src_memory =
                tensor_to_mkl_memory<dnnl::memory::format_tag::nhwc, true>(
                        src, mkldnn_eng, dnnl::memory::data_type::s8);
        dnnl::memory&& megdnn_dst_memory =
                tensor_to_mkl_memory<dnnl::memory::format_tag::nhwc, true>(
                        dst, mkldnn_eng, dnnl::memory::data_type::s8);

        auto reorder_src =
                dnnl::reorder(megdnn_src_memory_ori, megdnn_src_memory);
        auto reorder_dst =
                dnnl::reorder(megdnn_dst_memory, megdnn_dst_memory_ori);
        auto pool1_desc = dnnl::pooling_forward::desc(
                dnnl::prop_kind::forward_inference, mkldnn_pooling_mode,
                megdnn_src_memory.get_desc(), megdnn_dst_memory.get_desc(),
                pool_strides, pool_kernel, pool_padding, pool_padding);
        auto pool_pd =
                dnnl::pooling_forward::primitive_desc(pool1_desc, mkldnn_eng);
        auto pool = dnnl::pooling_forward(pool_pd);

        auto run = [mkldnn_stream, mkldnn_eng, reorder_src, pool, reorder_dst,
                    megdnn_src_memory_ori, megdnn_src_memory, megdnn_dst_memory,
                    megdnn_dst_memory_ori](void) {
            MEGDNN_MARK_USED_VAR(mkldnn_eng);
            auto mkl_stream = mkldnn_stream;
            reorder_src.execute(mkl_stream,
                                {{DNNL_ARG_FROM, megdnn_src_memory_ori},
                                 {DNNL_ARG_TO, megdnn_src_memory}});
            pool.execute(mkl_stream, {{DNNL_ARG_SRC, megdnn_src_memory},
                                      {DNNL_ARG_DST, megdnn_dst_memory}});
            reorder_dst.execute(mkl_stream,
                                {{DNNL_ARG_FROM, megdnn_dst_memory},
                                 {DNNL_ARG_TO, megdnn_dst_memory_ori}});
            mkl_stream.wait();
        };
        MEGDNN_DISPATCH_CPU_KERN_OPR(run());
        return;
    }

    if (src.layout.dtype == dtype::Float32() && mode == Mode::MAX &&
        param().format == Param::Format::NCHW88) {
        auto x86_handle = static_cast<HandleImpl*>(inplace_cpu_handle().get());
        auto mkldnn_eng = x86_handle->mkldnn_engine();
        auto mkldnn_stream = x86_handle->mkldnn_stream();
        auto mkldnn_pooling_mode = dnnl::algorithm::pooling_max;
        switch (mode) {
            case Mode::MAX:
                mkldnn_pooling_mode = dnnl::algorithm::pooling_max;
                break;
            case Mode::AVERAGE:
                mkldnn_pooling_mode =
                        dnnl::algorithm::pooling_avg_include_padding;
                break;
            case Mode::AVERAGE_COUNT_EXCLUDE_PADDING:
                mkldnn_pooling_mode =
                        dnnl::algorithm::pooling_avg_exclude_padding;
                break;
            default:
                megdnn_assert(0, "not supported pooling mode\n");
        };

        dnnl::memory::dims pool_strides = {SH, SW};
        dnnl::memory::dims pool_padding = {PH, PW};
        dnnl::memory::dims pool_kernel = {FH, FW};
        dnnl::memory&& megdnn_src_memory_ori =
                tensor_to_mkl_memory<dnnl::memory::format_tag::nChw8c, false>(
                        src, mkldnn_eng, dnnl::memory::data_type::f32);
        dnnl::memory&& megdnn_dst_memory_ori =
                tensor_to_mkl_memory<dnnl::memory::format_tag::nChw8c, false>(
                        dst, mkldnn_eng, dnnl::memory::data_type::f32);
        auto pool_desc = dnnl::pooling_forward::desc(
                dnnl::prop_kind::forward_inference, mkldnn_pooling_mode,
                megdnn_src_memory_ori.get_desc(),
                megdnn_dst_memory_ori.get_desc(), pool_strides, pool_kernel,
                pool_padding, pool_padding);
        auto pool_pd =
                dnnl::pooling_forward::primitive_desc(pool_desc, mkldnn_eng);
        auto pool = dnnl::pooling_forward(pool_pd);

        auto run = [mkldnn_stream, pool, mkldnn_eng, megdnn_src_memory_ori,
                    megdnn_dst_memory_ori](void) {
            MEGDNN_MARK_USED_VAR(mkldnn_eng);
            auto mkl_stream = mkldnn_stream;

            pool.execute(mkl_stream, {{DNNL_ARG_SRC, megdnn_src_memory_ori},
                                      {DNNL_ARG_DST, megdnn_dst_memory_ori}});
            mkl_stream.wait();
        };
        MEGDNN_DISPATCH_CPU_KERN_OPR(run());
        return;
    }
#endif

    fallback::PoolingImpl::exec(src, dst, Workspace());
}

// vim: syntax=cpp.doxygen
