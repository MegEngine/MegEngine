/**
 * \file dnn/src/fallback/convolution/algos.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "src/fallback/conv_bias/algos.h"
#include "src/fallback/convolution/opr_impl.h"
#include "src/naive/convolution/helper.h"

namespace megdnn {
namespace fallback {

template <typename ST, typename DT, typename CT>
void kern_naive_forward(const ConvolutionImpl::NCBKernParam& p,
                        const ConvolutionImpl::NCBKernIndex& /*index*/) {
    auto IC = p.filter_meta.icpg, IH = p.isz[0], IW = p.isz[1],
         OC = p.filter_meta.ocpg, OH = p.osz[0], OW = p.osz[1];
    TensorND src, dst;
    src.raw_ptr = const_cast<void*>(p.src_ptr);
    dst.raw_ptr = p.dst_ptr;

    src.layout.dtype = p.src_type;
    dst.layout.dtype = p.dst_type;
    if (p.filter_meta.format == param::Convolution::Format::NCHW) {
        src.layout.init_contiguous_stride({1, IC, IH, IW});
        dst.layout.init_contiguous_stride({1, OC, OH, OW});
    } else {
        // Must be NHWC
        megdnn_assert(
                p.filter_meta.format == param::Convolution::Format::NHWC,
                "AlgoNaive only support NCHW and NHWC, not support format %d",
                static_cast<int>(p.filter_meta.format));
        src.layout.init_contiguous_stride({1, IH, IW, IC});
        dst.layout.init_contiguous_stride({1, OH, OW, OC});
    }
    std::copy(p.inp_s, p.inp_s + 4, src.layout.stride);
    std::copy(p.out_s, p.out_s + 4, dst.layout.stride);
    naive::convolution::forward<ST, ST, DT, CT>(src, p.filter<ST>(), dst,
                                                p.filter_meta);
}

template <typename ftype, typename dtype, typename gtype>
void kern_naive(const ConvolutionBackwardDataImpl::NCBKernParam& p) {
    TensorND diff(const_cast<void*>(p.diff_ptr), p.diff_layout),
            filter(const_cast<void*>(p.filter_ptr), p.filter_layout),
            grad(p.grad_ptr, p.grad_layout);
    naive::convolution::backward_data<ftype, dtype, gtype>(filter, diff, grad,
                                                           p.filter_meta);
}

class ConvolutionImpl::AlgoFallback final : public AlgoBase {
public:
    bool is_reproducible() const override { return true; }
    const char* name() const override { return "FALLBACK_ALGO"; }
    bool usable(ConvolutionImpl* opr, const NCBKernSizeParam& param,
                AlgoSelectionStrategy algo_selection_strategy) const override;

    size_t get_workspace(ConvolutionImpl* opr,
                         const NCBKernSizeParam& param) const override;

    SmallVector<NCBKern> dispatch_kern(
            ConvolutionImpl* /*opr*/,
            const NCBKernSizeParam& /*param*/) const override;
};

class ConvolutionImpl::AlgoNaive final : public AlgoBase {
public:
    bool is_reproducible() const override { return true; }
    const char* name() const override { return "NAIVE_ALGO"; }
    bool usable(ConvolutionImpl* /*opr*/, const NCBKernSizeParam& /*param*/,
                AlgoSelectionStrategy algo_selection_strategy) const override;

    size_t get_workspace(ConvolutionImpl*,
                         const NCBKernSizeParam&) const override {
        return 0;
    };

    SmallVector<NCBKern> dispatch_kern(
            ConvolutionImpl* /*opr*/,
            const NCBKernSizeParam& /*param*/) const override;
};

class ConvolutionImpl::AlgoDefault final : public AlgoBase {
    static ConvBiasImpl::NCBKernSizeParam init_convbias_opr_and_param(
            ConvBiasImpl* conv_bias_opr, const NCBKernSizeParam& param);
    WorkspaceBundle get_bundle(const NCBKernSizeParam& param) const;
    static SmallVector<NCBKern> get_kimpl(ConvBiasImpl* conv_bias_opr,
                                          ConvBiasImpl::AlgoBase* algo,
                                          const NCBKernSizeParam& param);

public:
    AlgoDefault(fallback::ConvBiasImpl* conv_bias_opr, ConvBiasImpl::AlgoBase*);
    bool is_reproducible() const override { return true; }
    const char* name() const override { return m_name.c_str(); }
    bool usable(ConvolutionImpl* opr, const NCBKernSizeParam& param,
                AlgoSelectionStrategy algo_selection_strategy) const override;

    size_t get_workspace(ConvolutionImpl* opr,
                         const NCBKernSizeParam& param) const override;

    SmallVector<NCBKern> dispatch_kern(
            ConvolutionImpl* /*opr*/,
            const NCBKernSizeParam& param) const override {
        return get_kimpl(m_conv_bias_opr, m_algorithm, param);
    }

    void* type() const override { return sm_fallback_conv_algo_type; }

    //! select matmul to the highest preference
    bool is_preferred(ConvolutionImpl* opr,
                      const NCBKernSizeParam& param) const override;

private:
    std::string m_name;
    fallback::ConvBiasImpl* m_conv_bias_opr;
    ConvBiasImpl::AlgoBase* m_algorithm;
};

class ConvolutionBackwardDataImpl::AlgoDirect final : public AlgoBase {
public:
    bool is_reproducible() const override { return true; }
    const char* name() const override { return "DeconvDirect"; }
    bool usable(ConvolutionBackwardDataImpl* opr,
                const NCBKernSizeParam& param) const override;
    size_t get_workspace(ConvolutionBackwardDataImpl*,
                         const NCBKernSizeParam& param) const override;
    ncb_kern_t dispatch_kern(ConvolutionBackwardDataImpl*,
                             const NCBKernSizeParam&) const override;
    void* type() const override { return sm_fallback_deconv_algo_type; }
};

class ConvolutionBackwardDataImpl::AlgoMatrixMul final : public AlgoBase {
public:
    bool is_reproducible() const override { return true; }
    const char* name() const override { return "DeconvMatmul"; }
    bool usable(ConvolutionBackwardDataImpl* opr,
                const NCBKernSizeParam& param) const override;
    size_t get_workspace(ConvolutionBackwardDataImpl*,
                         const NCBKernSizeParam& param) const override;
    ncb_kern_t dispatch_kern(ConvolutionBackwardDataImpl*,
                             const NCBKernSizeParam&) const override;
    void* type() const override { return sm_fallback_deconv_algo_type; }
};

}  // namespace fallback
}  // namespace megdnn

// vim: syntax=cpp.doxygen
