/**
 * \file dnn/src/fallback/convolution/algos.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#pragma once

#include "src/fallback/conv_bias/algos.h"
#include "src/fallback/convolution/opr_impl.h"
#include "src/naive/convolution/helper.h"
#include "src/common/algo_chooser.h"

namespace megdnn {
namespace fallback {

template <typename ST, typename DT, typename CT>
void kern_naive_forward(const ConvolutionImpl::NCBKernParam& p,
                        const ConvolutionImpl::NCBKernIndex& ncb_index) {
    size_t batch_id = ncb_index.ndrange_id[1];
    size_t group_id = ncb_index.ndrange_id[0];
    auto IC = p.filter_meta.icpg, IH = p.isz[0], IW = p.isz[1],
         OC = p.filter_meta.ocpg, OH = p.osz[0], OW = p.osz[1];
    ptrdiff_t fstrd = p.filter_meta.icpg * p.filter_meta.ocpg *
                      p.filter_meta.spatial[0] * p.filter_meta.spatial[1] *
                      p.filter_type.size();
    ptrdiff_t istrd = p.filter_meta.icpg * p.src_type.size();
    ptrdiff_t ostrd = p.filter_meta.ocpg * p.dst_type.size();
    TensorND src, dst;

    src.layout.dtype = p.src_type;
    dst.layout.dtype = p.dst_type;
    if (p.filter_meta.format == param::Convolution::Format::NCHW) {
        istrd *= p.isz[0] * p.isz[1];
        ostrd *= p.osz[0] * p.osz[1];
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
    src.raw_ptr = reinterpret_cast<void*>(
            reinterpret_cast<uintptr_t>(p.src_ptr) +
            batch_id * p.inp_bs * p.src_type.size() + group_id * istrd);
    dst.raw_ptr = reinterpret_cast<void*>(
            reinterpret_cast<uintptr_t>(p.dst_ptr) +
            batch_id * p.out_bs * p.dst_type.size() + group_id * ostrd);
    ST* filter = reinterpret_cast<ST*>(
            reinterpret_cast<uintptr_t>(p.filter_ptr) + group_id * fstrd);
    std::copy(p.inp_s, p.inp_s + 4, src.layout.stride);
    std::copy(p.out_s, p.out_s + 4, dst.layout.stride);
    naive::convolution::forward<ST, ST, DT, CT>(src, filter, dst,
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
    bool usable(const NCBKernSizeParam& param,
                AlgoSelectionStrategy algo_selection_strategy) const override;

    size_t get_workspace(const NCBKernSizeParam& param) const override;

    SmallVector<NCBKern> dispatch_kern(
            const NCBKernSizeParam& /*param*/) const override;

    ConvAlgoTypePack get_algo_type() const override {
        return {AlgoDataType::FLOAT32, AlgoCategory::NAIVE};
    }
    MEGDNN_DECL_ALGO_TYPE(FB_ALGO)
};

class ConvolutionImpl::AlgoNaive final : public AlgoBase {
public:
    bool is_reproducible() const override { return true; }
    const char* name() const override { return "NAIVE_ALGO"; }
    bool usable(const NCBKernSizeParam& /*param*/,
                AlgoSelectionStrategy algo_selection_strategy) const override;

    size_t get_workspace(const NCBKernSizeParam&) const override { return 0; };

    SmallVector<NCBKern> dispatch_kern(
            const NCBKernSizeParam& /*param*/) const override;

    ConvAlgoTypePack get_algo_type() const override {
        auto support_data_type = static_cast<AlgoDataType>(
                static_cast<uint32_t>(AlgoDataType::INT8X8X16) |
                static_cast<uint32_t>(AlgoDataType::QINT8X8X32) |
                static_cast<uint32_t>(AlgoDataType::QUINT8X8X32));
        return {support_data_type, AlgoCategory::NAIVE};
    }
    MEGDNN_DECL_ALGO_TYPE(FB_NAIVE)
};

class ConvolutionImpl::AlgoDefault final : public AlgoBase {
    WorkspaceBundle get_bundle(const NCBKernSizeParam& param) const;
    static SmallVector<NCBKern> get_kimpl(ConvBiasImpl::AlgoBase* algo,
                                          const NCBKernSizeParam& param);
    static SmallVector<NCBKern> get_preprocess_kimpl(
            ConvBiasImpl::AlgoBase* algo, const NCBKernSizeParam& param);

public:
    AlgoDefault(ConvBiasImpl::AlgoBase*);
    bool is_reproducible() const override { return true; }
    const char* name() const override { return m_name.c_str(); }
    bool usable(const NCBKernSizeParam& param,
                AlgoSelectionStrategy algo_selection_strategy) const override;

    size_t get_workspace(const NCBKernSizeParam& param) const override;

    size_t get_preprocess_workspace(const NCBKernSizeParam&) const override;

    SmallVector<TensorLayout> deduce_preprocessed_filter_layout(
            const NCBKernSizeParam&) const override;

    SmallVector<NCBKern> dispatch_preprocess_kern(
            const NCBKernSizeParam& param) const override {
        return get_preprocess_kimpl(m_algorithm, param);
    }

    SmallVector<NCBKern> dispatch_kern(
            const NCBKernSizeParam& param) const override {
        return get_kimpl(m_algorithm, param);
    }

    //! select matmul to the highest preference
    bool is_preferred(const NCBKernSizeParam& param) const override;

    std::string param() const override {
        std::string ret;
        serialize_write_pod(m_algorithm, ret);
        return ret;
    }

    static ConvBiasImpl::NCBKernSizeParam init_conv_bias_param(
            const NCBKernSizeParam& param);

    ConvAlgoTypePack get_algo_type() const override {
        return m_algorithm->get_algo_type();
    }
    MEGDNN_DECL_ALGO_TYPE(FB_DEFAULT)

private:
    std::string m_name;
    ConvBiasImpl::AlgoBase* m_algorithm;
};

////////////////////////// convolutionbackwarddata ////////////////////////
class ConvolutionBackwardDataImpl::AlgoNaive final : public AlgoBase {
public:
    bool is_reproducible() const override { return true; }
    const char* name() const override { return "DeconvNaive"; }
    bool usable(ConvolutionBackwardDataImpl* opr,
                const NCBKernSizeParam& param) const override;
    size_t get_workspace(ConvolutionBackwardDataImpl*,
                         const NCBKernSizeParam& param) const override;
    ncb_kern_t dispatch_kern(ConvolutionBackwardDataImpl*,
                             const NCBKernSizeParam&) const override;
    bool is_naive() const override { return true; }
    MEGDNN_DECL_ALGO_TYPE(FB_NAIVE)
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
    MEGDNN_DECL_ALGO_TYPE(FB_DIRECT)
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
    bool is_preferred(const NCBKernSizeParam& param) const override;
    MEGDNN_DECL_ALGO_TYPE(FB_MATMUL)
};

}  // namespace fallback
}  // namespace megdnn

// vim: syntax=cpp.doxygen
