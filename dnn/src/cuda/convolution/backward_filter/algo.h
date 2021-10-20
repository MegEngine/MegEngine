/**
 * \file dnn/src/cuda/convolution/backward_filter/algo.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#pragma once

#include <unordered_map>
#include "src/common/algo_base.h"
#include "src/common/metahelper.h"
#include "src/cuda/convolution/helper.h"

namespace megdnn {
namespace cuda {

/*!
 * \brief base class for convolution algos
 *
 * All the algo impls should try to support non-contiguous batch dim, for group
 * conv execution.
 */
class ConvolutionBackwardFilterImpl::AlgoBase : public Algorithm {
protected:
    ~AlgoBase() = default;

public:
    enum class AlgoType : uint32_t {
        CUDA_CUDNN,
        CUDA_MATMUL,
        CUDA_CHANWISE,
        CUDA_BFLOAT16,
        CUDA_GROUP_CONV_GENERAL,
    };
    using Mapper = std::unordered_map<AlgorithmDesc, AlgoBase*>;

    AlgoBase() : Algorithm() { m_handle_type = Handle::HandleType::CUDA; }
    struct SizeArgs {
        HandleImpl* handle;
        const TensorLayout *src_layout, *diff_layout, *grad_layout;
        CanonizedFilterMeta grad_filter_meta;
        const ConvolutionBackwardFilterImpl* opr;

        std::string to_string() const;
        void init_desc(convolution::CUDNNBwdFilterDescs& desc) const {
            desc.set(*src_layout, *diff_layout, grad_filter_meta, opr->param());
        }
        SizeArgs(
                const ConvolutionBackwardFilterImpl* opr, const TensorLayout& src,
                const TensorLayout& diff, const TensorLayout& grad);
        SizeArgs(
                const ConvolutionBackwardFilterImpl* opr, const TensorLayout& src,
                const TensorLayout& diff, const TensorLayout& grad,
                const CanonizedFilterMeta& grad_meta);

        convolution::ForwardSizeArgs as_fwd_args() const {
            return {handle, src_layout, grad_layout, grad_filter_meta, diff_layout};
        }
    };
    struct ExecArgs : public SizeArgs {
        const TensorND *src_tensor, *diff_tensor, *grad_tensor;
        Workspace workspace;

        ExecArgs(
                const ConvolutionBackwardFilterImpl* opr, _megdnn_tensor_in src,
                _megdnn_tensor_in diff, _megdnn_tensor_out grad,
                _megdnn_workspace workspace);
    };
    virtual bool is_available(const SizeArgs& args) const = 0;
    virtual size_t get_workspace_in_bytes(const SizeArgs& args) const = 0;
    virtual void exec(const ExecArgs& args) const = 0;

    bool is_available_wk(const SizeArgs& args, size_t limit) {
        return is_available(args) && get_workspace_in_bytes(args) <= limit;
    }

    bool is_available_attribute(
            const SizeArgs& args,
            const AlgoAttribute& positive_attr = AlgoAttribute::REPRODUCIBLE,
            const AlgoAttribute& negative_attr = AlgoAttribute::DEFAULT,
            size_t limit = std::numeric_limits<size_t>::max()) {
        return contain_attribute_all(positive_attr) &&
               !contain_attribute_any(negative_attr) && is_available_wk(args, limit);
    }

    AlgoBase& check_workspace(const SizeArgs& args, const Workspace& workspace) {
        auto req = get_workspace_in_bytes(args);
        megdnn_assert(
                req <= workspace.size,
                "conv bwd filter algo %s: "
                "required workspace %zu bytes, got %zu",
                name(), req, workspace.size);
        return *this;
    }

    virtual bool is_cudnn() const { return false; }
};

class ConvolutionBackwardFilterImpl::AlgoCUDNN final : public AlgoBase {
    cudnnConvolutionBwdFilterAlgo_t m_cudnn_enum;
    CudnnAlgoPack::Attr m_attr;

public:
    AlgoCUDNN(cudnnConvolutionBwdFilterAlgo_t cudnn_enum) : m_cudnn_enum(cudnn_enum) {
        megdnn_assert(
                CudnnAlgoPack::conv_bwd_flt_algos().find(cudnn_enum) !=
                CudnnAlgoPack::conv_bwd_flt_algos().end());
        m_attr = CudnnAlgoPack::conv_bwd_flt_algos().at(cudnn_enum);
    }

    bool is_available(const SizeArgs& args) const override;
    size_t get_workspace_in_bytes(const SizeArgs& args) const override;
    void exec(const ExecArgs& args) const override;

    const char* name() const override { return m_attr.name.c_str(); }
    AlgoAttribute attribute() const override {
        auto ret = static_cast<AlgoAttribute>(0);
        if (m_attr.is_reproducible) {
            ret |= AlgoAttribute::REPRODUCIBLE;
        }
        if (m_attr.accuracy_depend_on_batch) {
            ret |= AlgoAttribute::ACCURACY_DEPEND_ON_BATCH;
        }
        return ret;
    }

    cudnnConvolutionBwdFilterAlgo_t cudnn_enum() const { return m_cudnn_enum; }

    bool is_cudnn() const override { return true; }

    MEGDNN_DECL_ALGO_TYPE(CUDA_CUDNN)
    std::string param() const override {
        std::string ret;
        serialize_write_pod(m_cudnn_enum, ret);
        return ret;
    }
};

//! im2col and matmul, with dilation
class ConvolutionBackwardFilterImpl::AlgoMatmul final : public AlgoBase {
    template <typename T>
    static void exec_internal(const ExecArgs& args);

public:
    bool is_available(const SizeArgs& args) const override;
    size_t get_workspace_in_bytes(const SizeArgs& args) const override;
    void exec(const ExecArgs& args) const override;

    std::vector<SearchItem> get_subopr_list(
            const TensorLayoutArray& layouts, const OperatorBase* opr) const override;

    const char* name() const override { return "MATMUL"; }
    MEGDNN_DECL_ALGO_TYPE(CUDA_MATMUL)
    AlgoAttribute attribute() const override {
        return AlgoAttribute::REPRODUCIBLE | AlgoAttribute::ACCURACY_DEPEND_ON_BATCH;
    }
};

class ConvolutionBackwardFilterImpl::AlgoChanwise final : public AlgoBase {
public:
    bool is_available(const SizeArgs& args) const override;
    size_t get_workspace_in_bytes(const SizeArgs& args) const override;
    void exec(const ExecArgs& args) const override;

    const char* name() const override { return "CHANNEL_WISE"; }
    MEGDNN_DECL_ALGO_TYPE(CUDA_CHANWISE)
    AlgoAttribute attribute() const override { return AlgoAttribute::REPRODUCIBLE; }
};

class ConvolutionBackwardFilterImpl::AlgoBFloat16 final : public AlgoBase {
public:
    bool is_available(const SizeArgs& args) const override;
    size_t get_workspace_in_bytes(const SizeArgs& args) const override;
    void exec(const ExecArgs& args) const override;

    std::vector<SearchItem> get_subopr_list(
            const TensorLayoutArray& layouts, const OperatorBase* opr) const override;

    const char* name() const override { return "CONVOLUTION_BACKWARD_FILTER_BFLOAT16"; }

    AlgoAttribute attribute() const override { return AlgoAttribute::REPRODUCIBLE; }

    MEGDNN_DECL_ALGO_TYPE(CUDA_BFLOAT16)

private:
    WorkspaceBundle get_workspace_bundle(void* ptr, const SizeArgs& args) const;
};

//! implement group conv by another algo
class ConvolutionBackwardFilterImpl::AlgoGroupConvGeneral final : public AlgoBase {
public:
    bool is_available(const SizeArgs& args) const override;
    size_t get_workspace_in_bytes(const SizeArgs& args) const override;
    void exec(const ExecArgs& args) const override;
    std::vector<SearchItem> get_subopr_list(
            const TensorLayoutArray& layouts, const OperatorBase* opr) const override;

    const char* name() const override { return "CUDA:GROUP_CONV_BACKWARD_FILTER"; }

    MEGDNN_DECL_ALGO_TYPE(CUDA_GROUP_CONV_GENERAL)
    AlgoAttribute attribute() const override { return AlgoAttribute::REPRODUCIBLE; }

private:
    WorkspaceBundle get_workspace_bundle(void* ptr, const SizeArgs& args) const;
};

class ConvolutionBackwardFilterImpl::AlgoPack : NonCopyableObj {
    // defined in cudnn.cpp
    void fill_cudnn_algos();

    AlgoBase::Mapper m_all_algos_map;

public:
    AlgoPack();

    std::vector<AlgoCUDNN> cudnn;
    AlgoMatmul matmul;
    AlgoChanwise chanwise;
    AlgoGroupConvGeneral group;
    AlgoBFloat16 bfloat16;

    std::vector<AlgoBase*>
            //! all algorithms
            all_algos,
            //! non-cudnn algos, used for heuristic if cudnn is not supported
            non_cudnn_algos, bfloat16_algos;

    AlgoCUDNN* cudnn_from_enum(cudnnConvolutionBwdFilterAlgo_t algo);

    const AlgoBase::Mapper& all_algos_map() const { return m_all_algos_map; }
};

}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen
