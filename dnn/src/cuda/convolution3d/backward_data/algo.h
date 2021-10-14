/**
 * \file dnn/src/cuda/convolution3d/backward_data/algo.h
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
#include "src/cuda/convolution3d/helper.h"

namespace megdnn {
namespace cuda {

/*!
 * \brief base class for convolution3d algos
 *
 * All the algo impls should try to support non-contiguous batch dim, for group
 * conv execution.
 */
class Convolution3DBackwardDataImpl::AlgoBase : public Algorithm {
protected:
    ~AlgoBase() = default;

public:
    enum class AlgoType : uint32_t {
        CUDA_GROUP_CONV_GENERAL,
        CUDA_CUDNN,
        CUDA_CHANWISE,
    };
    using Mapper = std::unordered_map<AlgorithmDesc, AlgoBase*>;

    AlgoBase() : Algorithm() { m_handle_type = Handle::HandleType::CUDA; }
    struct SizeArgs {
        HandleImpl* handle;
        CanonizedFilterMeta filter_meta;
        const TensorLayout *diff_layout, *grad_layout, *filter_layout;
        const Convolution3DBackwardDataImpl* opr;

        std::string to_string() const;
        void init_desc(convolution3d::CUDNNBwdDataDescs& desc) const {
            desc.set(filter_meta, *diff_layout, *grad_layout, opr->param());
        }
        SizeArgs(
                const Convolution3DBackwardDataImpl* opr, const TensorLayout& filter,
                const TensorLayout& diff, const TensorLayout& grad);
        SizeArgs(
                const Convolution3DBackwardDataImpl* opr, const TensorLayout& filter,
                const CanonizedFilterMeta& filter_meta, const TensorLayout& diff,
                const TensorLayout& grad);

        convolution3d::ForwardSizeArgs as_fwd_args() const {
            return {handle,      grad_layout, filter_layout,
                    filter_meta, diff_layout, opr->param().data_type};
        }
    };
    struct ExecArgs : public SizeArgs {
        const TensorND *filter_tensor, *diff_tensor, *grad_tensor;
        Workspace workspace;

        ExecArgs(
                const Convolution3DBackwardDataImpl* opr, _megdnn_tensor_in filter,
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
                "conv bwd data algo %s: "
                "required workspace %zu bytes, got %zu",
                name(), req, workspace.size);
        return *this;
    }

    virtual bool is_cudnn() const { return false; }
};

class Convolution3DBackwardDataImpl::AlgoCUDNN final : public AlgoBase {
    cudnnConvolutionBwdDataAlgo_t m_cudnn_enum;
    CudnnAlgoPack::Attr m_attr;

public:
    AlgoCUDNN(cudnnConvolutionBwdDataAlgo_t cudnn_enum) : m_cudnn_enum(cudnn_enum) {
        megdnn_assert(
                CudnnAlgoPack::conv3d_bwd_data_algos().find(cudnn_enum) !=
                CudnnAlgoPack::conv3d_bwd_data_algos().end());
        m_attr = CudnnAlgoPack::conv3d_bwd_data_algos().at(cudnn_enum);
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

    cudnnConvolutionBwdDataAlgo_t cudnn_enum() const { return m_cudnn_enum; }

    bool is_cudnn() const override { return true; }

    MEGDNN_DECL_ALGO_TYPE(CUDA_CUDNN)

    std::string param() const override {
        std::string ret;
        serialize_write_pod(m_cudnn_enum, ret);
        return ret;
    }
};

class Convolution3DBackwardDataImpl::AlgoChanwise final : public AlgoBase {
public:
    bool is_available(const SizeArgs& args) const override;
    size_t get_workspace_in_bytes(const SizeArgs& args) const override;
    void exec(const ExecArgs& args) const override;

    const char* name() const override { return "CHANNEL_WISE"; }
    MEGDNN_DECL_ALGO_TYPE(CUDA_CHANWISE)
    AlgoAttribute attribute() const override { return AlgoAttribute::REPRODUCIBLE; }
};

//! implement group conv by another algo
class Convolution3DBackwardDataImpl::AlgoGroupConvGeneral final : public AlgoBase {
public:
    bool is_available(const SizeArgs& args) const override;
    size_t get_workspace_in_bytes(const SizeArgs& args) const override;
    void exec(const ExecArgs& args) const override;
    std::vector<SearchItem> get_subopr_list(
            const TensorLayoutArray& layouts, const OperatorBase* opr) const override;

    const char* name() const override { return "CUDA:GROUP_CONV3D_BACKWARD_DATA"; }

    AlgoAttribute attribute() const override { return AlgoAttribute::REPRODUCIBLE; }

    MEGDNN_DECL_ALGO_TYPE(CUDA_GROUP_CONV_GENERAL)
private:
    WorkspaceBundle get_workspace_bundle(void* ptr, const SizeArgs& args) const;
};

class Convolution3DBackwardDataImpl::AlgoPack : NonCopyableObj {
    // defined in cudnn.cpp
    void fill_cudnn_algos();

    AlgoBase::Mapper m_all_algos_map;

public:
    AlgoPack();

    std::vector<AlgoCUDNN> cudnn;
    AlgoChanwise chanwise;
    AlgoGroupConvGeneral group;

    std::vector<AlgoBase*>
            //! all algorithms
            all_algos,
            //! non-cudnn algos, used for heuristic if cudnn is not supported
            non_cudnn_algos;

    AlgoCUDNN* cudnn_from_enum(cudnnConvolutionBwdDataAlgo_t algo);

    const AlgoBase::Mapper& all_algos_map() const { return m_all_algos_map; }
};

}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen
