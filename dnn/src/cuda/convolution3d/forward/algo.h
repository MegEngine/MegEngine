/**
 * \file dnn/src/cuda/convolution3d/forward/algo.h
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

#include "megdnn/oprs.h"

#include "src/common/utils.h"
#include "src/cuda/convolution3d/helper.h"
#include "src/cuda/convolution3d/opr_impl.h"
#include "src/cuda/handle.h"
#include "src/common/algo_base.h"
#include "src/common/metahelper.h"

#include <unordered_map>

namespace megdnn {
namespace cuda {

/*!
 * \brief base class for convolution3d algos
 *
 * All the algo impls should try to support non-contiguous batch dim, for group
 * conv execution.
 */
class Convolution3DForwardImpl::AlgoBase : public Algorithm {
protected:
    ~AlgoBase() = default;

public:
    enum class AlgoType : uint32_t {
        CUDA_1X1X1,
        CUDA_GROUP_CONV_GENERAL,
        CUDA_CUDNN,
        CUDA_INPLACE_MATMUL,
        CUDA_CHANWISE,
    };
    using Mapper = std::unordered_map<AlgorithmDesc, AlgoBase*>;

    AlgoBase() : Algorithm() { m_handle_type = Handle::HandleType::CUDA; }
    struct SizeArgs : public convolution3d::ForwardSizeArgs {
        Convolution3DForwardImpl* opr;

        std::string to_string() const;
        void init_desc(convolution3d::CUDNNForwardDescs& desc) const {
            desc.set(*src_layout, filter_meta, *dst_layout, opr->param());
        }
        SizeArgs(Convolution3DForwardImpl* opr, const TensorLayout& src,
                 const TensorLayout& filter, const TensorLayout& dst);
        SizeArgs(Convolution3DForwardImpl* opr, const TensorLayout& src,
                 const CanonizedFilterMeta& filter, const TensorLayout& dst);
    };
    struct ExecArgs : public SizeArgs {
        const TensorND *src_tensor, *filter_tensor, *dst_tensor;
        Workspace workspace;

        ExecArgs(Convolution3DForwardImpl* opr, _megdnn_tensor_in src,
                 _megdnn_tensor_in filter, _megdnn_tensor_out dst,
                 _megdnn_workspace workspace);
    };
    virtual bool is_available(const SizeArgs& args) const = 0;
    virtual size_t get_workspace_in_bytes(const SizeArgs& args) const = 0;
    virtual void exec(const ExecArgs& args) const = 0;

    bool is_available_wk(const SizeArgs& args, size_t limit) {
        return is_available(args) && get_workspace_in_bytes(args) <= limit;
    }
    bool is_available_reproducible(
            const SizeArgs& args, bool reproducible = true,
            size_t limit = std::numeric_limits<size_t>::max()) {
        return (!reproducible || is_reproducible()) &&
               is_available_wk(args, limit);
    }
    AlgoBase& check_workspace(const SizeArgs& args,
                              const Workspace& workspace) {
        auto req = get_workspace_in_bytes(args);
        megdnn_assert(
                req <= workspace.size,
                "conv3d fwd algo %s: required workspace %zu bytes, got %zu",
                name(), req, workspace.size);
        return *this;
    }

    virtual bool is_cudnn() const { return false; }
};
class Convolution3DForwardImpl::Algo1x1x1 final : public AlgoBase {
    static void extract_matmul_layouts(const SizeArgs& args, TensorLayout& A,
                                       TensorLayout& B, TensorLayout& C);

public:
    bool is_available(const SizeArgs& args) const override;
    size_t get_workspace_in_bytes(const SizeArgs& args) const override;
    void exec(const ExecArgs& args) const override;

    const char* name() const override { return "1x1x1"; }
    bool is_reproducible() const override { return true; }
    MEGDNN_DECL_ALGO_TYPE(CUDA_1X1X1)
};

//! implement group conv by another algo
class Convolution3DForwardImpl::AlgoGroupConvGeneral final : public AlgoBase {
    AlgoBase* m_impl;
    std::string m_name;

public:
    AlgoGroupConvGeneral(AlgoBase* impl);

    bool is_available(const SizeArgs& args) const override;
    size_t get_workspace_in_bytes(const SizeArgs& args) const override;
    void exec(const ExecArgs& args) const override;

    const char* name() const override { return m_name.c_str(); }

    bool is_reproducible() const override { return m_impl->is_reproducible(); }

    static void modify_size_args(SizeArgs& args, TensorLayout& src_pg,
                                 TensorLayout& dst_pg);
    MEGDNN_DECL_ALGO_TYPE(CUDA_GROUP_CONV_GENERAL)
    std::string param() const override {
        std::string ret;
        serialize_write_pod(m_impl, ret);
        return ret;
    }
};

class Convolution3DForwardImpl::AlgoCUDNN final : public AlgoBase {
    cudnnConvolutionFwdAlgo_t m_cudnn_enum;
    CudnnAlgoPack::Attr m_attr;

public:
    AlgoCUDNN(cudnnConvolutionFwdAlgo_t cudnn_enum) : m_cudnn_enum(cudnn_enum) {
        megdnn_assert(CudnnAlgoPack::conv3d_fwd_algos().find(cudnn_enum) !=
                      CudnnAlgoPack::conv3d_fwd_algos().end());
        m_attr = CudnnAlgoPack::conv3d_fwd_algos().at(cudnn_enum);
    }

    bool is_available(const SizeArgs& args) const override;
    size_t get_workspace_in_bytes(const SizeArgs& args) const override;
    void exec(const ExecArgs& args) const override;

    bool is_reproducible() const override { return m_attr.is_reproducible; }

    const char* name() const override { return m_attr.name.c_str(); }

    cudnnConvolutionFwdAlgo_t cudnn_enum() const { return m_cudnn_enum; }

    bool is_cudnn() const override { return true; }

    MEGDNN_DECL_ALGO_TYPE(CUDA_CUDNN)

    std::string param() const override {
        std::string ret;
        serialize_write_pod(m_cudnn_enum, ret);
        return ret;
    }

};

class Convolution3DForwardImpl::AlgoInplaceMatmul final : public AlgoBase {
public:
    bool is_available(const SizeArgs& args) const override;
    size_t get_workspace_in_bytes(const SizeArgs& args) const override;
    void exec(const ExecArgs& args) const override;

    const char* name() const override { return "INPLACE_MATMUL"; }
    bool is_reproducible() const override { return true; }
    MEGDNN_DECL_ALGO_TYPE(CUDA_INPLACE_MATMUL)
};

class Convolution3DForwardImpl::AlgoChanwise final : public AlgoBase {
public:
    bool is_available(const SizeArgs& args) const override;
    size_t get_workspace_in_bytes(const SizeArgs& args) const override;
    void exec(const ExecArgs& args) const override;

    const char* name() const override { return "CHANNEL_WISE"; }
    bool is_reproducible() const override { return true; }
    MEGDNN_DECL_ALGO_TYPE(CUDA_CHANWISE)
};

class Convolution3DForwardImpl::AlgoPack : NonCopyableObj {
    // defined in cudnn.cpp
    void fill_cudnn_algos();

    AlgoBase::Mapper m_all_algos_map;

public:
    AlgoPack();

    std::vector<AlgoCUDNN> cudnn;
    Algo1x1x1 a1x1x1;
    AlgoInplaceMatmul inplace_matmul;
    AlgoChanwise chanwise;
    std::vector<AlgoGroupConvGeneral> gconv;
    std::unordered_map<AlgoBase*, AlgoGroupConvGeneral*> algo2gconv;

    std::vector<AlgoBase*>
            //! all algorithms
            all_algos,
            //! non-cudnn algos, used for heuristic if cudnn is not supported
            non_cudnn_algos;

    AlgoCUDNN* cudnn_from_enum(cudnnConvolutionFwdAlgo_t algo);

    const AlgoBase::Mapper& all_algos_map() const { return m_all_algos_map; }
};

}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen
