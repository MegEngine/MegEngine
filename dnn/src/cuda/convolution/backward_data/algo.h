/**
 * \file dnn/src/cuda/convolution/backward_data/algo.h
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
#include "src/cuda/cudnn_wrapper.h"

namespace megdnn {
namespace cuda {

/*!
 * \brief base class for convolution algos
 *
 * All the algo impls should try to support non-contiguous batch dim, for group
 * conv execution.
 */
class ConvolutionBackwardDataImpl::AlgoBase : public Algorithm {
protected:
    ~AlgoBase() = default;

public:
    enum class AlgoType : uint32_t {
        CUDA_CUDNN,
        CUDA_MATMUL,
        CUDA_CHANWISE,
        CUDA_CHANWISE_SMALL,
        CUDA_BFLOAT16,
        CUDA_GROUP_CONV_GENERAL,
        CUDA_IMPLICIT_GEMM_NCHW4_DOTPROD_INT8,
        CUDA_IMPLICIT_GEMM_NCHW_DOTPROD_INT8
    };
    using Mapper = std::unordered_map<AlgorithmDesc, AlgoBase*>;

    AlgoBase() : Algorithm() { m_handle_type = Handle::HandleType::CUDA; }
    struct SizeArgs {
        HandleImpl* handle;
        CanonizedFilterMeta filter_meta;
        const TensorLayout *diff_layout, *grad_layout, *filter_layout;
        ConvolutionBackwardDataImpl* opr;

        std::string to_string() const;
        void init_desc(convolution::CUDNNBwdDataDescs& desc) const {
            desc.set(filter_meta, *diff_layout, *grad_layout, opr->param());
        }
        SizeArgs(ConvolutionBackwardDataImpl* opr, const TensorLayout& filter,
                 const TensorLayout& diff, const TensorLayout& grad);
        SizeArgs(ConvolutionBackwardDataImpl* opr, const TensorLayout& filter,
                 const CanonizedFilterMeta& filter_meta,
                 const TensorLayout& diff, const TensorLayout& grad);

        convolution::ForwardSizeArgs as_fwd_args() const {
            return {handle, grad_layout, filter_layout, filter_meta,
                    diff_layout};
        }
    };
    struct ExecArgs : public SizeArgs {
        const TensorND *filter_tensor, *diff_tensor, *grad_tensor;
        Workspace workspace;

        ExecArgs(ConvolutionBackwardDataImpl* opr, _megdnn_tensor_in filter,
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
               !contain_attribute_any(negative_attr) &&
               is_available_wk(args, limit);
    }

    AlgoBase& check_workspace(const SizeArgs& args,
                              const Workspace& workspace) {
        auto req = get_workspace_in_bytes(args);
        megdnn_assert(req <= workspace.size,
                      "conv bwd data algo %s: "
                      "required workspace %zu bytes, got %zu",
                      name(), req, workspace.size);
        return *this;
    }

    virtual bool is_cudnn() const { return false; }
};

class ConvolutionBackwardDataImpl::AlgoCUDNN final : public AlgoBase {
    cudnnConvolutionBwdDataAlgo_t m_cudnn_enum;
    CudnnAlgoPack::Attr m_attr;

public:
    AlgoCUDNN(cudnnConvolutionBwdDataAlgo_t cudnn_enum)
            : m_cudnn_enum(cudnn_enum) {
        megdnn_assert(CudnnAlgoPack::conv_bwd_data_algos().find(cudnn_enum) !=
                      CudnnAlgoPack::conv_bwd_data_algos().end());
        m_attr = CudnnAlgoPack::conv_bwd_data_algos().at(cudnn_enum);
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

//! im2col and matmul, with dilation
class ConvolutionBackwardDataImpl::AlgoMatmul final : public AlgoBase {
    template <typename T>
    static void exec_internal(const ExecArgs& args);

public:
    bool is_available(const SizeArgs& args) const override;
    size_t get_workspace_in_bytes(const SizeArgs& args) const override;
    void exec(const ExecArgs& args) const override;

    std::vector<SearchItem> get_subopr_list(
            const TensorLayoutArray& layouts,
            const OperatorBase* opr) const override;

    const char* name() const override { return "MATMUL"; }
    MEGDNN_DECL_ALGO_TYPE(CUDA_MATMUL)
    AlgoAttribute attribute() const override {
        return AlgoAttribute::REPRODUCIBLE;
    }
};

class ConvolutionBackwardDataImpl::AlgoChanwise final : public AlgoBase {
public:
    bool is_available(const SizeArgs& args) const override;
    size_t get_workspace_in_bytes(const SizeArgs& args) const override;
    void exec(const ExecArgs& args) const override;

    const char* name() const override { return "CHANNEL_WISE"; }
    MEGDNN_DECL_ALGO_TYPE(CUDA_CHANWISE)
    AlgoAttribute attribute() const override {
        return AlgoAttribute::REPRODUCIBLE;
    }
};

class ConvolutionBackwardDataImpl::AlgoChanwiseSmall final : public AlgoBase {
public:
    bool is_available(const SizeArgs& args) const override;
    size_t get_workspace_in_bytes(const SizeArgs& args) const override;
    void exec(const ExecArgs& args) const override;

    const char* name() const override { return "CHANNEL_WISE_SMALL"; }
    MEGDNN_DECL_ALGO_TYPE(CUDA_CHANWISE_SMALL)
    AlgoAttribute attribute() const override {
        return AlgoAttribute::REPRODUCIBLE;
    }
};

class ConvolutionBackwardDataImpl::AlgoBFloat16 final : public AlgoBase {
public:
    bool is_available(const SizeArgs& args) const override;
    size_t get_workspace_in_bytes(const SizeArgs& args) const override;
    void exec(const ExecArgs& args) const override;

    std::vector<SearchItem> get_subopr_list(
            const TensorLayoutArray& layouts,
            const OperatorBase* opr) const override;

    const char* name() const override {
        return "CONVOLUTION_BACKWARD_DATD_BFLOAT16";
    }

    AlgoAttribute attribute() const override {
        return AlgoAttribute::REPRODUCIBLE;
    }

private:
    WorkspaceBundle get_workspace_bundle(void* ptr, const SizeArgs& args) const;
    MEGDNN_DECL_ALGO_TYPE(CUDA_BFLOAT16)
};

//! implement group conv by another algo
class ConvolutionBackwardDataImpl::AlgoGroupConvGeneral final
        : public AlgoBase {
    AlgoBase* m_impl;
    std::string m_name;

public:
    AlgoGroupConvGeneral(AlgoBase* impl);

    bool is_available(const SizeArgs& args) const override;
    size_t get_workspace_in_bytes(const SizeArgs& args) const override;
    void exec(const ExecArgs& args) const override;

    const char* name() const override { return m_name.c_str(); }


    static void modify_size_args(SizeArgs& args, TensorLayout& diff_pg,
                                 TensorLayout& grad_pg);
    MEGDNN_DECL_ALGO_TYPE(CUDA_GROUP_CONV_GENERAL)
    AlgoAttribute attribute() const override {
        auto ret = static_cast<AlgoAttribute>(0);
        if (m_impl->contain_attribute_all(AlgoAttribute::REPRODUCIBLE)) {
            ret |= AlgoAttribute::REPRODUCIBLE;
        }
        return ret;
    }
};

class ConvolutionBackwardDataImpl::AlgoInt8NCHW4DotProdImplicitGemm final
        : public AlgoBase {
public:
    struct AlgoParam {
        int threadblock_m;
        int threadblock_n;
        int threadblock_k;
        int warp_m;
        int warp_n;
        int warp_k;
        int stage;
        std::string to_string() {
            return ssprintf("_%dX%dX%d_%dX%dX%d_%dstage", threadblock_m,
                            threadblock_n, threadblock_k, warp_m, warp_n,
                            warp_k, stage);
        }
    };
    AlgoInt8NCHW4DotProdImplicitGemm(AlgoParam algo_param)
            : m_algo_param{algo_param},
              m_name{ssprintf("INT8_NCHW4_DOTPROD_IMPLICIT_GEMM%s",
                              m_algo_param.to_string().c_str())} {}
    bool is_available(const SizeArgs& args) const override;
    size_t get_workspace_in_bytes(const SizeArgs& args) const override;
    void exec(const ExecArgs& args) const override;
    const char* name() const override { return m_name.c_str(); }
    AlgoAttribute attribute() const override {
        return AlgoAttribute::REPRODUCIBLE;
    }
    MEGDNN_DECL_ALGO_TYPE(CUDA_IMPLICIT_GEMM_NCHW4_DOTPROD_INT8)
private:
    WorkspaceBundle get_workspace_bundle(dt_byte* raw_ptr,
                                         const SizeArgs& args) const;
    AlgoParam m_algo_param;
    std::string m_name;
};

class ConvolutionBackwardDataImpl::AlgoInt8NCHWDotProdImplicitGemm final
        : public AlgoBase {
public:
    bool is_available(const SizeArgs& args) const override;
    size_t get_workspace_in_bytes(const SizeArgs& args) const override;
    void exec(const ExecArgs& args) const override;
    const char* name() const override {
        return "INT8_NCHW_DOTPROD_IMPLICIT_GEMM";
    }
    AlgoAttribute attribute() const override {
        return AlgoAttribute::REPRODUCIBLE;
    }
    MEGDNN_DECL_ALGO_TYPE(CUDA_IMPLICIT_GEMM_NCHW_DOTPROD_INT8);
private:
    WorkspaceBundle get_workspace_bundle(dt_byte* raw_ptr,
                                         const SizeArgs& args) const;
};

class ConvolutionBackwardDataImpl::AlgoPack : NonCopyableObj {
    // defined in cudnn.cpp
    void fill_cudnn_algos();
    // defined in implicit_gemm_int8_nchw4_dp4a.cpp
    void fill_int8_dp4a_algos();

    AlgoBase::Mapper m_all_algos_map;

public:
    AlgoPack();

    std::vector<AlgoCUDNN> cudnn;
    AlgoMatmul matmul;
    AlgoChanwise chanwise;
    AlgoChanwiseSmall chanwise_small;
    std::vector<AlgoGroupConvGeneral> gconv;
    std::unordered_map<AlgoBase*, AlgoGroupConvGeneral*> algo2gconv;
    AlgoBFloat16 bfloat16;
    std::vector<AlgoInt8NCHW4DotProdImplicitGemm> int8_nchw4_dotprod;
    AlgoInt8NCHWDotProdImplicitGemm int8_nchw_dotprod;

    std::vector<AlgoBase*>
            //! all algorithms
            all_algos,
            //! non-cudnn algos, used for heuristic if cudnn is not supported
            non_cudnn_algos, bfloat16_algos, int8_algos;

    AlgoCUDNN* cudnn_from_enum(cudnnConvolutionBwdDataAlgo_t algo);

    const AlgoBase::Mapper& all_algos_map() const { return m_all_algos_map; }
};

}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen
