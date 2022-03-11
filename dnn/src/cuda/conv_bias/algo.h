/**
 * \file dnn/src/cuda/conv_bias/algo.h
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

#include "megdnn/oprs.h"

#include "src/common/algo_base.h"
#include "src/common/metahelper.h"
#include "src/common/utils.h"
#include "src/cuda/conv_bias/conv_bias_int8.cuh"
#include "src/cuda/conv_bias/helper.h"
#include "src/cuda/conv_bias/opr_impl.h"
#include "src/cuda/convolution_helper/parameter.cuh"
#include "src/cuda/cudnn_wrapper.h"

#include <cuda.h>
#include <memory>
#include <unordered_map>

namespace cutlass {
namespace library {

// forward declaration of cutlass library concepts, we hope that algo.h does
// not depend on cutlass headers

class Operation;

}  // namespace library
}  // namespace cutlass

namespace megdnn {
namespace cuda {

/*!
 * \brief base class for conv bias algos
 *
 * All the algo impls should try to support non-contiguous batch dim, for group
 * conv execution.
 */
class ConvBiasForwardImpl::AlgoBase : public Algorithm {
protected:
    ~AlgoBase() = default;

public:
    enum class AlgoType : uint32_t {
        CUDA_CUDNN_CONVBIAS,
        CUDA_CHANWISE,
        CUDA_CHANWISE_SMALL,
        CUDA_DEPTHWISE_LARGE_FILTER,
        CUDA_CHANWISE_INT8X8X32,
        CUDA_CUDNN_CONV,
        CUDA_INPLACE_MATMUL,
        CUDA_MATMUL,
        CUDA_MATMUL_INT8X8X32,
        CUDA_BATCHED_MATMUL,
        CUDA_GROUP_CONV_GENERAL,
        CUDA_WMMA_UINT4X4X32,
        CUDA_IMPLICIT_GEMM_CHWN4_DOTPROD_INT8,
        CUDA_IMPLICIT_GEMM_NCHW4_DOTPROD_INT8,
        CUDA_IMPLICIT_GEMM_CHWN4_IMMA_INT8,
        CUDA_IMPLICIT_GEMM_NCHW4_IMMA_INT8,
        CUDA_IMPLICIT_GEMM_REORDER_FILTER_CHWN4_IMMA_INT8,
        CUDA_IMPLICIT_GEMM_UNROLL_WIDTH_CHWN4_IMMA_INT8,
        CUDA_IMPLICIT_GEMM_IMMA_NCHW32_INT8,
        CUDA_IMPLICIT_GEMM_IMMA_NHWC_INT8,
        CUDA_IMPLICIT_GEMM_IMMA_NCHW64_INT4_INT4,
        CUDA_IMPLICIT_GEMM_IMMA_NCHW64_UINT4_INT4,
        CUDA_IMPLICIT_GEMM_IMMA_NHWC_INT4_INT4,
        CUDA_IMPLICIT_GEMM_IMMA_NHWC_UINT4_INT4,
        CUDA_BFLOAT16,
        CUDA_IMPLICIT_GEMM_SASS_NCHW4_DOTPROD_INT8,
        CUDA_IMPLICIT_GEMM_1X1_SASS_NCHW4_DOTPROD_INT8,
        CUDA_IMPLICIT_GEMM_SASS_NCHW32_IMMA_INT8,
        CUDA_IMPLICIT_GEMM_1X1_SASS_NCHW32_IMMA_INT8,
        CUDA_IMPLICIT_GEMM_SASS_NCHW64_IMMA_INT4_INT4,
        CUDA_IMPLICIT_GEMM_SASS_NCHW64_IMMA_UINT4_INT4,
        CUDA_FALLBACK_NCHW_INT4,
        CUDA_IMPLICIT_BATCHED_GEMM_FMA_NCHW_F32,
        CUDA_IMPLICIT_BATCHED_GEMM_HMMA_NCHW_F16,
        CUDA_SIMPLE_INT1,
    };
    using Mapper = std::unordered_map<AlgorithmDesc, AlgoBase*>;

    AlgoBase() : Algorithm() { m_handle_type = Handle::HandleType::CUDA; }
    struct SizeArgs : public conv_bias::BiasForwardSizeArgs {
        const ConvBiasForwardImpl* opr;
        const PreprocessedFilter* preprocessed_filter;

        std::string to_string() const;
        SizeArgs(
                const ConvBiasForwardImpl* opr, const TensorLayout& src,
                const TensorLayout& filter, const TensorLayout& bias,
                const TensorLayout& z, const TensorLayout& dst,
                const PreprocessedFilter* preprocessed_filter = nullptr);
        SizeArgs(
                const ConvBiasForwardImpl* opr, const TensorLayout& src,
                const TensorLayout& filter, const CanonizedFilterMeta& filter_meta,
                const TensorLayout& bias, const TensorLayout& z,
                const TensorLayout& dst,
                const PreprocessedFilter* preprocessed_filter = nullptr);

        void init_conv_bias_desc(conv_bias::CUDNNForwardDescs& desc) const {
            desc.set_conv_bias(
                    *src_layout, filter_meta, *dst_layout, *bias_layout, *z_layout,
                    opr->param());
        }

        void init_conv_desc(conv_bias::CUDNNForwardDescs& desc) const {
            desc.set_conv(*src_layout, filter_meta, *dst_layout, opr->param());
        }
    };
    struct ExecArgs : public SizeArgs {
        const TensorND *src_tensor, *filter_tensor, *bias_tensor, *z_tensor,
                *dst_tensor;
        Workspace workspace;

        ExecArgs(
                ConvBiasForwardImpl* opr, _megdnn_tensor_in src,
                _megdnn_tensor_in filter, _megdnn_tensor_in bias, _megdnn_tensor_in z,
                _megdnn_tensor_out dst, _megdnn_workspace workspace,
                const PreprocessedFilter* preprocessed_filter = nullptr);
    };
    virtual bool is_available(const SizeArgs& args) const = 0;
    virtual size_t get_workspace_in_bytes(const SizeArgs& args) const = 0;
    virtual void exec(const ExecArgs& args) const = 0;
    virtual size_t get_preprocess_workspace_in_bytes(const SizeArgs& args) const {
        MEGDNN_MARK_USED_VAR(args);
        return 0;
    }
    virtual SmallVector<TensorLayout> deduce_preprocessed_filter_layout(
            const SizeArgs& args) const {
        MEGDNN_MARK_USED_VAR(args);
        return {};
    }
    virtual void exec_preprocess(const ExecArgs& args) const {
        MEGDNN_MARK_USED_VAR(args);
    }

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
                "conv bias fwd algo %s: required workspace %zu bytes, got %zu", name(),
                req, workspace.size);
        return *this;
    }

    virtual bool is_cudnn() const { return false; }
};

class ConvBiasForwardImpl::AlgoCUDNNConvBiasActivation final : public AlgoBase {
public:
    AlgoCUDNNConvBiasActivation(cudnnConvolutionFwdAlgo_t cudnn_enum)
            : m_cudnn_enum(cudnn_enum) {
        megdnn_assert(
                CudnnAlgoPack::conv_fwd_algos().find(cudnn_enum) !=
                CudnnAlgoPack::conv_fwd_algos().end());
        m_attr = CudnnAlgoPack::conv_fwd_algos().at(cudnn_enum);
        m_name = ConvBiasForward::algo_name<DefaultParam>(
                "CUDNN:ConvBiasActivation:" + m_attr.name, {});
    }

    size_t get_workspace_in_bytes(const SizeArgs& args) const override;
    void exec(const ExecArgs& args) const override;
    param::Convolution get_param_convolution(const SizeArgs& args) const;
    bool is_available(const SizeArgs&) const override;

    const char* name() const override { return m_name.c_str(); }

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

    cudnnConvolutionFwdAlgo_t cudnn_enum() { return m_cudnn_enum; }

    bool is_cudnn() const override { return true; }

    MEGDNN_DECL_ALGO_TYPE(CUDA_CUDNN_CONVBIAS)

    std::string param() const override {
        std::string ret;
        serialize_write_pod(m_cudnn_enum, ret);
        return ret;
    }

private:
    std::string m_name;
    cudnnConvolutionFwdAlgo_t m_cudnn_enum;
    CudnnAlgoPack::Attr m_attr;
};

class ConvBiasForwardImpl::AlgoChanwise final : public AlgoBase {
public:
    bool is_available(const SizeArgs& args) const override;
    size_t get_workspace_in_bytes(const SizeArgs& args) const override;
    void exec(const ExecArgs& args) const override;

    const char* name() const override {
        if (m_name.empty()) {
            m_name = ConvBiasForward::algo_name<DirectParam>("CHANNEL_WISE", {});
        }
        return m_name.c_str();
    }
    AlgoAttribute attribute() const override { return AlgoAttribute::REPRODUCIBLE; }

    MEGDNN_DECL_ALGO_TYPE(CUDA_CHANWISE)

private:
    mutable std::string m_name;
};

class ConvBiasForwardImpl::AlgoChanwiseSmall final : public AlgoBase {
public:
    bool is_available(const SizeArgs& args) const override;
    size_t get_workspace_in_bytes(const SizeArgs& args) const override;
    void exec(const ExecArgs& args) const override;

    const char* name() const override {
        if (m_name.empty()) {
            m_name = ConvBiasForward::algo_name<DirectParam>("CHANNEL_WISE_SMALL", {});
        }
        return m_name.c_str();
    }
    MEGDNN_DECL_ALGO_TYPE(CUDA_CHANWISE_SMALL)
    AlgoAttribute attribute() const override { return AlgoAttribute::REPRODUCIBLE; }

private:
    mutable std::string m_name;
};

class ConvBiasForwardImpl::AlgoDepthwiseLargeFilter final : public AlgoBase {
public:
    bool is_available(const SizeArgs& args) const override;
    size_t get_workspace_in_bytes(const SizeArgs& args) const override;
    void exec(const ExecArgs& args) const override;

    const char* name() const override {
        if (m_name.empty()) {
            m_name = ConvBiasForward::algo_name<DirectParam>(
                    "DEPTHWISE_LARGE_FILTER", {});
        }
        return m_name.c_str();
    }
    MEGDNN_DECL_ALGO_TYPE(CUDA_DEPTHWISE_LARGE_FILTER)
    AlgoAttribute attribute() const override { return AlgoAttribute::REPRODUCIBLE; }

private:
    mutable std::string m_name;
};

class ConvBiasForwardImpl::AlgoChanwise8x8x32 final : public AlgoBase {
public:
    bool is_available(const SizeArgs& args) const override;
    size_t get_workspace_in_bytes(const SizeArgs& args) const override;
    void exec(const ExecArgs& args) const override;
    const char* name() const override {
        if (m_name.empty()) {
            m_name = ConvBiasForward::algo_name<DirectParam>("CHANNEL_WISE_8X8X32", {});
        }
        return m_name.c_str();
    }
    MEGDNN_DECL_ALGO_TYPE(CUDA_CHANWISE_INT8X8X32)
    AlgoAttribute attribute() const override { return AlgoAttribute::REPRODUCIBLE; }

private:
    mutable std::string m_name;
};

class ConvBiasForwardImpl::AlgoCUDNNConv final : public AlgoBase {
public:
    AlgoCUDNNConv(cudnnConvolutionFwdAlgo_t cudnn_enum) : m_cudnn_enum(cudnn_enum) {
        megdnn_assert(
                CudnnAlgoPack::conv_fwd_algos().find(cudnn_enum) !=
                CudnnAlgoPack::conv_fwd_algos().end());
        m_attr = CudnnAlgoPack::conv_fwd_algos().at(cudnn_enum);
        m_name = ConvBiasForward::algo_name<DefaultParam>(
                "CUDNN:Convolution:" + m_attr.name, {});
    }

    bool is_available(const SizeArgs& args) const override;
    size_t get_workspace_in_bytes(const SizeArgs& args) const override;
    void exec(const ExecArgs& args) const override;

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

    const char* name() const override { return m_name.c_str(); }

    cudnnConvolutionFwdAlgo_t cudnn_enum() const { return m_cudnn_enum; }

    bool is_cudnn() const override { return true; }

    MEGDNN_DECL_ALGO_TYPE(CUDA_CUDNN_CONV)

    std::string param() const override {
        std::string ret;
        serialize_write_pod(m_cudnn_enum, ret);
        return ret;
    }

private:
    std::string m_name;
    cudnnConvolutionFwdAlgo_t m_cudnn_enum;
    CudnnAlgoPack::Attr m_attr;

    WorkspaceBundle get_workspace_bundle(void* ptr, const SizeArgs& args) const;
};

//! compute small matmul in the kernel
class ConvBiasForwardImpl::AlgoInplaceMatmul final : public AlgoBase {
public:
    bool is_available(const SizeArgs& args) const override;
    size_t get_workspace_in_bytes(const SizeArgs& args) const override;
    void exec(const ExecArgs& args) const override;

    const char* name() const override {
        if (m_name.empty()) {
            m_name = ConvBiasForward::algo_name<ConvBias::MatmulParam>(
                    "INPLACE_MATMUL", {});
        }
        return m_name.c_str();
    }
    MEGDNN_DECL_ALGO_TYPE(CUDA_INPLACE_MATMUL)
    AlgoAttribute attribute() const override { return AlgoAttribute::REPRODUCIBLE; }

private:
    mutable std::string m_name;
};

//! im2col and matmul, with dilation
class ConvBiasForwardImpl::AlgoMatmul final : public AlgoBase {
    template <typename T>
    static void exec_internal(const ExecArgs& args, const WorkspaceBundle& bundle);

public:
    bool is_available(const SizeArgs& args) const override;
    size_t get_workspace_in_bytes(const SizeArgs& args) const override;
    void exec(const ExecArgs& args) const override;

    const char* name() const override {
        if (m_name.empty()) {
            m_name = ConvBiasForward::algo_name<ConvBias::MatmulParam>("MATMUL", {});
        }
        return m_name.c_str();
    }

    std::vector<SearchItem> get_subopr_list(
            const TensorLayoutArray& layouts, const OperatorBase* opr) const override;
    MEGDNN_DECL_ALGO_TYPE(CUDA_MATMUL)
    AlgoAttribute attribute() const override {
        return AlgoAttribute::REPRODUCIBLE | AlgoAttribute::ACCURACY_DEPEND_ON_BATCH;
    }

private:
    WorkspaceBundle get_workspace_bundle(void* ptr, const SizeArgs& args) const;
    mutable std::string m_name;
};

class ConvBiasForwardImpl::AlgoMatmul8x8x32 final : public AlgoBase {
public:
    bool is_available(const SizeArgs& args) const override;
    size_t get_workspace_in_bytes(const SizeArgs& args) const override;
    void exec(const ExecArgs& args) const override;
    const char* name() const override {
        if (m_name.empty()) {
            m_name = ConvBiasForward::algo_name<ConvBiasForward::MatmulParam>(
                    "MATMUL8X8X32", {});
        }
        return m_name.c_str();
    }
    MEGDNN_DECL_ALGO_TYPE(CUDA_MATMUL_INT8X8X32)
    AlgoAttribute attribute() const override { return AlgoAttribute::REPRODUCIBLE; }

private:
    bool need_src_unroll(const SizeArgs& args) const;
    bool need_filter_reshape(const SizeArgs& args) const;
    template <Param::Format>
    WorkspaceBundle get_bundle(const SizeArgs& args) const;
    template <Param::Format>
    void exec_internal(const ExecArgs& args) const;
    mutable std::string m_name;
};

class ConvBiasForwardImpl::AlgoBatchedMatmul final : public AlgoBase {
public:
    bool is_available(const SizeArgs& args) const override;
    size_t get_workspace_in_bytes(const SizeArgs& args) const override;
    void exec(const ExecArgs& args) const override;

    const char* name() const override {
        if (m_name.empty()) {
            m_name = ConvBiasForward::algo_name<ConvBiasForward::MatmulParam>(
                    "BATCHED_MATMUL", {});
        }
        return m_name.c_str();
    }

    std::vector<SearchItem> get_subopr_list(
            const TensorLayoutArray& layouts, const OperatorBase* opr) const override;

    AlgoAttribute attribute() const override {
        return AlgoAttribute::REPRODUCIBLE | AlgoAttribute::ACCURACY_DEPEND_ON_BATCH;
    }

    MEGDNN_DECL_ALGO_TYPE(CUDA_BATCHED_MATMUL)

private:
    WorkspaceBundle get_workspace_bundle(void* ptr, const SizeArgs& args) const;
    mutable std::string m_name;
};

//! implement group conv by another algo
class ConvBiasForwardImpl::AlgoGroupConvGeneral final : public AlgoBase {
public:
    bool is_available(const SizeArgs& args) const override;
    size_t get_workspace_in_bytes(const SizeArgs& args) const override;
    void exec(const ExecArgs& args) const override;

    std::vector<SearchItem> get_subopr_list(
            const TensorLayoutArray& layouts, const OperatorBase* opr) const override;

    const char* name() const override {
        if (m_name.empty()) {
            m_name = ConvBiasForward::algo_name<DirectParam>("CUDA:GROUP_CONV", {});
        }
        return m_name.c_str();
    }

    AlgoAttribute attribute() const override { return AlgoAttribute::REPRODUCIBLE; }

    MEGDNN_DECL_ALGO_TYPE(CUDA_GROUP_CONV_GENERAL)

private:
    WorkspaceBundle get_workspace_bundle(void* ptr, const SizeArgs& args) const;
    mutable std::string m_name;
};

#if CUDA_VERSION >= 10000
class ConvBiasForwardImpl::AlgoQUInt4x4x32WMMA final : public AlgoBase {
public:
    AlgoQUInt4x4x32WMMA() = default;
    bool is_available(const SizeArgs& args) const override;
    size_t get_workspace_in_bytes(const SizeArgs& args) const override;
    void exec(const ExecArgs& args) const override;
    const char* name() const override { return "QUINT4x4x32_WMMA"; }
    AlgoAttribute attribute() const override { return AlgoAttribute::REPRODUCIBLE; }

private:
    WorkspaceBundle get_workspace_bundle(dt_byte* raw_ptr, const SizeArgs& args) const;
    bool use_kernel_fhxfw(const SizeArgs& args) const;
    size_t get_workspace_in_bytes_do_conv(const SizeArgs& args) const;
    MEGDNN_DECL_ALGO_TYPE(CUDA_WMMA_UINT4X4X32)
};
#endif

class ConvBiasForwardImpl::AlgoInt8CHWN4DotProdImplicitGemm final : public AlgoBase {
public:
    AlgoInt8CHWN4DotProdImplicitGemm() = default;
    bool is_available(const SizeArgs& args) const override;
    size_t get_workspace_in_bytes(const SizeArgs& args) const override;
    void exec(const ExecArgs& args) const override;
    const char* name() const override { return "INT8_CHWN4_DOTPROD_IMPLICIT_GEMM"; }
    AlgoAttribute attribute() const override { return AlgoAttribute::REPRODUCIBLE; }
    template <typename BiasVisitor>
    static void dispatch_nonlinear_mode(
            const int8_t* d_src, const int8_t* d_filter, BiasVisitor bias_visitor,
            const int8_t* d_z, int8_t* d_dst, const convolution::ConvParam& param,
            float alpha, float beta, float gamma, float scale, cudaStream_t stream,
            param::ConvBias::NonlineMode nonlinear_mode);
    MEGDNN_DECL_ALGO_TYPE(CUDA_IMPLICIT_GEMM_CHWN4_DOTPROD_INT8)
};

/*********************** Cutlass Algorithms ************************/

/* The inheritance of cutlass algorithm classes:
 *
 * AlgoCutlassConvolutionBase
 * +
 * +--- AlgoInt8NCHW4DotProdImplicitGemm
 * +--- AlgoInt8NCHW32IMMAImplicitGemm
 * +--- AlgoInt8NHWCIMMAImplicitGemm
 * +
 * +--- AlgoInt4NCHW64IMMAImplicitGemmBase
 * +----+--- AlgoInt4Int4NCHW64IMMAImplicitGemm
 * +----+--- AlgoUInt4Int4NCHW64IMMAImplicitGemm
 * +
 * +--- AlgoInt4NHWCIMMAImplicitGemmBase
 * +----+--- AlgoInt4Int4NHWCIMMAImplicitGemm
 * +----+--- AlgoUInt4Int4NHWCIMMAImplicitGemm
 * +
 * +--- AlgoFloat32NCHWImplicitBatchedGemm
 * +--- AlgoFloat16NCHWHMMAImplicitBatchedGemm
 */

/*
 * The base class for all cutlass algorithm classes
 */
class ConvBiasForwardImpl::AlgoCutlassConvolutionBase : public AlgoBase {
public:
    // corresponds to cutlass::conv::Operator. we hope that algo.h does not
    // depend on cutlass headers
    enum class ConvOperator { kFprop, kDgrad, kWgrad };

    // corresponds to cutlass::conv::ConvType. we hope that algo.h does not
    // depend on cutlass headers
    enum class ConvType {
        kConvolution,
        kBatchConvolution,
        kLocal,
        kLocalShare,
        kDepthwiseConvolution,
    };

    // common parameters for operation selection
    struct AlgoParam {
        int threadblock_m;
        int threadblock_n;
        int threadblock_k;
        int warp_m;
        int warp_n;
        int warp_k;
        int instruction_m;
        int instruction_n;
        int instruction_k;
        int stage;
        int access_size;

        AlgoParam(
                int threadblock_m_, int threadblock_n_, int threadblock_k_, int warp_m_,
                int warp_n_, int warp_k_, int instruction_m_, int instruction_n_,
                int instruction_k_, int stage_, int access_size_ = 0);

        std::string to_string() const;
    };

    AlgoCutlassConvolutionBase(AlgoParam algo_param) : m_algo_param{algo_param} {}

    // generate a cutlass::library::ConvolutionKey and find the corresponding
    // operation (cutlass kernel) from the global OperationTable
    const cutlass::library::Operation* get_cutlass_conv_op(
            const SizeArgs& args, ConvOperator conv_op, ConvType conv_type,
            bool use_conv_filter_unity_opt, bool without_shared_load) const;

    // execute the cutlass kernel found by get_cutlass_conv_op. we give
    // subclasses full freedom to decide where and how these arguments are
    // extracted
    void execute_cutlass_conv_op(
            const cutlass::library::Operation* op, const void* src, const void* filter,
            const void* bias, const void* z, void* dst, void* workspace, size_t n,
            size_t hi, size_t wi, size_t ci, size_t co, size_t fh, size_t fw, size_t ho,
            size_t wo, size_t ph, size_t pw, size_t sh, size_t sw, size_t dh, size_t dw,
            const void* alpha, const void* beta, const void* gamma, const void* delta,
            const void* theta, const void* threshold, const void* dst_scale,
            cudaStream_t stream, const void* extra_param = nullptr,
            size_t groups = 1) const;

protected:
    AlgoParam m_algo_param;
};

class ConvBiasForwardImpl::AlgoInt8NCHW4DotProdImplicitGemm final
        : public AlgoCutlassConvolutionBase {
public:
    AlgoInt8NCHW4DotProdImplicitGemm(AlgoParam algo_param)
            : AlgoCutlassConvolutionBase(algo_param),
              m_name{ssprintf(
                      "INT8_NCHW4_DOTPROD_IMPLICIT_GEMM%s",
                      m_algo_param.to_string().c_str())} {}
    bool is_available(const SizeArgs& args) const override;
    size_t get_workspace_in_bytes(const SizeArgs& args) const override;
    void exec(const ExecArgs& args) const override;
    const char* name() const override { return m_name.c_str(); }
    AlgoAttribute attribute() const override { return AlgoAttribute::REPRODUCIBLE; }
    size_t get_preprocess_workspace_in_bytes(const SizeArgs& args) const override;
    SmallVector<TensorLayout> deduce_preprocessed_filter_layout(
            const SizeArgs& args) const override;
    void exec_preprocess(const ExecArgs& args) const override;
    MEGDNN_DECL_ALGO_TYPE(CUDA_IMPLICIT_GEMM_NCHW4_DOTPROD_INT8)

    std::string param() const override {
        std::string ret;
        serialize_write_pod(m_algo_param, ret);
        return ret;
    }

private:
    WorkspaceBundle get_workspace_bundle(dt_byte* raw_ptr, const SizeArgs& args) const;
    std::string m_name;
};

class ConvBiasForwardImpl::AlgoFallbackNCHWQS8 final : public AlgoBase {
public:
    bool is_available(const SizeArgs& args) const override;
    size_t get_workspace_in_bytes(const SizeArgs& args) const override;
    void exec(const ExecArgs& args) const override;
    const char* name() const override { return "FALLBACK_CONV_NCHW_QS8"; }
    AlgoAttribute attribute() const override { return AlgoAttribute::REPRODUCIBLE; }
    MEGDNN_DECL_ALGO_TYPE(CUDA_IMPLICIT_GEMM_NCHW4_DOTPROD_INT8)
    std::vector<SearchItem> get_subopr_list(
            const TensorLayoutArray& layouts, const OperatorBase* opr) const override;

private:
    WorkspaceBundle get_workspace_bundle(void* ptr, const SizeArgs& args) const;
};

#if CUDA_VERSION >= 10000
class ConvBiasForwardImpl::AlgoInt8CHWN4IMMAImplicitGemm final : public AlgoBase {
public:
    enum class MMATileSize : uint32_t { IMMA16x16x16, IMMA32x8x16, IMMA8x32x16 };
    AlgoInt8CHWN4IMMAImplicitGemm(MMATileSize mma_tile_size)
            : m_mma_tile_size{mma_tile_size},
              m_name{"INT8_CHWN4_IMMA_IMPLICIT_GEMM_" + to_string(m_mma_tile_size)} {}
    bool is_available(const SizeArgs& args) const override;
    size_t get_workspace_in_bytes(const SizeArgs& args) const override;
    void exec(const ExecArgs& args) const override;
    const char* name() const override { return m_name.c_str(); }
    AlgoAttribute attribute() const override { return AlgoAttribute::REPRODUCIBLE; }
    template <typename BiasVisitor>
    static void dispatch_nonlinear_mode(
            const int8_t* d_src, const int8_t* d_filter, BiasVisitor bias_visitor,
            int8_t* d_z, int8_t* d_dst, const convolution::ConvParam& param,
            float alpha, float beta, float gamma, float scale, cudaStream_t stream,
            param::ConvBias::NonlineMode nonlinear_mode, MMATileSize mma_tile_size);
    static std::string to_string(MMATileSize mma_tile_size);

    MEGDNN_DECL_ALGO_TYPE(CUDA_IMPLICIT_GEMM_CHWN4_IMMA_INT8)

    std::string param() const override {
        std::string ret;
        serialize_write_pod(m_mma_tile_size, ret);
        return ret;
    }

private:
    MMATileSize m_mma_tile_size;
    std::string m_name;
};

class ConvBiasForwardImpl::AlgoInt8NCHW4IMMAImplicitGemm final : public AlgoBase {
public:
    using MMATileSize = AlgoInt8CHWN4IMMAImplicitGemm::MMATileSize;
    AlgoInt8NCHW4IMMAImplicitGemm(MMATileSize mma_tile_size)
            : m_mma_tile_size{mma_tile_size},
              m_name{"INT8_NCHW4_IMMA_IMPLICIT_GEMM_" +
                     AlgoInt8CHWN4IMMAImplicitGemm::to_string(m_mma_tile_size)} {}
    bool is_available(const SizeArgs& args) const override;
    size_t get_workspace_in_bytes(const SizeArgs& args) const override;
    void exec(const ExecArgs& args) const override;
    const char* name() const override { return m_name.c_str(); }
    MEGDNN_DECL_ALGO_TYPE(CUDA_IMPLICIT_GEMM_NCHW4_IMMA_INT8)

    std::string param() const override {
        std::string ret;
        serialize_write_pod(m_mma_tile_size, ret);
        return ret;
    }
    AlgoAttribute attribute() const override { return AlgoAttribute::REPRODUCIBLE; }

private:
    WorkspaceBundle get_workspace_bundle(dt_byte* raw_ptr, const SizeArgs& args) const;
    MMATileSize m_mma_tile_size;
    std::string m_name;
};

class ConvBiasForwardImpl::AlgoInt8CHWN4IMMAImplicitGemmReorderFilter final
        : public AlgoBase {
public:
    using MMATileSize = AlgoInt8CHWN4IMMAImplicitGemm::MMATileSize;
    AlgoInt8CHWN4IMMAImplicitGemmReorderFilter(MMATileSize mma_tile_size)
            : m_mma_tile_size{mma_tile_size},
              m_name{"INT8_CHWN4_IMMA_IMPLICIT_GEMM_REORDER_FILTER_" +
                     AlgoInt8CHWN4IMMAImplicitGemm::to_string(m_mma_tile_size)} {}
    bool is_available(const SizeArgs& args) const override;
    size_t get_workspace_in_bytes(const SizeArgs& args) const override;
    void exec(const ExecArgs& args) const override;
    const char* name() const override { return m_name.c_str(); }
    MEGDNN_DECL_ALGO_TYPE(CUDA_IMPLICIT_GEMM_REORDER_FILTER_CHWN4_IMMA_INT8)

    std::string param() const override {
        std::string ret;
        serialize_write_pod(m_mma_tile_size, ret);
        return ret;
    }
    AlgoAttribute attribute() const override { return AlgoAttribute::REPRODUCIBLE; }

private:
    MMATileSize m_mma_tile_size;
    std::string m_name;
};

class ConvBiasForwardImpl::AlgoInt8CHWN4IMMAImplicitGemmUnrollWidth final
        : public AlgoBase {
public:
    using MMATileSize = AlgoInt8CHWN4IMMAImplicitGemm::MMATileSize;
    AlgoInt8CHWN4IMMAImplicitGemmUnrollWidth(MMATileSize mma_tile_size)
            : m_mma_tile_size{mma_tile_size},
              m_name{"INT8_CHWN4_IMMA_IMPLICIT_GEMM_UNROLL_WIDTH_" +
                     AlgoInt8CHWN4IMMAImplicitGemm::to_string(m_mma_tile_size)} {}
    bool is_available(const SizeArgs& args) const override;
    size_t get_workspace_in_bytes(const SizeArgs& args) const override;
    void exec(const ExecArgs& args) const override;
    const char* name() const override { return m_name.c_str(); }
    MEGDNN_DECL_ALGO_TYPE(CUDA_IMPLICIT_GEMM_UNROLL_WIDTH_CHWN4_IMMA_INT8)

    std::string param() const override {
        std::string ret;
        serialize_write_pod(m_mma_tile_size, ret);
        return ret;
    }
    AlgoAttribute attribute() const override { return AlgoAttribute::REPRODUCIBLE; }

private:
    MMATileSize m_mma_tile_size;
    std::string m_name;
};
#endif

#if CUDA_VERSION >= 10020
class ConvBiasForwardImpl::AlgoInt8NCHW32IMMAImplicitGemm final
        : public AlgoCutlassConvolutionBase {
public:
    AlgoInt8NCHW32IMMAImplicitGemm(AlgoParam algo_param)
            : AlgoCutlassConvolutionBase(algo_param) {
        m_name = ConvBias::algo_name<ConvBias::DirectParam>(
                ssprintf(
                        "INT8_NCHW32_IMMA_IMPLICIT_GEMM_%s",
                        to_string(m_algo_param).c_str()),
                ConvBias::DirectParam{});
    }
    bool is_available(const SizeArgs& args) const override;
    size_t get_workspace_in_bytes(const SizeArgs& args) const override;
    void exec(const ExecArgs& args) const override;
    const char* name() const override { return m_name.c_str(); }
    AlgoAttribute attribute() const override { return AlgoAttribute::REPRODUCIBLE; }
    static std::string to_string(AlgoParam algo_param);
    size_t get_preprocess_workspace_in_bytes(const SizeArgs& args) const override;
    SmallVector<TensorLayout> deduce_preprocessed_filter_layout(
            const SizeArgs& args) const override;
    void exec_preprocess(const ExecArgs& args) const override;
    MEGDNN_DECL_ALGO_TYPE(CUDA_IMPLICIT_GEMM_IMMA_NCHW32_INT8)

    std::string param() const override {
        std::string ret;
        serialize_write_pod(m_algo_param, ret);
        return ret;
    }

private:
    WorkspaceBundle get_workspace_bundle(dt_byte* raw_ptr, const SizeArgs& args) const;

    std::string m_name;
};

class ConvBiasForwardImpl::AlgoInt8NHWCIMMAImplicitGemm final
        : public AlgoCutlassConvolutionBase {
public:
    AlgoInt8NHWCIMMAImplicitGemm(AlgoParam algo_param)
            : AlgoCutlassConvolutionBase(algo_param) {
        m_name = ConvBias::algo_name<ConvBias::DirectParam>(
                ssprintf(
                        "INT8_NHWC_IMMA_IMPLICIT_GEMM_%s",
                        to_string(m_algo_param).c_str()),
                ConvBias::DirectParam{});
    }
    bool is_available(const SizeArgs& args) const override;
    size_t get_workspace_in_bytes(const SizeArgs& args) const override;
    void exec(const ExecArgs& args) const override;
    const char* name() const override { return m_name.c_str(); }
    AlgoAttribute attribute() const override { return AlgoAttribute::REPRODUCIBLE; }
    static std::string to_string(AlgoParam algo_param);
    size_t get_preprocess_workspace_in_bytes(const SizeArgs& args) const override;
    SmallVector<TensorLayout> deduce_preprocessed_filter_layout(
            const SizeArgs& args) const override;
    void exec_preprocess(const ExecArgs& args) const override;
    MEGDNN_DECL_ALGO_TYPE(CUDA_IMPLICIT_GEMM_IMMA_NHWC_INT8)

    std::string param() const override {
        std::string ret;
        serialize_write_pod(m_algo_param, ret);
        return ret;
    }

private:
    std::tuple<float, float, float, float, float> get_constants(
            const ExecArgs& args) const;

    void reorder_filter(
            const ExecArgs& args, int interleaved, void* reordered_filter) const;

    std::string m_name;
};

class ConvBiasForwardImpl::AlgoInt4NCHW64IMMAImplicitGemmBase
        : public AlgoCutlassConvolutionBase {
public:
    AlgoInt4NCHW64IMMAImplicitGemmBase(AlgoParam algo_param)
            : AlgoCutlassConvolutionBase(algo_param) {}

    AlgoAttribute attribute() const override { return AlgoAttribute::REPRODUCIBLE; }
    const char* name() const override { return m_name.c_str(); }
    std::string param() const override;

    bool is_available(const SizeArgs& args) const override;
    void exec(const ExecArgs& args) const override;

    std::string to_string(AlgoParam algo_param);

protected:
    virtual DTypeEnum src_dtype() const = 0;

    // return filter_ptr, bias_ptr
    virtual std::tuple<void*, void*> prepare_filter_bias(
            const ExecArgs& args) const = 0;

    // return alpha, beta, gamma, delta, theta
    virtual std::tuple<float, float, float, float, float> get_constants(
            const ExecArgs& args) const = 0;

    void reorder_filter(const ExecArgs& args, void* reordered_filter) const;

    std::string m_name;
};

class ConvBiasForwardImpl::AlgoInt4Int4NCHW64IMMAImplicitGemm final
        : public AlgoInt4NCHW64IMMAImplicitGemmBase {
public:
    using Base = AlgoInt4NCHW64IMMAImplicitGemmBase;
    using AlgoParam = Base::AlgoParam;

    AlgoInt4Int4NCHW64IMMAImplicitGemm(AlgoParam algo_param) : Base{algo_param} {
        m_name = ConvBias::algo_name<ConvBias::DirectParam>(
                ssprintf(
                        "INT4_INT4_NCHW64_IMMA_IMPLICIT_GEMM_%s",
                        to_string(m_algo_param).c_str()),
                ConvBias::DirectParam{});
    }

    size_t get_workspace_in_bytes(const SizeArgs& args) const override;
    size_t get_preprocess_workspace_in_bytes(const SizeArgs& args) const override;
    SmallVector<TensorLayout> deduce_preprocessed_filter_layout(
            const SizeArgs& args) const override;
    void exec_preprocess(const ExecArgs& args) const override;

    MEGDNN_DECL_ALGO_TYPE(CUDA_IMPLICIT_GEMM_IMMA_NCHW64_INT4_INT4)

private:
    DTypeEnum src_dtype() const override { return DTypeEnum::QuantizedS4; }

    std::tuple<void*, void*> prepare_filter_bias(const ExecArgs& args) const override;

    std::tuple<float, float, float, float, float> get_constants(
            const ExecArgs& args) const override;
};

class ConvBiasForwardImpl::AlgoUInt4Int4NCHW64IMMAImplicitGemm final
        : public AlgoInt4NCHW64IMMAImplicitGemmBase {
public:
    using Base = AlgoInt4NCHW64IMMAImplicitGemmBase;
    using AlgoParam = Base::AlgoParam;

    AlgoUInt4Int4NCHW64IMMAImplicitGemm(AlgoParam algo_param) : Base{algo_param} {
        m_name = ConvBias::algo_name<ConvBias::DirectParam>(
                ssprintf(
                        "UINT4_INT4_NCHW64_IMMA_IMPLICIT_GEMM_%s",
                        to_string(m_algo_param).c_str()),
                ConvBias::DirectParam{});
    }

    size_t get_workspace_in_bytes(const SizeArgs& args) const override;
    size_t get_preprocess_workspace_in_bytes(const SizeArgs& args) const override;
    SmallVector<TensorLayout> deduce_preprocessed_filter_layout(
            const SizeArgs& args) const override;
    void exec_preprocess(const ExecArgs& args) const override;

    MEGDNN_DECL_ALGO_TYPE(CUDA_IMPLICIT_GEMM_IMMA_NCHW64_UINT4_INT4)

private:
    DTypeEnum src_dtype() const override { return DTypeEnum::Quantized4Asymm; }

    std::tuple<void*, void*> prepare_filter_bias(const ExecArgs& args) const override;

    std::tuple<float, float, float, float, float> get_constants(
            const ExecArgs& args) const override;

    void update_bias(
            const ExecArgs& args, void* updated_bias, void* reduce_filter_ptr,
            void* reduce_workspace) const;
};

class ConvBiasForwardImpl::AlgoInt4NHWCIMMAImplicitGemmBase
        : public AlgoCutlassConvolutionBase {
public:
    AlgoInt4NHWCIMMAImplicitGemmBase(AlgoParam algo_param)
            : AlgoCutlassConvolutionBase(algo_param) {}

    AlgoAttribute attribute() const override { return AlgoAttribute::REPRODUCIBLE; }
    const char* name() const override { return m_name.c_str(); }
    std::string param() const override;

    bool is_available(const SizeArgs& args) const override;
    void exec(const ExecArgs& args) const override;

    std::string to_string(AlgoParam algo_param);

protected:
    virtual DTypeEnum src_dtype() const = 0;

    // return filter_ptr, bias_ptr
    virtual std::tuple<void*, void*> prepare_filter_bias(
            const ExecArgs& args) const = 0;

    // return alpha, beta, gamma, delta, theta
    virtual std::tuple<float, float, float, float, float> get_constants(
            const ExecArgs& args) const = 0;

    void reorder_filter(
            const ExecArgs& args, int interleaved, void* reordered_filter) const;

    std::string m_name;
};

class ConvBiasForwardImpl::AlgoInt4Int4NHWCIMMAImplicitGemm final
        : public AlgoInt4NHWCIMMAImplicitGemmBase {
public:
    using Base = AlgoInt4NHWCIMMAImplicitGemmBase;
    using AlgoParam = Base::AlgoParam;

    AlgoInt4Int4NHWCIMMAImplicitGemm(AlgoParam algo_param) : Base{algo_param} {
        m_name = ConvBias::algo_name<ConvBias::DirectParam>(
                ssprintf(
                        "INT4_INT4_NHWC_IMMA_IMPLICIT_GEMM_%s",
                        to_string(m_algo_param).c_str()),
                ConvBias::DirectParam{});
    }

    size_t get_workspace_in_bytes(const SizeArgs& args) const override;
    size_t get_preprocess_workspace_in_bytes(const SizeArgs& args) const override;
    SmallVector<TensorLayout> deduce_preprocessed_filter_layout(
            const SizeArgs& args) const override;
    void exec_preprocess(const ExecArgs& args) const override;

    MEGDNN_DECL_ALGO_TYPE(CUDA_IMPLICIT_GEMM_IMMA_NHWC_INT4_INT4)

private:
    DTypeEnum src_dtype() const override { return DTypeEnum::QuantizedS4; }

    std::tuple<void*, void*> prepare_filter_bias(const ExecArgs& args) const override;

    std::tuple<float, float, float, float, float> get_constants(
            const ExecArgs& args) const override;
};

class ConvBiasForwardImpl::AlgoUInt4Int4NHWCIMMAImplicitGemm final
        : public AlgoInt4NHWCIMMAImplicitGemmBase {
public:
    using Base = AlgoInt4NHWCIMMAImplicitGemmBase;
    using AlgoParam = Base::AlgoParam;

    AlgoUInt4Int4NHWCIMMAImplicitGemm(AlgoParam algo_param) : Base{algo_param} {
        m_name = ConvBias::algo_name<ConvBias::DirectParam>(
                ssprintf(
                        "UINT4_INT4_NHWC_IMMA_IMPLICIT_GEMM_%s",
                        to_string(m_algo_param).c_str()),
                ConvBias::DirectParam{});
    }

    size_t get_workspace_in_bytes(const SizeArgs& args) const override;
    size_t get_preprocess_workspace_in_bytes(const SizeArgs& args) const override;
    SmallVector<TensorLayout> deduce_preprocessed_filter_layout(
            const SizeArgs& args) const override;
    void exec_preprocess(const ExecArgs& args) const override;

    MEGDNN_DECL_ALGO_TYPE(CUDA_IMPLICIT_GEMM_IMMA_NHWC_UINT4_INT4)

private:
    DTypeEnum src_dtype() const override { return DTypeEnum::Quantized4Asymm; }

    std::tuple<void*, void*> prepare_filter_bias(const ExecArgs& args) const override;

    std::tuple<float, float, float, float, float> get_constants(
            const ExecArgs& args) const override;

    void update_bias(
            const ExecArgs& args, void* updated_bias, void* reduce_filter_ptr,
            void* reduce_workspace) const;
};
#endif

class ConvBiasForwardImpl::AlgoFloat32NCHWFMAImplicitBatchedGemm final
        : public AlgoCutlassConvolutionBase {
public:
    AlgoFloat32NCHWFMAImplicitBatchedGemm(AlgoParam algo_param)
            : AlgoCutlassConvolutionBase(algo_param) {
        m_name = ConvBias::algo_name<ConvBias::DirectParam>(
                ssprintf(
                        "FLOAT32_NCHW_FMA_IMPLICIT_BATCHED_GEMM%s",
                        m_algo_param.to_string().c_str()),
                ConvBias::DirectParam{});
    }
    bool is_available(const SizeArgs& args) const override;
    size_t get_workspace_in_bytes(const SizeArgs& /* args */) const override {
        return 0;
    }
    void exec(const ExecArgs& args) const override;
    const char* name() const override { return m_name.c_str(); };
    AlgoAttribute attribute() const override { return AlgoAttribute::REPRODUCIBLE; }
    MEGDNN_DECL_ALGO_TYPE(CUDA_IMPLICIT_BATCHED_GEMM_FMA_NCHW_F32);

private:
    std::string m_name;
};

class ConvBiasForwardImpl::AlgoFloat16NCHWHMMAImplicitBatchedGemm final
        : public AlgoCutlassConvolutionBase {
public:
    AlgoFloat16NCHWHMMAImplicitBatchedGemm(AlgoParam algo_param)
            : AlgoCutlassConvolutionBase(algo_param) {
        m_name = ConvBias::algo_name<ConvBias::DirectParam>(
                ssprintf(
                        "FLOAT16_NCHW_HMMA_IMPLICIT_BATCHED_GEMM%s",
                        m_algo_param.to_string().c_str()),
                ConvBias::DirectParam{});
    }
    bool is_available(const SizeArgs& args) const override;
    size_t get_workspace_in_bytes(const SizeArgs& /* args */) const override {
        return 0;
    }
    void exec(const ExecArgs& args) const override;
    const char* name() const override { return m_name.c_str(); };
    AlgoAttribute attribute() const override { return AlgoAttribute::REPRODUCIBLE; }
    MEGDNN_DECL_ALGO_TYPE(CUDA_IMPLICIT_BATCHED_GEMM_HMMA_NCHW_F16);

private:
    std::string m_name;
};

class ConvBiasForwardImpl::AlgoBFloat16 final : public AlgoBase {
public:
    bool is_available(const SizeArgs& args) const override;
    size_t get_workspace_in_bytes(const SizeArgs& args) const override;
    void exec(const ExecArgs& args) const override;

    std::vector<SearchItem> get_subopr_list(
            const TensorLayoutArray& layouts, const OperatorBase* opr) const override;

    const char* name() const override { return "CONVBIAS_BFLOAT16"; }

    AlgoAttribute attribute() const override { return AlgoAttribute::REPRODUCIBLE; }

    MEGDNN_DECL_ALGO_TYPE(CUDA_BFLOAT16)
private:
    WorkspaceBundle get_workspace_bundle(void* ptr, const SizeArgs& args) const;
};

class ConvBiasForwardImpl::AlgoSimpleInt1 final : public AlgoBase {
public:
    bool is_available(const SizeArgs& args) const override;
    size_t get_workspace_in_bytes(const SizeArgs& args) const override;
    void exec(const ExecArgs& args) const override;

    std::vector<SearchItem> get_subopr_list(
            const TensorLayoutArray& layouts, const OperatorBase* opr) const override;

    const char* name() const override { return "CONVBIAS_SIMPLE_INT1"; }

    AlgoAttribute attribute() const override { return AlgoAttribute::REPRODUCIBLE; }

    MEGDNN_DECL_ALGO_TYPE(CUDA_SIMPLE_INT1)
private:
    WorkspaceBundle get_workspace_bundle(void* ptr, const SizeArgs& args) const;
};

class ConvBiasForwardImpl::AlgoPack : NonCopyableObj {
private:
    AlgoBase::Mapper m_all_algos_map;

public:
    AlgoPack();

    std::vector<AlgoBase*> all_algos,
            //! non-cudnn algos, used for heuristic if cudnn is not supported
            non_cudnn_algos, bfloat16_algos;
    std::vector<AlgoCUDNNConvBiasActivation> cudnn_conv_bias_activations;
    std::vector<AlgoCUDNNConv> cudnn_convs;
    AlgoFallbackNCHWQS8 fallback_nchw_qs8;
    AlgoChanwise chanwise;
    AlgoChanwiseSmall chanwise_small;
    AlgoDepthwiseLargeFilter depthwise_large_filter;
    AlgoChanwise8x8x32 chanwise8x8x32;
    AlgoInplaceMatmul inplace_matmul;
    AlgoMatmul matmul;
    AlgoMatmul8x8x32 matmul8x8x32;
    AlgoBatchedMatmul batched_matmul;
    std::vector<AlgoInt8NCHW4DotProdImplicitGemm> int8_nchw4_dotprod;
    AlgoInt8CHWN4DotProdImplicitGemm int8_chwn4_dotprod;
#if CUDA_VERSION >= 10000
    AlgoQUInt4x4x32WMMA wmma_quint4x4x32;
    std::vector<AlgoInt8CHWN4IMMAImplicitGemm> int8_chwn4_imma;
    std::vector<AlgoInt8NCHW4IMMAImplicitGemm> int8_nchw4_imma;
    std::vector<AlgoInt8CHWN4IMMAImplicitGemmReorderFilter>
            int8_chwn4_imma_reorder_filter;
    std::vector<AlgoInt8CHWN4IMMAImplicitGemmUnrollWidth> int8_chwn4_imma_unroll_width;
#endif
#if CUDA_VERSION >= 10020
    std::vector<AlgoInt8NCHW32IMMAImplicitGemm> int8_nchw32_imma;
    std::vector<AlgoInt8NHWCIMMAImplicitGemm> int8_nhwc_imma;
    std::vector<AlgoInt4Int4NCHW64IMMAImplicitGemm> int4_int4_nchw64_imma;
    std::vector<AlgoUInt4Int4NCHW64IMMAImplicitGemm> uint4_int4_nchw64_imma;
    std::vector<AlgoInt4Int4NHWCIMMAImplicitGemm> int4_int4_nhwc_imma;
    std::vector<AlgoUInt4Int4NHWCIMMAImplicitGemm> uint4_int4_nhwc_imma;
#endif
    std::vector<AlgoFloat32NCHWFMAImplicitBatchedGemm> f32_implicit_bmm;
    std::vector<AlgoFloat16NCHWHMMAImplicitBatchedGemm> f16_implicit_bmm;
    AlgoGroupConvGeneral group;
    AlgoBFloat16 bfloat16;
    AlgoSimpleInt1 int1_simple;

    AlgoBase* cudnn_conv_bias_act_from_enum(cudnnConvolutionFwdAlgo_t algo);

    AlgoBase* cudnn_conv_from_enum(cudnnConvolutionFwdAlgo_t algo);

    const AlgoBase::Mapper& all_algos_map() const { return m_all_algos_map; }

private:
#if CUDA_VERSION >= 10000
    void fill_imma_algos();
#endif
    void fill_cudnn_algos();
    void fill_dp4a_algos();
    void fill_dwconv_algos();
};

}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen
